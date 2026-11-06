"""Round 3 data (plan section 24): the three behaviours the external benchmarks showed to be missing, and the patch mixtures.

Round 1's questions all needed two passages with a single-span answer, so "the first search shows one relevant passage" only ever
meant "search again" (state S3), and a model that could not find a second passage abstained. Three families of new episodes,
all built with the scripted policy under real search, as before:

  single    SQuAD 2.0 train paragraphs (CC BY-SA 4.0), one passage holds the answer. Variants: open (S1/S2/S6), support masked (S4),
            first search masked (S6), evidence supplied (S5), plus the two prompt variants below.
  compound  two single-passage questions from one article joined into one question, answer "A1; A2" (the form RGB's integration
            set uses). Variants: open, both supports masked, ONE support masked (part_masked: the correct ending is an abstention),
            plus the prompt variants.
  hotpot    the existing HotpotQA train questions, with the prompt variants only.

Prompt variants (the same function builds the training episodes and the evaluation episodes):
  prefill_noisy      the support plus three on-topic passages that do not hold the answer, supplied in the user turn: answer, no search.
  prefill_negatives  three to five on-topic passages that do not hold the answer supplied, every support copy masked from search:
                     search at least once, then abstain.
  part_masked        compound only: one of the two support paragraphs masked from search, nothing supplied: abstain.

Nothing here reads a model output. Natural unanswerable SQuAD questions are never used for training (they stay a transfer test).
"""

from __future__ import annotations

import hashlib
import random
import re

from . import builder, episodes
from .corpus import Corpus
from .data import Paragraph, Question
from .retrieval import MaskedSearch
from .tools import CorpusSearchTool

_ARTICLE = re.compile(r" \(part \d+\)$")
_LEAD = frozenset("what who whom whose which when where why how in on at by for from during after before to with of under about as into since until through".split())


def article_of(title: str) -> str:
    return _ARTICLE.sub("", title)


def _h(*parts: str) -> int:
    return int(hashlib.sha1("|".join(parts).encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------- questions


def eligible_single(q: Question) -> bool:
    return (not q.natural_unanswerable and bool(q.support) and q.question.endswith("?") and 1 <= len(q.answer) <= 60 and len(q.question.split()) >= 4)


def split_articles(articles: list[str], seed: int, n_dev: int, n_train: int) -> tuple[list[str], list[str]]:
    """Seeded order over the sorted article names: the first n_dev are development articles, the next n_train are training articles."""
    names = sorted(set(articles))
    random.Random(seed).shuffle(names)
    return names[:n_dev], names[n_dev:n_dev + n_train]


def pick_singles(qs: list[Question], articles: list[str], n: int, seed: int) -> list[Question]:
    """Up to n eligible single-passage questions, spread over the articles (round-robin)."""
    by_art: dict[str, list[Question]] = {a: [] for a in articles}
    for q in sorted(qs, key=lambda q: q.qid):
        a = article_of(q.paragraphs[0].title)
        if a in by_art and eligible_single(q):
            by_art[a].append(q)
    rng = random.Random(seed)
    for v in by_art.values():
        rng.shuffle(v)
    order = list(by_art)
    rng.shuffle(order)
    out: list[Question] = []
    while len(out) < n and any(by_art.values()):
        for a in order:
            if by_art[a] and len(out) < n:
                out.append(by_art[a].pop())
    return out


def join_questions(a: str, b: str) -> str | None:
    a, b = a.strip(), b.strip()
    if not (a.endswith("?") and b.endswith("?")) or b.split()[0].lower() not in _LEAD:
        return None
    return f"{a[:-1]}, and {b[0].lower()}{b[1:]}"


def make_compound(q1: Question, q2: Question) -> Question | None:
    text = join_questions(q1.question, q2.question)
    t1, t2 = q1.paragraphs[0].title, q2.paragraphs[0].title
    if text is None or t1 == t2 or q1.answer.lower() == q2.answer.lower():
        return None
    return Question(
        qid=f"cmp-{q1.qid}-{q2.qid}", question=text, answer=f"{q1.answer}; {q2.answer}", qtype="compound", level="round3",
        paragraphs=[q1.paragraphs[0], q2.paragraphs[0]], support={t1: list(q1.support[t1]), t2: list(q2.support[t2])},
        answer_parts=[[q1.answer, *q1.aliases], [q2.answer, *q2.aliases]],
    )


def pick_compounds(singles_pool: list[Question], articles: list[str], n: int, seed: int) -> list[Question]:
    """Pairs of single-passage questions from different paragraphs of one article; no question is used in two pairs."""
    by_art: dict[str, list[Question]] = {a: [] for a in articles}
    for q in sorted(singles_pool, key=lambda q: q.qid):
        a = article_of(q.paragraphs[0].title)
        if a in by_art and eligible_single(q):
            by_art[a].append(q)
    rng = random.Random(seed)
    per_art: dict[str, list[Question]] = {}
    for a, qs in by_art.items():
        rng.shuffle(qs)
        pairs = [c for c in (make_compound(qs[i], qs[i + 1]) for i in range(0, len(qs) - 1, 2)) if c is not None]
        per_art[a] = pairs
    order = list(per_art)
    rng.shuffle(order)
    out: list[Question] = []
    while len(out) < n and any(per_art.values()):
        for a in order:
            if per_art[a] and len(out) < n:
                out.append(per_art[a].pop())
    return out


# ---------------------------------------------------------------- prompt variants


def hard_negatives(q: Question, corp: Corpus, search: MaskedSearch, copies: frozenset[str], n: int = 5) -> list[str]:
    """Top-ranked passages for the question that are not a support copy and show no accepted form of the answer."""
    out = []
    for h in search.search(q.question, 40):
        if h.chunk_id in copies or builder.answer_visible(q, h.text):
            continue
        out.append(h.chunk_id)
        if len(out) == n:
            break
    return out


def _shuffled(qid: str, name: str, ids: list[str]) -> list[str]:
    return sorted(ids, key=lambda c: hashlib.sha1(f"{qid}|{name}|{c}".encode()).hexdigest())


def prompt_variants(q: Question, corp: Corpus, search: MaskedSearch) -> dict[str, dict]:
    """{variant: {"prefill": [chunk ids], "masked": [chunk ids], "answerable": bool}}; the format `evaluation.prompt_jobs` reads."""
    support = corp.support_chunks(q)
    copies = corp.support_copies(q)
    negs = hard_negatives(q, corp, search, copies)
    out: dict[str, dict] = {}
    if len(negs) >= 3:
        k = 3 + _h(q.qid, "k") % 3  # three to five passages, so the number of supplied passages does not reveal the label
        out["prefill_noisy"] = {"prefill": _shuffled(q.qid, "noisy", list(support.values()) + negs[:3]), "masked": [], "answerable": True}
        out["prefill_negatives"] = {"prefill": _shuffled(q.qid, "neg", negs[:min(k, len(negs))]), "masked": sorted(copies), "answerable": False}
    if q.qtype == "compound" and len(support) == 2:
        titles = sorted(support)
        hidden = titles[_h(q.qid, "hidden") % 2]
        out["part_masked"] = {"prefill": [], "masked": sorted(corp.title_copies(hidden, support[hidden])), "answerable": False, "hidden_title": hidden}
    return out


STATE = {"prefill_noisy": "S7", "prefill_negatives": "S8", "part_masked": "S9"}


def build_extras(q: Question, split: str, corp: Corpus, tool: CorpusSearchTool, search: MaskedSearch, budget: int = 3) -> list[builder.Built]:
    """Scripted episodes for the prompt variants, with the same rejection rules as the first-round builder."""
    if not builder.support_visible(q, corp, tool.max_chars):
        return []
    out: list[builder.Built] = []
    support = corp.support_chunks(q)
    for variant, v in prompt_variants(q, corp, search).items():
        prefill = [builder._hit(corp, c) for c in v["prefill"]]
        masked = frozenset(v["masked"])
        finish = "answer_if_complete" if v["answerable"] else "abstain"
        ep = episodes.run_scripted(q, corp, tool, variant, masked, budget, frozenset(), prefill, finish)
        seen = [h.text for h in ep.retrieved_chunks()]
        reason = None
        if v["answerable"]:
            if not ep.complete:
                reason = "support_not_complete"
            elif ep.n_searches != 0:
                reason = "searched_despite_sufficient_prefill"
        else:
            if ep.complete:
                reason = "masked_support_was_observed"
            elif ep.n_searches < 1:
                reason = "no_search_before_abstain"
            elif variant == "part_masked":
                forms = _hidden_forms(q, v["hidden_title"])
                visible_part = [c for t, c in support.items() if t != v["hidden_title"]]
                if any(builder.answer_in_text(a, t) is True for a in forms for t in seen):
                    reason = "alternate_evidence_path:hidden_part_answer_in_visible_passage"
                elif not all(c in {h.chunk_id for h in ep.retrieved_chunks()} for c in visible_part):
                    reason = "the_visible_part_was_not_found"  # otherwise the episode is the same as both parts masked
            elif any(builder.answer_visible(q, t) for t in seen):
                reason = "alternate_evidence_path:answer_text_in_visible_passage"
        if reason:
            out.append(builder.Built(q.qid, split, variant, "rejected", v["answerable"], None, reason))
        else:
            out.append(builder.Built(q.qid, split, variant, STATE[variant], v["answerable"], ep, flags={"n_searches": ep.n_searches}))
    return out


def _hidden_forms(q: Question, hidden_title: str) -> list[str]:
    """Accepted forms of the answer part that lives in the hidden paragraph (compound questions: parts follow the support order)."""
    idx = list(q.support).index(hidden_title)
    return [a for a in q.answer_parts[idx] if a]


# ---------------------------------------------------------------- mixtures

SEARCH_SHARE = 1 / 3

# The two patch mixtures (plan section 24). Both continue from the main adapter for one epoch over the same processed-token budget.
PATCH_TOKENS = 2_200_000
FAMILY_SHARE = {"single": 0.40, "compound": 0.30, "hotpot": 0.30}
PATCHES = {
    "candC": {"new_share": 0.65, "abstain_new": 0.35, "abstain_replay": 0.35},  # stronger: more new rows, fewer abstentions
    "candD": {"new_share": 0.45, "abstain_new": 0.45, "abstain_replay": 0.45},  # careful: more rehearsal, abstentions close to main's 0.50
}


def kind_fractions(abstain_share_of_finals: float) -> dict[str, float]:
    finals = 1 - SEARCH_SHARE
    return {"search": SEARCH_SHARE, "answer": finals * (1 - abstain_share_of_finals), "abstain": finals * abstain_share_of_finals}


def build_patch(new_rows: list[dict], replay_rows: list[dict], seed: int, cap_tokens: int, target_tokens: int, new_share: float,
                abstain_new: float, abstain_replay: float, family_share: dict[str, float]) -> tuple[list[dict], dict]:
    """A patch set: `new_share` of the rows are new-family decisions (split over families by `family_share`), the rest are rehearsal
    rows, decisions the main adapter was already trained on. Within each part, search : answer : abstain follows `abstain_*`."""
    from . import mixture

    assert abs(sum(family_share.values()) - 1) < 1e-9
    rng = random.Random(seed)
    ok = lambda rows: [r for r in rows if r["meta"]["est_tokens"] <= cap_tokens]
    groups: dict[str, list[dict]] = {}
    frac: dict[str, float] = {}
    for fam, fs in family_share.items():
        pool = [r for r in ok(new_rows) if r["meta"]["family"] == fam]
        for kind, kf in kind_fractions(abstain_new).items():
            groups[f"new_{fam}_{kind}"] = mixture._round_robin([r for r in pool if r["meta"]["kind"] == kind], rng, 10**9)
            frac[f"new_{fam}_{kind}"] = new_share * fs * kf
    pool = ok(replay_rows)
    for kind, kf in kind_fractions(abstain_replay).items():
        groups[f"replay_{kind}"] = mixture._round_robin([r for r in pool if r["meta"]["kind"] == kind], rng, 10**9)
        frac[f"replay_{kind}"] = (1 - new_share) * kf
    chosen, info = mixture._fill_to_budget(groups, frac, target_tokens)
    rng.shuffle(chosen)
    return chosen, info
