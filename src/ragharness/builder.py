"""Matched-state episode builder (plan sections 4.2 and 5.1).

For one question it produces up to five scripted episodes under real search, differing only by masks or
pre-supplied evidence, and labels each with one of the six states:

  S1 clean         full support after one search, support ranked above every non-support result
  S2 distractors   full support after one search, but a non-support result outranks a support passage
  S3 second hop    first search returns some support, a visible-text follow-up query recovers the rest
  S4 missing       every support copy masked; search within budget, then abstain
  S5 evidence given  support supplied in the user turn: answer with no search (prefill_sufficient), or
                   only part supplied: search for the rest (prefill_partial)
  S6 look-alike    first search returns no support, a reformulated query recovers it, then answer
                   (naturally weak first results, or the support masked for the first search only)

Episodes whose support is never reached within the budget are "unreachable": reported, never trained on.
The scripted policy is the same for all variants, so a label can only come from what is visible at each turn.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from .corpus import Corpus
from .data import Question
from .episodes import Episode, run_scripted
from .retrieval import Hit, MaskedSearch, visible_text
from .tools import CorpusSearchTool

VARIANTS = ("open", "support_masked", "first_search_masked", "prefill_sufficient", "prefill_partial", "prefill_partial_masked")
_YESNO = {"yes", "no"}


@dataclass
class Spec:
    variant: str
    masked: frozenset[str] = frozenset()
    first_search_masked: frozenset[str] = frozenset()
    prefill: list[Hit] = field(default_factory=list)
    answerable: bool = True
    finish: str = "answer_if_complete"


@dataclass
class Built:
    qid: str
    split: str
    variant: str
    state: str  # S1..S6, or "unreachable" / "rejected"
    answerable: bool
    episode: Episode | None
    reject_reason: str | None = None
    flags: dict = field(default_factory=dict)


def _hit(corpus: Corpus, cid: str) -> Hit:
    c = corpus.chunk(cid)
    return Hit(cid, c["title"], c["text"], 0.0)


def _order(qid: str, hits: list[Hit]) -> list[Hit]:
    return sorted(hits, key=lambda h: hashlib.sha1(f"{qid}|{h.chunk_id}".encode()).hexdigest())


def answer_in_text(answer: str, text: str) -> bool | None:
    """Whole-word, case-insensitive match. None when the check is meaningless (yes/no or very short answers)."""
    a = " ".join(answer.split())
    if a.lower() in _YESNO or len(a) < 3:
        return None
    return re.search(rf"(?<!\w){re.escape(a)}(?!\w)", text, flags=re.IGNORECASE) is not None


def gold_forms(q: Question) -> list[str]:
    """Every accepted form of the answer: just `q.answer` for HotpotQA; benchmarks add aliases and RGB answer parts."""
    return [q.answer] + list(q.aliases) + [a for part in q.answer_parts for a in part]


def answer_visible(q: Question, text: str) -> bool:
    return any(answer_in_text(a, text) is True for a in gold_forms(q) if a)


def make_specs(q: Question, corpus: Corpus, search: MaskedSearch) -> dict[str, Spec]:
    support = corpus.support_chunks(q)
    support_ids = set(support.values())
    copies = corpus.support_copies(q)
    ranked = search.search(q.question, len(copies) + 4)
    extras = [h for h in ranked if h.chunk_id not in copies]
    distractor = extras[0] if extras else None

    specs = {
        "open": Spec("open"),
        "support_masked": Spec("support_masked", masked=copies, answerable=False, finish="abstain"),
        "first_search_masked": Spec("first_search_masked", first_search_masked=copies),
    }
    # Both prefill variants show |support|+1 passages, so passage count does not reveal sufficiency.
    if distractor is not None:
        full = [_hit(corpus, c) for c in support.values()] + [distractor]
        specs["prefill_sufficient"] = Spec("prefill_sufficient", prefill=_order(q.qid, full))
        # evaluation only (never trained on): the same evidence without the distractor, for the paired distractor effect
        specs["prefill_clean"] = Spec("prefill_clean", prefill=_order(q.qid, [_hit(corpus, c) for c in support.values()]))
        if len(support) >= 2 and len(extras) >= 2:
            # leave out the support paragraph whose title the question does not mention (the hop-2 page)
            from .policy import contains, mention

            hidden = next((t for t in support if not contains(q.question, mention(t))), list(support)[-1])
            part = [_hit(corpus, c) for t, c in support.items() if t != hidden] + extras[:2]
            specs["prefill_partial"] = Spec("prefill_partial", prefill=_order(q.qid, part))
            # identical user turn, but the missing hop cannot be retrieved: the correct ending is an abstention
            specs["prefill_partial_masked"] = Spec(
                "prefill_partial_masked", masked=corpus.title_copies(hidden, support[hidden]), prefill=_order(q.qid, part),
                answerable=False, finish="abstain",
            )
    assert support_ids <= copies
    return specs


def classify_open(ep: Episode, support_ids: set[str]) -> str:
    first = ep.steps[0].hits
    found_first = [h for h in first if h.chunk_id in support_ids]
    if ep.n_searches == 1:
        ranks = [i for i, h in enumerate(first) if h.chunk_id in support_ids]
        clean = ranks == list(range(len(ranks)))  # support occupies the top ranks
        return "S1" if clean else "S2"
    return "S3" if found_first else "S6"


def _same_trace(a: Episode, b: Episode) -> bool:
    return [(s.text, s.observation) for s in a.steps] == [(s.text, s.observation) for s in b.steps]


def support_visible(q: Question, corpus: Corpus, max_chars: int | None) -> bool:
    """True when every gold supporting sentence survives the display limit (plan section 4.1)."""
    for title, sents in q.support_sentences().items():
        shown = visible_text(corpus.chunk(corpus.support_chunks(q)[title])["text"], max_chars)
        if any(s not in shown for s in sents):
            return False
    return True


def build_question(
    q: Question, split: str, corpus: Corpus, tool: CorpusSearchTool, search: MaskedSearch, budget: int = 3
) -> list[Built]:
    support = corpus.support_chunks(q)
    support_ids = set(support.values())
    if not support_visible(q, corpus, tool.max_chars):
        # the whole question is skipped so that matched variants stay matched
        return [Built(q.qid, split, "open", "rejected", True, None, "support_sentence_cut_by_display_limit")]
    specs = make_specs(q, corpus, search)
    out: list[Built] = []

    def run(spec: Spec) -> Episode:
        return run_scripted(q, corpus, tool, spec.variant, spec.masked, budget, spec.first_search_masked, spec.prefill, spec.finish)

    open_ep = run(specs["open"])
    if not open_ep.complete:
        out.append(Built(q.qid, split, "open", "unreachable", True, None, "support_not_reached_within_budget",
                         {"stop": open_ep.stop_reason, "n_searches": open_ep.n_searches}))
        return out  # matched variants are only built for reachable questions
    state = classify_open(open_ep, support_ids)
    out.append(Built(q.qid, split, "open", state, True, open_ep, flags={"n_searches": open_ep.n_searches}))

    # S4: all support copies masked
    ep = run(specs["support_masked"])
    assert not ep.complete, "a masked support chunk was still observed"
    seen = [h.text for h in ep.retrieved_chunks()]
    alt = [answer_in_text(q.answer, t) for t in seen]
    if any(a is True for a in alt) or any(answer_visible(q, t) for t in seen):
        out.append(Built(q.qid, split, "support_masked", "rejected", False, None, "alternate_evidence_path:answer_text_in_observed_passage"))
    else:
        out.append(Built(q.qid, split, "support_masked", "S4", False, ep,
                         flags={"answer_check": "skipped_yes_no_or_short" if all(a is None for a in alt) else "ok", "n_searches": ep.n_searches}))

    # S6 by masking the support for the first search only
    ep = run(specs["first_search_masked"])
    if not ep.complete:
        out.append(Built(q.qid, split, "first_search_masked", "unreachable", True, None, "not_recovered_after_first_search_mask"))
    elif _same_trace(ep, open_ep):
        out.append(Built(q.qid, split, "first_search_masked", "rejected", True, None, "identical_to_open_episode"))
    elif ep.n_searches < 2:
        out.append(Built(q.qid, split, "first_search_masked", "rejected", True, None, "answered_without_second_search"))
    else:
        out.append(Built(q.qid, split, "first_search_masked", "S6", True, ep, flags={"n_searches": ep.n_searches}))

    # S5: evidence supplied in the user turn
    for name, want_searches in (("prefill_sufficient", 0), ("prefill_partial", None)):
        if name not in specs:
            out.append(Built(q.qid, split, name, "rejected", True, None, "no_distractor_or_single_support_paragraph"))
            continue
        ep = run(specs[name])
        if not ep.complete:
            out.append(Built(q.qid, split, name, "unreachable", True, None, "missing_support_not_recovered"))
        elif want_searches == 0 and ep.n_searches != 0:
            out.append(Built(q.qid, split, name, "rejected", True, None, "searched_despite_sufficient_prefill"))
        elif want_searches is None and ep.n_searches < 1:
            out.append(Built(q.qid, split, name, "rejected", True, None, "no_search_needed_for_partial_prefill"))
        else:
            out.append(Built(q.qid, split, name, "S5", True, ep, flags={"n_searches": ep.n_searches}))

    if "prefill_partial_masked" in specs:
        ep = run(specs["prefill_partial_masked"])
        assert not ep.complete, "a masked support chunk was still observed"
        seen = [h.text for h in ep.retrieved_chunks()]  # includes the pre-supplied passages
        if any(answer_in_text(q.answer, t) is True for t in seen) or any(answer_visible(q, t) for t in seen):
            out.append(Built(q.qid, split, "prefill_partial_masked", "rejected", False, None, "alternate_evidence_path:answer_text_in_visible_passage"))
        elif ep.n_searches < 1:
            out.append(Built(q.qid, split, "prefill_partial_masked", "rejected", False, None, "no_search_before_abstain"))
        else:
            out.append(Built(q.qid, split, "prefill_partial_masked", "S4", False, ep, flags={"n_searches": ep.n_searches}))
    return out
