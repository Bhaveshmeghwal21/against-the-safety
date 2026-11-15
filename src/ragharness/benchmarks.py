"""External benchmarks run through the same harness (plan section 22). Evaluation only: nothing here is trained on or chosen with.

Each loader turns one public dataset into `Question` objects with titled paragraphs, gold support sentences and accepted answers,
so the episode builder, masks, scoring and AuditKIT metrics are the ones used on HotpotQA.

  2wiki    2WikiMultihopQA dev (Ho et al., 2020; Apache-2.0): multi-hop over Wikipedia, four question templates.
  musique  MuSiQue-Ans dev, 2-hop questions (Trivedi et al., 2022; CC BY 4.0): multi-hop built to resist single-hop shortcuts.
  squad2   SQuAD 2.0 dev (Rajpurkar et al., 2018; CC BY-SA 4.0): single-hop, plus human-written questions the paragraph cannot
           answer ("natural unanswerable"), which test abstention without our masking.
  rgb      RGB English, refined (Chen et al., 2024; CC BY-NC-SA 4.0): 2022 news questions with positive and hard negative passages;
           noise robustness (prompt with 5 passages, 0% or 80% negatives) and negative rejection (100% negatives).
  rgb_int  RGB information integration (same source): questions whose answer has two or more parts found in different passages.
"""

from __future__ import annotations

import json
import hashlib
import random
import re
from collections import Counter
from pathlib import Path

from .data import Paragraph, Question, stratified_order

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"\u201c(])")
# Multilingual suites (research/2026-11-15-multilingual-plan.md). Hindi ends a sentence with the danda or double danda; a full stop only
# counts when a Latin capital, digit or quote follows (it also appears in abbreviations such as "डॉ. स्मिथ"). Spanish adds its inverted openers.
_SENT_BY_LANG = {
    "hi": re.compile(r"(?<=[\u0964\u0965!?])\s+|(?<=\.)\s+(?=[A-Z0-9\"\u201c(])"),
    "es": re.compile(r"(?<=[.!?])\s+(?=[A-Z\u00c1\u00c9\u00cd\u00d3\u00da\u00dc\u00d10-9\"\u201c(\u00bf\u00a1\u00ab])"),
    "en": _SENT,
}
MAX_CHARS = 800  # the per-passage display limit of the search tool


def sentences(text: str, lang: str = "") -> list[str]:
    """Split at sentence ends. `lang` ("hi", "es", "en") picks that language's rule; the default is the original English-only regex."""
    pattern = _SENT_BY_LANG.get(lang, _SENT)
    parts = [s.strip() for s in pattern.split(" ".join(text.split())) if s and s.strip()]
    return parts or [text.strip()]


def _find(sents: list[str], forms: list[str]) -> list[int]:
    low = [s.lower() for s in sents]
    for f in forms:
        f = f.strip().lower()
        if len(f) < 2:
            continue
        hits = [i for i, s in enumerate(low) if f in s]
        if hits:
            return hits[:1]
    return []


def _visible(sents: list[str], ids: list[int]) -> bool:
    shown = " ".join(sents)[:MAX_CHARS]
    return bool(ids) and all(sents[i] in shown for i in ids)


# ---------------------------------------------------------------- 2WikiMultihopQA


def load_2wiki(parquet: Path) -> tuple[list[Question], Counter]:
    import pyarrow.parquet as pq

    out, rej = [], Counter()
    for row in pq.read_table(parquet).to_pylist():
        ctx = json.loads(row["context"]) if isinstance(row["context"], str) else row["context"]
        sf = json.loads(row["supporting_facts"]) if isinstance(row["supporting_facts"], str) else row["supporting_facts"]
        paragraphs = [Paragraph(t, [s.strip() for s in ss]) for t, ss in ctx]
        titles = [p.title for p in paragraphs]
        support: dict[str, list[int]] = {}
        for t, sid in sf:
            support.setdefault(t, []).append(int(sid))
        if not support or any(t not in titles for t in support):
            rej["support_title_missing_from_context"] += 1
            continue
        if len(set(titles)) != len(titles):
            rej["duplicate_titles_in_context"] += 1
            continue
        out.append(Question(qid=row["_id"], question=row["question"].strip(), answer=row["answer"].strip(), qtype=row["type"], level="dev",
                            paragraphs=paragraphs, support={t: sorted(set(v)) for t, v in support.items()}))
    return out, rej


# ---------------------------------------------------------------- MuSiQue-Ans (2-hop)


def load_musique(jsonl: Path) -> tuple[list[Question], Counter]:
    out, rej, identities = [], Counter(), set()
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if not r["id"].startswith("2hop") or not r.get("answerable", True):
            rej["not_2hop_answerable"] += 1
            continue
        paras = r["paragraphs"]
        titles = [p["title"] for p in paras]
        aliases = [a for a in r.get("answer_aliases", []) if a and a != r["answer"]]
        support: dict[str, list[int]] = {}
        ok = True
        for k, step in enumerate(r["question_decomposition"]):
            idx = step.get("paragraph_support_idx")
            if idx is None:
                ok = False
                break
            p = paras[idx]
            if titles.count(p["title"]) > 1:
                ok = False  # a support title must name one paragraph only, or masking and citation checks become ambiguous
                break
            forms = [step["answer"]] + (aliases + [r["answer"]] if k == len(r["question_decomposition"]) - 1 else [])
            ids = _find(sentences(p["paragraph_text"]), forms)
            if not ids:
                ok = False
                break
            support.setdefault(p["title"], []).extend(ids)
        if not ok or len(support) < 2:
            rej["support_not_locatable"] += 1
            continue
        identity = (" ".join(r["question"].casefold().split()), " ".join(r["answer"].casefold().split()))
        if identity in identities:
            rej["duplicate_question_answer"] += 1
            continue
        identities.add(identity)
        paragraphs, seen = [], set()
        for p in paras:
            if p["title"] in seen:  # non-support duplicate titles: keep the first only
                continue
            seen.add(p["title"])
            paragraphs.append(Paragraph(p["title"], sentences(p["paragraph_text"])))
        out.append(Question(qid=r["id"], question=r["question"].strip(), answer=r["answer"].strip(), qtype="2hop", level="dev",
                            paragraphs=paragraphs, support={t: sorted(set(v)) for t, v in support.items()}, aliases=aliases))
    return out, rej


# ---------------------------------------------------------------- SQuAD 2.0


def annotated_span(context: str, answer: str, offset: int, lang: str = ""):
    """Map the official source offset through whitespace normalization and splitting."""
    if offset < 0 or context[offset:offset + len(answer)] != answer:
        return None
    offset += len(answer) - len(answer.lstrip())
    answer = answer.strip()
    if not answer:
        return None
    normalized = " ".join(context.split())
    start = len(" ".join(context[:offset].split()))
    if start and context[offset - 1].isspace():
        start += 1
    form = " ".join(answer.split())
    if normalized[start:start + len(form)] != form:
        return None
    parts = sentences(context, lang)
    cursor = 0
    for i, sentence in enumerate(parts):
        end = cursor + len(sentence)
        if cursor <= start and start + len(form) <= end:
            return i, [start, start + len(form)]
        cursor = end + 1
    return None


def load_squad2(parquet: Path) -> tuple[list[Question], list[Question], Counter]:
    """Returns (questions, corpus_questions). Every dev paragraph goes into the corpus, titled "<article> (part n)" so that a
    mask hides one paragraph and not the whole article. One question per paragraph and kind (answerable / unanswerable)."""
    import pyarrow.parquet as pq

    rows = pq.read_table(parquet).to_pylist()
    part: dict[str, int] = {}
    per_article: Counter = Counter()
    for r in rows:
        if r["context"] not in part:
            per_article[r["title"]] += 1
            part[r["context"]] = per_article[r["title"]]
    corpus_q, out, rej, used = {}, [], Counter(), set()
    for r in rows:
        title = f"{r['title'].replace('_', ' ')} (part {part[r['context']]})"
        para = Paragraph(title, sentences(r["context"]))
        corpus_q.setdefault(r["context"], Question(qid=f"ctx-{len(corpus_q)}", question="", answer="", qtype="paragraph", level="dev", paragraphs=[para]))
        texts = [t.strip() for t in r["answers"]["text"]]
        kind = "answerable" if texts else "unanswerable"
        if (r["context"], kind) in used:
            continue
        if texts:
            offsets = r["answers"].get("answer_start", [])
            spans = [annotated_span(r["context"], raw, int(offset))
                     for raw, offset in zip(r["answers"]["text"], offsets)]
            if not spans or spans[0] is None:
                rej["invalid_official_answer_offset"] += 1
                continue
            sid, span = spans[0]
            from .retrieval import visible_text
            displayed = visible_text(para.text, MAX_CHARS)
            if span[1] > len(displayed.removesuffix(" ...")) or para.sentences[sid] not in displayed:
                rej["annotated_answer_beyond_display"] += 1
                continue
            aliases = sorted({t for t in texts if t != texts[0]})
            q = Question(qid=r["id"], question=r["question"].strip(), answer=texts[0], qtype="answerable", level="dev",
                         paragraphs=[para], support={title: [sid]}, aliases=aliases,
                         evidence_scope="paragraph", support_spans={title: [span]})
        else:
            q = Question(qid=r["id"], question=r["question"].strip(), answer="", qtype="unanswerable", level="dev",
                         paragraphs=[para], natural_unanswerable=True, evidence_scope="paragraph")
        used.add((r["context"], kind))
        out.append(q)
    return out, list(corpus_q.values()), rej


# ---------------------------------------------------------------- XQuAD (multilingual check)


def xquad_passages(parts: list[str], max_chars: int = MAX_CHARS) -> list[list[int]]:
    """Group consecutive sentence indices into passages whose joined text is at most `max_chars`; a longer sentence stands alone."""
    groups: list[list[int]] = []
    length = 0
    for i, sentence in enumerate(parts):
        if groups and length + 1 + len(sentence) <= max_chars:
            groups[-1].append(i)
            length += 1 + len(sentence)
        else:
            groups.append([i])
            length = len(sentence)
    return groups


def load_xquad(parquet: Path, lang: str) -> tuple[list[Question], list[Question], Counter]:
    """XQuAD validation in one language (Artetxe et al., 2020; CC BY-SA 4.0): returns (questions, passage holders, rejections).

    Every paragraph is split into passages of at most 800 characters (the search tool's display limit). A passage is titled with its own
    first eight words, in its own language, so no English title appears in Hindi or Spanish documents. A question's support is the passage
    holding its official answer; the question is rejected when the answer span crosses a sentence end or is cut by the display limit.
    """
    import pyarrow.parquet as pq

    return load_squad_rows(pq.read_table(parquet).to_pylist(), lang)


def load_squad_rows(rows: list[dict], lang: str) -> tuple[list[Question], list[Question], Counter]:
    """The body of `load_xquad` for any SQuAD-format rows (`id`, `context`, `question`, `answers` with `text` and `answer_start`).

    The Hindi training and development sets (scripts/prepare_hindi.py, IndicSQuAD) are read through the same passage splitting, titles and
    display checks as the XQuAD test sets, so what a model is trained on looks like what it is tested on."""
    from .retrieval import visible_text

    rej: Counter = Counter()
    by_context: dict[str, list[tuple[Paragraph, int, int]]] = {}  # context -> [(passage, first sentence index, start offset)] in order
    sentence_passage: dict[str, list[tuple[int, int]]] = {}  # context -> per sentence (passage index, local sentence index)
    holders: list[Question] = []
    used_titles: Counter = Counter()
    for r in rows:
        ctx = r["context"]
        if ctx in by_context:
            continue
        parts = sentences(ctx, lang)
        starts, cursor = [], 0
        for s in parts:
            starts.append(cursor)
            cursor += len(s) + 1
        passages, where = [], [(0, 0)] * len(parts)
        for pi, group in enumerate(xquad_passages(parts)):
            title = " ".join(" ".join(parts[i] for i in group).split()[:8])
            used_titles[title.casefold()] += 1
            if used_titles[title.casefold()] > 1:
                rej["title_collision_suffixed"] += 1
                title = f"{title} ({used_titles[title.casefold()]})"
            para = Paragraph(title, [parts[i] for i in group])
            passages.append((para, group[0], starts[group[0]]))
            for li, i in enumerate(group):
                where[i] = (pi, li)
            holders.append(Question(qid=f"ctx-{len(holders)}", question="", answer="", qtype="paragraph", level="dev", paragraphs=[para], lang=lang))
        by_context[ctx], sentence_passage[ctx] = passages, where
    out = []
    for r in rows:
        texts = [t for t in r["answers"]["text"]]
        offsets = r["answers"].get("answer_start", [])
        if not texts or not offsets:
            rej["no_answer"] += 1
            continue
        span = annotated_span(r["context"], texts[0], int(offsets[0]), lang)
        if span is None:
            rej["invalid_official_answer_offset"] += 1
            continue
        sid, (start, end) = span
        pi, li = sentence_passage[r["context"]][sid]
        para, _, passage_start = by_context[r["context"]][pi]
        local = [start - passage_start, end - passage_start]
        displayed = visible_text(para.text, MAX_CHARS)
        if local[1] > len(displayed.removesuffix(" ...")) or para.sentences[li] not in displayed:
            rej["annotated_answer_beyond_display"] += 1
            continue
        aliases = sorted({t.strip() for t in texts[1:] if t.strip()} - {texts[0].strip()})
        out.append(Question(qid=r["id"], question=r["question"].strip(), answer=texts[0].strip(), qtype="single", level="dev",
                            paragraphs=[para], support={para.title: [li]}, aliases=aliases, evidence_scope="corpus",
                            support_spans={para.title: [local]}, lang=lang))
    return out, holders, rej


# ---------------------------------------------------------------- RGB


def _parts(answer) -> list[list[str]]:
    """RGB's answer list: each element is one required part, given as a string or a list of accepted forms."""
    return [[a] if isinstance(a, str) else [x for x in a if isinstance(x, str)] for a in answer]


def _title(text: str, n: int = 9) -> str:
    words = text.split()
    digest = hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()
    return " ".join(words[:n]) + (" ..." if len(words) > n else "") + f" [{digest}]"


def load_rgb(path: Path, integration: bool) -> tuple[list[Question], Counter]:
    """Positive passages hold the answer; negatives are retrieved passages about the same query that do not.
    Titles are the passage's first words (RGB passages have none). Support is chosen later, after the index exists."""
    out, rej = [], Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        parts = _parts(r["answer"])
        if integration:
            groups = [g if isinstance(g, list) else [g] for g in r["positive"]]
        else:
            groups = [list(r["positive"])]
        if not parts or any(not p for p in parts):
            rej["empty_answer"] += 1
            continue
        paragraphs, seen = [], set()
        pos_titles: list[list[str]] = []
        for g in groups:
            pt = []
            for text in g:
                t = _title(text)
                if not text.strip():
                    continue
                if t in seen:
                    if t not in pt:
                        pt.append(t)
                    continue
                seen.add(t)
                paragraphs.append(Paragraph(t, sentences(text)))
                pt.append(t)
            pos_titles.append(pt)
        neg_titles = []
        for text in r["negative"]:
            t = _title(text)
            if t in seen or not text.strip():
                continue
            seen.add(t)
            paragraphs.append(Paragraph(t, sentences(text)))
            neg_titles.append(t)
        q = Question(qid=f"rgb{'int' if integration else ''}-{r['id']}", question=r["query"].strip(), answer="; ".join(p[0] for p in parts),
                     qtype="integration" if integration else "single", level="news", paragraphs=paragraphs, answer_parts=parts)
        q._pos_titles, q._neg_titles = pos_titles, neg_titles  # used by choose_rgb_support, not saved
        out.append(q)
    return out, rej


def choose_rgb_support(q: Question, search, by_title: dict[str, str]) -> tuple[dict[str, list[int]], list[str]] | None:
    """Support = for each answer part, the positive passage that ranks highest for the question and shows that part within the
    display limit. Every other positive becomes an extra copy (masked with the support, accepted as a gold citation)."""
    ranked = [h.chunk_id for h in search.search(q.question, 60)]
    rank = {c: i for i, c in enumerate(ranked)}
    paras = {p.title: p for p in q.paragraphs}
    groups = q._pos_titles if q.qtype == "integration" else [q._pos_titles[0]]
    part_lists = q.answer_parts if q.qtype == "integration" else [[a for part in q.answer_parts for a in part]]
    if q.qtype == "integration" and len(groups) != len(part_lists):
        return None
    support: dict[str, list[int]] = {}
    for titles, forms in zip(groups, part_lists):
        cands = []
        for t in titles:
            ids = _find(paras[t].sentences, forms)
            if _visible(paras[t].sentences, ids):
                cands.append((rank.get(by_title[t], 10**6), t, ids))
        if not cands:
            return None
        _, t, ids = min(cands)
        support[t] = ids
    extra = [by_title[t] for g in q._pos_titles for t in g if t not in support]
    return support, extra


def rgb_prefill_variants(q: Question, by_title: dict[str, str], seed: int) -> dict[str, dict]:
    """RGB's own protocol inside our harness: five passages in the user turn.

    Single-answer questions (en_refine): noise_0 (all positive), noise_0.8 (one positive, four negatives): answer from the
    passages; noise_1 (all negative, the positives also masked from search): RGB's negative rejection, the correct ending
    is an abstention.
    Integration questions (en_int): noise_0 (positives only, at least one per answer part) and noise_0.4 (one positive per part
    plus positives to three, then two negatives), the two lowest noise rates of RGB's integration test."""
    rng = random.Random(f"{seed}|{q.qid}")
    paras = {p.title: p for p in q.paragraphs}
    neg = [t for t in q._neg_titles if not _find(paras[t].sentences, [a for part in q.answer_parts for a in part])]
    rng.shuffle(neg)
    allpos = [by_title[t] for g in q._pos_titles for t in g]
    out = {}
    if q.qtype == "integration":
        if len(q._pos_titles) != len(q.answer_parts):
            return {}
        per_part = []
        for titles, forms in zip(q._pos_titles, q.answer_parts):
            ok = [t for t in titles if _visible(paras[t].sentences, _find(paras[t].sentences, forms))]
            if not ok:
                return {}
            rng.shuffle(ok)
            per_part.append(ok)
        firsts = [p[0] for p in per_part]
        rest = [t for p in per_part for t in p[1:]]
        if len(firsts) > 3:
            return {}
        if len(firsts) + len(rest) >= 5:
            out["rgb_int_noise_0"] = {"prefill": [by_title[t] for t in (firsts + rest)[:5]], "masked": [], "answerable": True}
        if len(neg) >= 2:
            three = (firsts + rest)[:3]
            if len(three) == 3:
                out["rgb_int_noise_0.4"] = {"prefill": [by_title[t] for t in three + neg[:2]], "masked": [], "answerable": True}
    else:
        forms = [a for part in q.answer_parts for a in part]
        pos = [t for g in q._pos_titles for t in g if _visible(paras[t].sentences, _find(paras[t].sentences, forms))]
        rng.shuffle(pos)
        if len(pos) >= 1 and len(neg) >= 4:
            if len(pos) >= 5:
                out["rgb_noise_0"] = {"prefill": [by_title[t] for t in pos[:5]], "masked": [], "answerable": True}
            out["rgb_noise_0.8"] = {"prefill": [by_title[t] for t in pos[:1] + neg[:4]], "masked": [], "answerable": True}
        if len(neg) >= 5:
            out["rgb_noise_1"] = {"prefill": [by_title[t] for t in neg[:5]], "masked": allpos, "answerable": False}
    for v in out.values():
        rng.shuffle(v["prefill"])
    return out


# ---------------------------------------------------------------- sampling


def sample(questions: list[Question], n: int, seed: int, stratify: bool = True) -> list[Question]:
    order = stratified_order(questions, seed) if stratify else sorted(questions, key=lambda q: q.qid)
    if not stratify:
        random.Random(seed).shuffle(order)
    return order[:n]
