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
import random
import re
from collections import Counter
from pathlib import Path

from .data import Paragraph, Question, stratified_order

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"\u201c(])")
MAX_CHARS = 800  # the per-passage display limit of the search tool


def sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT.split(" ".join(text.split())) if s.strip()]
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
    out, rej = [], Counter()
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
            ids = _find(para.sentences, texts)
            if not ids:
                rej["answer_sentence_not_found"] += 1
                continue
            aliases = sorted({t for t in texts if t != texts[0]})
            q = Question(qid=r["id"], question=r["question"].strip(), answer=texts[0], qtype="answerable", level="dev",
                         paragraphs=[para], support={title: ids}, aliases=aliases)
        else:
            q = Question(qid=r["id"], question=r["question"].strip(), answer="", qtype="unanswerable", level="dev",
                         paragraphs=[para], natural_unanswerable=True)
        used.add((r["context"], kind))
        out.append(q)
    return out, list(corpus_q.values()), rej


# ---------------------------------------------------------------- RGB


def _parts(answer) -> list[list[str]]:
    """RGB's answer list: each element is one required part, given as a string or a list of accepted forms."""
    return [[a] if isinstance(a, str) else [x for x in a if isinstance(x, str)] for a in answer]


def _title(text: str, n: int = 9) -> str:
    words = text.split()
    return " ".join(words[:n]) + (" ..." if len(words) > n else "")


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
                if t in seen or not text.strip():
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
        if len(firsts) + len(rest) >= 5 or len(firsts) + len(rest) >= 3:
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
            n_pos = min(5, len(pos))
            out["rgb_noise_0"] = {"prefill": [by_title[t] for t in pos[:n_pos]], "masked": [], "answerable": True}
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
