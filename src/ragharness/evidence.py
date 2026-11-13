"""Annotation-backed evidence alternatives shared by scoring and AuditKIT.

An answer string in an unrelated passage is not evidence. Only annotated passages
and copies containing the annotated facts qualify. All checks use displayed text.
"""
from __future__ import annotations

import itertools
import re

from .retrieval import visible_text


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def contains(text: str, fact: str) -> bool:
    needle = normalize(fact)
    return bool(needle and f" {needle} " in f" {normalize(text)} ")


def rgb_contains(form: str, text: str) -> bool:
    """RGB's accepted substring forms, only within annotated positive passages.

    RGB includes ordinals (26/26th, eight/eighth), currency and compounds
    (GPT/ChatGPT). Its positive annotations supply the relevance boundary.
    Numeric forms cannot match the middle of a different number.
    """
    from .metrics import _norm_text
    form, text = _norm_text(form), _norm_text(text)
    if not form:
        return False
    if form.isdigit():
        return re.search(r"(?<!\d)" + re.escape(form) + r"(?!\d)", text) is not None
    return form in text


def facets(q, corpus, texts: dict[str, str] | None = None, max_chars=800) -> list[set[str]]:
    """One set of interchangeable passages per required fact/answer component."""
    def text(cid):
        return texts.get(cid, "") if texts is not None else visible_text(corpus.chunk(cid)["text"], max_chars)

    support = corpus.support_chunks(q)
    if q.answer_parts and q.qtype != "compound":
        candidates = set(support.values()) | set(corpus.extra_copies.get(q.qid, ()))
        return [{cid for cid in candidates if any(rgb_contains(a, text(cid)) for a in part if a)}
                for part in q.answer_parts]
    result = []
    for title, cid in support.items():
        facts = q.support_sentences()[title]
        # A same-title passage is an alternative only when it contains the facts.
        result.append({c for c in corpus.title_copies(title, cid)
                       if facts and all(contains(text(c), fact) for fact in facts)})
    return result


def sufficient_sets(q, corpus, texts=None, max_chars=800) -> list[list[str]]:
    groups = facets(q, corpus, texts, max_chars)
    if not groups or any(not g for g in groups):
        return []
    combinations = {frozenset(c) for c in itertools.product(*(sorted(g) for g in groups))}
    # If one passage supplies two parts, supersets of that passage are redundant.
    minimal = [s for s in combinations if not any(t < s for t in combinations)]
    return [sorted(s) for s in sorted(minimal, key=lambda s: (len(s), sorted(s)))]


def supported(q, corpus, cited: set[str], texts: dict[str, str]) -> bool:
    groups = facets(q, corpus, texts)
    return bool(groups) and all(g & cited for g in groups)


def allowed_ids(q, corpus, masked=()) -> set[str]:
    ids = set(corpus._by_id) - set(masked)
    if getattr(q, "evidence_scope", "corpus") == "paragraph" or q.natural_unanswerable:
        ids &= set(corpus.by_question[q.qid].values())
    return ids


def eligibility(q, corpus, nominal_answerable: bool, masked=(), max_chars=800) -> dict:
    """Fixed once from annotations, corpus and evidence mask, before model scoring.

    Ambiguous alternate support needs adjudication; gold-string coincidences never
    change the denominator. A nominally open case with truncated support is invalid.
    """
    permitted = allowed_ids(q, corpus, masked)
    groups = facets(q, corpus, max_chars=max_chars)
    available = bool(groups) and all(g & permitted for g in groups)
    reason = "eligible"
    if nominal_answerable and not available:
        reason = "annotated_support_not_displayable"
    elif not nominal_answerable and available:
        reason = "annotated_support_survives_mask"
    return {"eligible": reason == "eligible", "reason": reason,
            "answerable": nominal_answerable, "annotated_evidence_available": available}


def shown_texts(transcript: dict, corpus, max_chars=800) -> dict[str, str]:
    """Use saved excerpts when present; otherwise reproduce the recorded display."""
    result = {}
    hits = list(transcript.get("provided_evidence", []))
    hits += [h for s in transcript.get("steps", []) for h in s.get("retrieved", [])]
    for hit in hits:
        cid = hit["chunk_id"]
        if cid in result or cid not in corpus._by_id:
            continue
        raw = hit.get("text")
        result[cid] = visible_text(raw if raw is not None else corpus.chunk(cid)["text"], max_chars)
    for step in transcript.get("steps", []):
        observation = step.get("observation") or ""
        matches = list(re.finditer(r"\[chunk_id=([^\]]+)\] ([^\n]*)(?:\n|$)", observation))
        for i, match in enumerate(matches):
            cid, title = match.group(1), match.group(2)
            if title.endswith("(shown above)"):
                continue
            end = matches[i + 1].start() if i + 1 < len(matches) else len(observation)
            if cid in corpus._by_id:
                result[cid] = observation[match.end():end].strip()
    return result
