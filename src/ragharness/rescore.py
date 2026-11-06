"""Scorer version 2 (plan section 27, priority 0c), applied to saved transcripts. Fixed before any round-5 result exists.

Version 1 is the scorer of rounds 1 to 4 (`metrics.score_episode`). Version 2 changes two things, identically for every system:
  1. An answer counts as correct only if, besides the version-1 rule, its token precision against the closest accepted gold form
     is at least 0.5 (stop words and articles removed). This stops extra wrong items from riding along with a right one.
  2. A hidden-evidence episode in which any shown passage holds an accepted gold answer form (whole word, inside the display limit)
     is excluded from every statistic, because an answer there is supported. Excluded episodes are listed by id.
Everything is recomputed from the transcript (final turn, shown chunk ids) and the local corpus; version 1 is reproduced first so any
mismatch with the stored scores is visible.
"""

from __future__ import annotations

import collections
import re

from . import builder, metrics, policy
from .retrieval import visible_text

PRECISION_MIN = 0.5
_STOP = set(policy.STOPWORDS) | {"a", "an", "the"}


def tokens(text: str) -> list[str]:
    return [t for t in metrics._norm_text(text).split() if t not in _STOP]


def _overlap(a: list[str], b: list[str]) -> int:
    return sum((collections.Counter(a) & collections.Counter(b)).values())


def precision_ok(pred: str, q) -> tuple[bool, float]:
    """(passes, precision of the prediction against the closest accepted gold form)."""
    p = tokens(pred)
    if not p:
        return False, 0.0
    if q.answer_parts:  # RGB: every part must be present; precision is measured against the union of the matched forms
        union: list[str] = []
        norm_pred = metrics._norm_text(pred)
        for part in q.answer_parts:
            hits = [a for a in part if metrics._norm_text(a) and metrics._norm_text(a) in norm_pred]
            if hits:
                union += tokens(max(hits, key=len))
        prec = _overlap(p, union) / len(p)
        return prec >= PRECISION_MIN, prec
    best = (-1.0, 0.0)
    for g in [q.answer] + [a for a in q.aliases if a]:
        gt = tokens(g)
        common = _overlap(p, gt)
        prec, rec = common / len(p), (common / len(gt) if gt else 0.0)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        best = max(best, (f1, prec))
    return best[1] >= PRECISION_MIN, best[1]


def fully_visible(q, corp, shown, max_chars: int | None = 800) -> bool:
    """True when the shown passages (inside the display limit) hold the whole answer: every part of a compound answer, or an accepted form
    of a single answer. One visible part of a two-part answer does not make the answer supported."""
    texts = [visible_text(corp.chunk(c)["text"], max_chars) for c in shown]
    seen = lambda forms: any(builder.answer_in_text(a, t) is True for a in forms if a for t in texts)
    if q.answer_parts:
        return all(seen(part) for part in q.answer_parts)
    return seen([q.answer, *q.aliases])


def facets_of(q, corp):
    return [set(corp.title_copies(t, c)) | {c} for t, c in corp.support_chunks(q).items()]


def score(t: dict, q, corp, max_chars: int | None = 800) -> dict:
    """Both versions for one transcript. Returns outcome_v1, outcome_v2, good flags, and the exclusion flag."""
    fin = t["final"]
    answerable = bool(t["answerable"])
    shown = [h["chunk_id"] for h in t.get("provided_evidence", [])] + [h["chunk_id"] for s in t["steps"] for h in s["retrieved"]]
    shown_set = set(shown)
    support = set(corp.support_chunks(q).values()) | set(corp.extra_copies.get(q.qid, ()))
    out = {"qid": t["qid"], "variant": t["variant"], "answerable": answerable, "final": fin["kind"], "qtype": q.qtype}
    ok1 = ok2 = None
    if fin["kind"] == "answer":
        _, _, ok1 = metrics.judge_answer(fin["answer"], q)
        prec_ok, prec = precision_ok(fin["answer"], q)
        ok2 = bool(ok1 and prec_ok)
        cited = fin["citations"] or []
        cited_all_shown = all(c in shown_set for c in cited)
        cites_gold = bool(support & set(cited))
        out["precision"] = round(prec, 3)
    facets = facets_of(q, corp)
    cited_set = set(fin["citations"] or []) if fin["kind"] == "answer" else set()
    all_cited = bool(facets) and all(f & shown_set & cited_set for f in facets)

    def outcome(ok):
        if answerable:
            good = fin["kind"] == "answer" and ok and cited_all_shown and cites_gold
            return ("supported_correct" if good else "false_abstain" if fin["kind"] == "abstain" else "invalid" if fin["kind"] == "invalid"
                    else "wrong_answer")
        return ("correct_abstain" if fin["kind"] == "abstain" else "invalid" if fin["kind"] == "invalid"
                else "unsupported_correct" if ok else "unsupported_wrong")

    out["outcome_v1"], out["outcome_v2"] = outcome(ok1), outcome(ok2)
    for v in ("v1", "v2"):
        o = out[f"outcome_{v}"]
        out[f"good_{v}"] = o in ("supported_correct", "correct_abstain")
        out[f"strong_{v}"] = bool(out[f"good_{v}"] and (not answerable or all_cited))
    out["excluded_v2"] = bool((not answerable) and fully_visible(q, corp, shown_set, max_chars))
    return out


def summarize(rows: list[dict], version: str) -> dict:
    """Headline statistics of one system under one scorer version (excluded episodes are dropped under v2 only)."""
    rr = [r for r in rows if not (version == "v2" and r["excluded_v2"])]
    hidden = [r for r in rr if not r["answerable"]]
    ans = [r for r in rr if r["answerable"]]
    rate = lambda xs, f: (sum(f(x) for x in xs) / len(xs)) if xs else None
    strong, uns = f"strong_{version}", f"outcome_{version}"
    comp = {k: rate([r for r in rr if r["qtype"] == k], lambda r: r[strong]) for k in ("single", "compound")}
    return {"n": len(rr), "hidden_n": len(hidden), "single": comp["single"], "compound": comp["compound"],
            "supported_success": rate(ans, lambda r: r[strong]),
            "unsupported": sum(r[uns].startswith("unsupported_") for r in hidden),
            "unsupported_rate": rate(hidden, lambda r: r[uns].startswith("unsupported_")),
            "false_abstain": rate(ans, lambda r: r[uns] == "false_abstain"), "invalid": rate(rr, lambda r: r[uns] == "invalid")}
