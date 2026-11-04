"""Retrieval-score gate baseline (plan section 8.1): abstain when the BM25 score of the retrieved passages is low.

It needs no GPU and no model. The threshold is chosen on DEV only and applied unchanged to TEST. Because there is
no reader here, it is a decision baseline: it reports how well a score threshold separates answerable from
unanswerable episodes, not answer quality. The edge-RAG study in the plan found score gates weak on in-topic gaps.
"""

from __future__ import annotations

from dataclasses import dataclass

from .retrieval import MaskedSearch


@dataclass
class GateSample:
    qid: str
    variant: str
    answerable: bool
    top1: float
    top1_best_of_searches: float


def first_search_score(search: MaskedSearch, query: str, top_k: int, masked: frozenset[str]) -> float:
    hits = search.search(query, top_k, masked)
    return hits[0].score if hits else 0.0


def best_threshold(samples: list[GateSample], key: str = "top1") -> tuple[float, dict]:
    """Threshold on `key` (abstain when score < tau) maximizing balanced accuracy on the given samples."""
    values = sorted({getattr(s, key) for s in samples})
    best = (-1.0, 0.0, {})
    for lo, hi in zip(values, values[1:] + [values[-1] + 1]):
        tau = (lo + hi) / 2
        m = evaluate(samples, tau, key)
        if m["balanced_accuracy"] > best[0]:
            best = (m["balanced_accuracy"], tau, m)
    return best[1], best[2]


def evaluate(samples: list[GateSample], tau: float, key: str = "top1") -> dict:
    una = [s for s in samples if not s.answerable]
    ans = [s for s in samples if s.answerable]
    tpr = sum(getattr(s, key) < tau for s in una) / len(una) if una else float("nan")
    fpr = sum(getattr(s, key) < tau for s in ans) / len(ans) if ans else float("nan")
    return {"tau": round(tau, 4), "abstain_when_unanswerable": round(tpr, 3), "false_abstain_when_answerable": round(fpr, 3),
            "balanced_accuracy": round((tpr + (1 - fpr)) / 2, 3), "n_unanswerable": len(una), "n_answerable": len(ans)}
