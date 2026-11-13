"""Per-episode deterministic scoring and paired comparison (plan section 8).

Definitions (all deterministic, using AgentTune's QA helpers for answer text):
  answerable episode (gold support reachable)
    supported_correct   answered, answer F1 >= 0.5 against gold, every cited chunk was shown in the episode,
                        every cited chunk exists, and at least one cited chunk is a gold support paragraph
    false_abstain       abstained although the support was reachable
    wrong_answer        answered but not supported_correct
  unanswerable episode (gold support hidden by the mask)
    correct_abstain     abstained
    unsupported_correct answered and F1 >= 0.5 (the model knew it, or guessed; counted as an unsupported answer)
    unsupported_wrong   answered and F1 < 0.5
  invalid               an unparseable or rule-breaking turn; never counted as an abstention
AuditKIT's evidence_set_recall, citation_support and abstention are computed separately from the same episodes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache

from agenttune.rag.rewards.qa_metrics import _get_squad_metric

from .corpus import Corpus
from .data import Question
from .episodes import Episode

F1_THRESHOLD = 0.5


@lru_cache(maxsize=16384)
def _answer_scores(pred: str, gold: str):
    """Cache deterministic AgentTune QA scores during multi-version transcript replay."""
    # Use the same loaded SQuAD implementation as AgentTune. Its public
    # compute() writes a temporary Arrow dataset for every individual pair.
    result = _get_squad_metric()._compute(
        predictions=[{"id": "0", "prediction_text": pred}],
        references=[{"id": "0", "answers": {"text": [gold], "answer_start": [0]}}],
    )
    return result["f1"] / 100.0, result["exact_match"] / 100.0


def _norm_text(s: str) -> str:
    import re
    import string

    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def judge_answer(pred: str, q: Question) -> tuple[float, bool, bool]:
    """(F1, exact match, answer counts as correct) against every accepted form of the gold answer.

    HotpotQA questions have one gold string, so this is AgentTune's F1 / EM and the 0.5 threshold as before.
    `aliases` (MuSiQue, SQuAD 2.0) add accepted strings; the best match counts. `answer_parts` (RGB) switch to RGB's own rule:
    the answer is correct when every part has one of its accepted forms contained in it.
    """
    golds = [q.answer] + [a for a in q.aliases if a]
    qa = [_answer_scores(pred, g) for g in golds]
    f1 = max(s[0] for s in qa)
    em = max(s[1] for s in qa)
    if q.answer_parts:
        p = _norm_text(pred)
        ok = all(any(_norm_text(a) and _norm_text(a) in p for a in part) for part in q.answer_parts)
    else:
        ok = f1 >= F1_THRESHOLD
    return f1, em, ok


def score_episode(ep: Episode, q: Question, corpus: Corpus, answerable: bool, version: str = "v3") -> dict:
    final = ep.final.turn
    # Gold passages a citation may point to: the support chunks, plus (benchmarks only) other passages that each hold the answer
    support = set(corpus.support_chunks(q).values()) | set(corpus.extra_copies.get(q.qid, ()))
    shown = {h.chunk_id for h in ep.retrieved_chunks()}
    out = {
        "qid": ep.qid, "variant": ep.condition, "answerable": answerable, "final": final.kind,
        "n_searches": ep.n_searches, "evidence_complete": ep.complete, "stop": ep.stop_reason,
        "f1": None, "em": None, "answer_ok": None, "cited_all_shown": None, "cites_gold": None,
    }
    if final.kind == "answer":
        out["f1"], out["em"], out["answer_ok"] = judge_answer(final.answer, q)
        out["cited_all_shown"] = all(c in shown for c in final.citations)
        out["cites_gold"] = bool(support & set(final.citations))
    if answerable:
        ok = final.kind == "answer" and out["answer_ok"] and out["cited_all_shown"] and out["cites_gold"]
        out["outcome"] = "supported_correct" if ok else "false_abstain" if final.kind == "abstain" else "invalid" if final.kind == "invalid" else "wrong_answer"
    else:
        out["outcome"] = (
            "correct_abstain" if final.kind == "abstain" else "invalid" if final.kind == "invalid"
            else "unsupported_correct" if out["answer_ok"] else "unsupported_wrong"
        )
    out["good"] = out["outcome"] in ("supported_correct", "correct_abstain")
    if version == "v3":
        from . import audit, rescore
        transcript = {"qid": ep.qid, "variant": ep.condition, "answerable": answerable,
                      "max_chars": ep.max_chars, "masked": list(ep.masked),
                      "provided_evidence": [{"chunk_id": h.chunk_id, "text": h.text} for h in ep.prefill],
                      "steps": [{"retrieved": [{"chunk_id": h.chunk_id, "text": h.text} for h in s.hits]} for s in ep.steps],
                      "final": {"kind": final.kind, "answer": final.answer, "citations": final.citations}}
        corrected = rescore.score(transcript, q, corpus, ep.max_chars,
                                  chunk_metadata=audit.CHUNK_META, decision_date=audit.DECISION_DATE)
        out.update(corrected)
        out["outcome"], out["good"] = corrected["outcome_v3"], corrected["good_v3"]
        out["scorer"] = "v3"
        out["evidence_supported_good"] = corrected["strong_v3"]
    elif version != "v1":
        raise ValueError(f"unknown scorer version: {version}")
    return out


def summarize(scores: list[dict]) -> dict:
    total = len(scores)
    scores = [s for s in scores if not s.get("excluded_v3", False)]
    def rate(rs, key):
        return round(sum(1 for r in rs if key(r)) / len(rs), 3) if rs else None

    ans = [s for s in scores if s["answerable"]]
    una = [s for s in scores if not s["answerable"]]
    return {
        "n_total": total, "n_excluded": total - len(scores),
        "n": len(scores), "n_answerable": len(ans), "n_unanswerable": len(una),
        "end_to_end_good": rate(scores, lambda s: s["good"]),
        "supported_correct": rate(ans, lambda s: s["outcome"] == "supported_correct"),
        "false_abstain": rate(ans, lambda s: s["outcome"] == "false_abstain"),
        "wrong_answer": rate(ans, lambda s: s["outcome"] == "wrong_answer"),
        "correct_abstain": rate(una, lambda s: s["outcome"] == "correct_abstain"),
        "unsupported_answer": rate(una, lambda s: s["outcome"] in ("unsupported_correct", "unsupported_wrong")),
        "unsupported_correct": rate(una, lambda s: s["outcome"] == "unsupported_correct"),
        "unsupported_wrong": rate(una, lambda s: s["outcome"] == "unsupported_wrong"),
        "invalid": rate(scores, lambda s: s["outcome"] == "invalid"),
        "mean_searches": round(sum(s["n_searches"] for s in scores) / len(scores), 2) if scores else None,
        "answer_f1_when_answered": round(sum(s["f1"] for s in scores if s["f1"] is not None) / max(1, sum(s["f1"] is not None for s in scores)), 3),
    }


@dataclass
class Paired:
    n: int
    mean_a: float
    mean_b: float
    diff: float
    ci_low: float
    ci_high: float
    a_only: int
    b_only: int


def paired_bootstrap(a: list[bool], b: list[bool], seed: int = 13, reps: int = 5000) -> Paired:
    """Paired percentile bootstrap over questions for the difference in a success rate (b minus a)."""
    assert len(a) == len(b) and a
    n = len(a)
    rng = random.Random(seed)
    diffs = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(b[i] for i in idx) / n - sum(a[i] for i in idx) / n)
    diffs.sort()
    return Paired(n, sum(a) / n, sum(b) / n, sum(b) / n - sum(a) / n, diffs[int(0.025 * reps)], diffs[int(0.975 * reps) - 1],
                  sum(x and not y for x, y in zip(a, b)), sum(y and not x for x, y in zip(a, b)))
