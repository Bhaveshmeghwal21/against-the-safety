"""Before-and-after scorecard built only from the competition's own criteria.

The hackathon asks for before-and-after metrics (more than one), a write-up of the metrics chosen and why, and evidence the
model is meaningfully better. Problem statement 4 names three abilities: decide WHEN and WHAT to retrieve, GROUND answers in
retrieved text, and IGNORE distracting documents. Track 3's evaluation tool, AuditKIT, is described as evaluating tool-call
accuracy and reliability. So the scorecard reports exactly those, each as its own number with its denominator, plus the
abstention behaviour the same documents-only setting requires. There is no composite score and no weighting.

Input per system: the per-episode `scores.jsonl` and `summary.json` written by scripts/run_eval.py.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from .metrics import F1_THRESHOLD

NO_EVIDENCE = ("open", "first_search_masked", "support_masked")  # the user turn carries no evidence
SUFFICIENT = ("prefill_sufficient", "prefill_clean")


def load(run_dir: Path) -> tuple[list[dict], dict]:
    scores = [json.loads(l) for l in (run_dir / "scores.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return scores, summary


def _rate(rows: list[dict], pred) -> dict:
    n = len(rows)
    k = sum(1 for r in rows if pred(r))
    return {"value": round(k / n, 3) if n else None, "k": k, "n": n}


def clause_metrics(scores: list[dict], summary: dict) -> dict:
    by = defaultdict(list)
    for s in scores:
        by[s["variant"]].append(s)
    get = lambda *vs: [s for v in vs for s in by.get(v, [])]
    answerable = [s for s in scores if s["answerable"]]
    unanswerable = [s for s in scores if not s["answerable"]]
    ak = summary.get("auditkit", {}).get("all", {})
    tokens = [s.get("input_tokens", 0) for s in scores]
    return {
        "when_to_retrieve": {
            "missed_search_rate (no evidence supplied; lower is better)": _rate(get(*NO_EVIDENCE), lambda s: s["n_searches"] == 0),
            "unnecessary_search_rate (sufficient evidence supplied; lower is better)": _rate(get(*SUFFICIENT), lambda s: s["n_searches"] > 0),
            "search_when_evidence_partial (higher is better)": _rate(get("prefill_partial"), lambda s: s["n_searches"] > 0),
        },
        "what_to_retrieve": {
            "full_support_found_by_own_queries (open questions)": _rate(get("open"), lambda s: s["evidence_complete"]),
            "second_hop_recovery (partial or weak first result)": _rate(get("first_search_masked", "prefill_partial"), lambda s: s["evidence_complete"]),
            "auditkit_evidence_set_recall": ak.get("evidence_set_recall"),
        },
        "grounding": {
            "supported_correct (answerable episodes)": _rate(answerable, lambda s: s["outcome"] == "supported_correct"),
            "supported_correct (open questions)": _rate(get("open"), lambda s: s["outcome"] == "supported_correct"),
            "answer_correct_ignoring_citations (answerable; F1 >= 0.5)": _rate(answerable, lambda s: s["final"] == "answer" and (s["f1"] or 0) >= F1_THRESHOLD),
            "cites_a_gold_passage_when_answering (answerable)": _rate([s for s in answerable if s["final"] == "answer"], lambda s: bool(s["cites_gold"])),
            "answer_f1_when_answered": summary.get("overall", {}).get("answer_f1_when_answered"),
            "auditkit_citation_support": ak.get("citation_support"),
        },
        "ignores_distractors": {
            "supported_correct with distractor passage": _rate(get("prefill_sufficient"), lambda s: s["outcome"] == "supported_correct"),
            "supported_correct, same evidence without distractor": _rate(get("prefill_clean"), lambda s: s["outcome"] == "supported_correct"),
        },
        "abstention (documents-only requirement)": {
            "correct_abstain (unanswerable episodes)": _rate(unanswerable, lambda s: s["outcome"] == "correct_abstain"),
            "unsupported_answer_rate (unanswerable; lower is better)": _rate(unanswerable, lambda s: s["final"] == "answer"),
            "false_abstain (answerable; lower is better)": _rate(answerable, lambda s: s["outcome"] == "false_abstain"),
            "auditkit_abstention": ak.get("abstention"),
        },
        "tool_call_reliability (AuditKIT, Track 3)": {
            "tool_call_validity": ak.get("tool_call_validity"),
            "redundant_tool_calls (lower is better)": ak.get("redundant_tool_calls"),
            "agent_loop_detection (1.0 = loop-free)": ak.get("agent_loop_detection"),
            "invalid_turn_rate (unparseable or rule-breaking; lower is better)": _rate(scores, lambda s: s["outcome"] == "invalid"),
        },
        "efficiency": {
            "mean_searches_per_episode": round(sum(s["n_searches"] for s in scores) / len(scores), 2) if scores else None,
            "mean_input_tokens_per_episode": round(sum(tokens) / len(tokens), 1) if scores else None,
            "mean_seconds_per_episode": summary.get("cost", {}).get("mean_seconds"),
        },
        "episodes": len(scores),
    }


def product_table(systems: dict[str, Path]) -> str:
    """Per-kind results on the fictional Nimbus Home suite (24 tasks, exploratory). Each cell is k/n."""
    names = list(systems)
    data = {n: load(p) for n, p in systems.items()}
    kinds = [("ordinary", "ordinary (one document), supported_correct"), ("two_doc", "two documents, supported_correct"),
             ("dated", "dated change (current edition needed), supported_correct"), ("missing", "missing evidence, correct_abstain")]
    lines = ["| metric | " + " | ".join(names) + " |", "| --- |" + " ---: |" * len(names)]
    for kind, label in kinds:
        want = "correct_abstain" if kind == "missing" else "supported_correct"
        cells = []
        for n in names:
            rows = [s for s in data[n][0] if s.get("qtype") == kind]
            cells.append(fmt(_rate(rows, lambda s: s["outcome"] == want)))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    extra = [
        ("dated change: cited a superseded edition (lower is better)", lambda s: s.get("cites_superseded"), "dated"),
        ("missing evidence: gave an unsupported answer (lower is better)", lambda s: s["final"] == "answer", "missing"),
        ("all tasks: invalid turn (lower is better)", lambda s: s["outcome"] == "invalid", None),
    ]
    for label, pred, kind in extra:
        cells = [fmt(_rate([s for s in data[n][0] if kind is None or s.get("qtype") == kind], pred)) for n in names]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    for key, label in (("freshness", "AuditKIT freshness of retrieved documents (higher is better)"), ("stale_hit_rate", "AuditKIT stale hit rate (lower is better)")):
        cells = [fmt(data[n][1].get("auditkit", {}).get("all", {}).get(key)) for n in names]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("| mean searches per task | " + " | ".join(fmt(round(sum(s["n_searches"] for s in data[n][0]) / max(1, len(data[n][0])), 2)) for n in names) + " |")
    return "\n".join(lines)


def paired(a: list[dict], b: list[dict], select, seed: int = 13, reps: int = 5000) -> dict | None:
    """Difference in the share of `good` episodes, b minus a, over episodes both systems ran; bootstrap resamples whole questions."""
    ka = {(s["qid"], s["variant"]): select(s) for s in a}
    kb = {(s["qid"], s["variant"]): select(s) for s in b}
    keys = sorted(k for k in ka if k in kb and ka[k] is not None and kb[k] is not None)
    if not keys:
        return None
    by_q: dict[str, list] = defaultdict(list)
    for k in keys:
        by_q[k[0]].append((ka[k], kb[k]))
    qs = list(by_q)
    rng = random.Random(seed)
    diffs = []
    for _ in range(reps):
        picked = [x for q in (rng.choice(qs) for _ in qs) for x in by_q[q]]
        diffs.append(sum(y for _, y in picked) / len(picked) - sum(x for x, _ in picked) / len(picked))
    diffs.sort()
    n = len(keys)
    a_rate, b_rate = sum(ka[k] for k in keys) / n, sum(kb[k] for k in keys) / n
    return {
        "n_episodes": n, "n_questions": len(qs), "before": round(a_rate, 3), "after": round(b_rate, 3), "difference": round(b_rate - a_rate, 3),
        "ci95": [round(diffs[int(0.025 * reps)], 3), round(diffs[int(0.975 * reps) - 1], 3)],
        "after_only_good": sum(1 for k in keys if kb[k] and not ka[k]), "before_only_good": sum(1 for k in keys if ka[k] and not kb[k]),
    }


PAIRED_SELECTORS = {
    "end-to-end good (headline: supported answer when answerable, abstain when not)": lambda s: float(s["good"]) if s["variant"] in ("open", "support_masked") else None,
    "supported_correct (answerable episodes)": lambda s: float(s["outcome"] == "supported_correct") if s["answerable"] else None,
    "answer correct, ignoring citations (answerable; F1 >= 0.5)": lambda s: float(s["final"] == "answer" and (s["f1"] or 0) >= F1_THRESHOLD) if s["answerable"] else None,
    "correct_abstain (unanswerable episodes)": lambda s: float(s["outcome"] == "correct_abstain") if not s["answerable"] else None,
    "unsupported answer (unanswerable; lower is better)": lambda s: float(s["final"] == "answer") if not s["answerable"] else None,
    "supported_correct with distractor passage": lambda s: float(s["outcome"] == "supported_correct") if s["variant"] == "prefill_sufficient" else None,
}


def build(systems: dict[str, Path], before: str, after: str, controls: list[str]) -> dict:
    data = {name: load(p) for name, p in systems.items()}
    out = {"systems": {name: clause_metrics(*data[name]) for name in data}, "paired": {}}
    for label, ref in [(f"{after} vs {before}", before)] + [(f"{after} vs {c}", c) for c in controls if c in data]:
        out["paired"][label] = {name: paired(data[ref][0], data[after][0], sel) for name, sel in PAIRED_SELECTORS.items()}
    return out


def fmt(v) -> str:
    if isinstance(v, dict) and "value" in v:
        return "n/a" if v["value"] is None else f"{v['value']:.3f} ({v['k']}/{v['n']})"
    return "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def markdown(card: dict, title: str) -> str:
    names = list(card["systems"])
    lines = [f"# {title}", "", "Each number is reported separately with its denominator. There is no composite score.", ""]
    first = card["systems"][names[0]]
    for section in [k for k in first if k != "episodes"]:
        lines += [f"## {section}", "", "| metric | " + " | ".join(names) + " |", "| --- |" + " ---: |" * len(names)]
        for metric in first[section]:
            lines.append(f"| {metric} | " + " | ".join(fmt(card["systems"][n][section].get(metric)) for n in names) + " |")
        lines.append("")
    lines += ["## Episodes", "", "| system | episodes |", "| --- | ---: |"] + [f"| {n} | {card['systems'][n]['episodes']} |" for n in names] + [""]
    for label, res in card["paired"].items():
        lines += [f"## Paired difference: {label}", "", "95% interval from resampling whole questions; both systems ran the same episodes.", "",
                  "| metric | before | after | difference | 95% CI | episodes / questions | only-after / only-before good |", "| --- | ---: | ---: | ---: | --- | ---: | ---: |"]
        for name, r in res.items():
            if r is None:
                lines.append(f"| {name} | n/a | n/a | n/a | n/a | 0 | n/a |")
            else:
                lines.append(f"| {name} | {r['before']:.3f} | {r['after']:.3f} | {r['difference']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['n_episodes']} / {r['n_questions']} | {r['after_only_good']} / {r['before_only_good']} |")
        lines.append("")
    return "\n".join(lines)
