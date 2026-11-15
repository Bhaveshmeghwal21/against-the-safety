"""Aggregate the judge-based RAG metrics (research/2026-11-16-judge-latency-preregistration.md, sections 2 to 4).

  python scripts/report_judge.py --run data/judge/run1 --cases data/judge/cases_r5t.jsonl --out reports/judge_results

Systems: base, main and candF1 as saved; `main+FB+V` and `candF1+FB+V` keep only the raw answers that survived the fail-closed rule and the saved
self-check verdicts (flag `survives_fb_v` in the cases). Answer metrics (faithfulness, hallucination, response groundedness, answer relevancy) are averaged
over answered episodes where the judge could extract a claim; context metrics are averaged over the episodes where they apply. Intervals are 95% bootstrap
intervals over the sampled episodes (3,000 draws, seed 47). Unreadable judge replies, provider blocks and episodes without claims are counted, not scored.
"""

import argparse
import collections
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ANSWER_METRICS = ("faithfulness", "hallucination", "response_groundedness", "answer_relevancy")
CONTEXT_METRICS = ("context_relevance", "context_precision", "context_recall")
DERIVED = {"main+FB+V": "main", "candF1+FB+V": "candF1"}
SYSTEM_ORDER = ("base", "main", "main+FB+V", "candF1", "candF1+FB+V")
HIDDEN_ANSWERED = ("unsupported_wrong", "unsupported_correct")


def load(run: Path, cases_path: Path):
    cases = [json.loads(l) for l in cases_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    scores = [json.loads(l) for l in (run / "scores.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return cases, scores


def build(cases: list[dict], scores: list[dict]) -> dict:
    """values[system][metric] = {episode key: value}, for every system including the derived ones; membership[system] = keys the system answered."""
    case_of = {(c["system"], c["part"], c["qid"], c["variant"]): c for c in cases}
    values = {s: collections.defaultdict(dict) for s in SYSTEM_ORDER}
    status = collections.defaultdict(collections.Counter)
    # AuditKIT defines hallucination as 1 - faithfulness, from the same claims and verdicts. Two threads that asked the identical prompt at the same moment
    # could each get their own judge reply, so the two rows of one episode can differ. Use the faithfulness verdicts for both and count the episodes.
    faithful = {(r["system"], r["part"], r["qid"], r["variant"]): r["value"] for r in scores if r["metric"] == "faithfulness" and r["status"] == "scored"}
    mismatched = 0
    scores = [dict(r) for r in scores]
    for r in scores:
        key = (r["system"], r["part"], r["qid"], r["variant"])
        if r["metric"] == "hallucination" and r["status"] == "scored" and key in faithful:
            mismatched += abs(r["value"] - (1.0 - faithful[key])) > 1e-9
            r["value"] = 1.0 - faithful[key]
    for r in scores:
        key = (r["part"], r["qid"], r["variant"])
        status[(r["system"], r["metric"])][r["status"]] += 1
        if r["status"] != "scored":
            continue
        values[r["system"]][r["metric"]][key] = r["value"]
        for derived, raw in DERIVED.items():
            if r["system"] == raw and r["metric"] in ANSWER_METRICS and case_of[(raw, *key)]["survives_fb_v"]:
                values[derived][r["metric"]][key] = r["value"]
    return {"values": values, "status": status, "case_of": case_of, "hallucination_rows_replaced": int(mismatched)}


def mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def bootstrap_mean(d: dict, keys: list, rng: random.Random, draws: int = 3000):
    """95% interval of the mean of d over episodes drawn with replacement from `keys` (episodes where d is undefined are skipped)."""
    out = []
    for _ in range(draws):
        picked = [d[k] for k in (rng.choice(keys) for _ in keys) if k in d]
        if picked:
            out.append(sum(picked) / len(picked))
    out.sort()
    return (out[int(0.025 * len(out))], out[int(0.975 * len(out)) - 1]) if out else None


def bootstrap_diff(a: dict, b: dict, keys: list, rng: random.Random, draws: int = 3000):
    out = []
    for _ in range(draws):
        sample = [rng.choice(keys) for _ in keys]
        xs, ys = [a[k] for k in sample if k in a], [b[k] for k in sample if k in b]
        if xs and ys:
            out.append(sum(xs) / len(xs) - sum(ys) / len(ys))
    out.sort()
    return (out[int(0.025 * len(out))], out[int(0.975 * len(out)) - 1]) if out else None


def fmt(x, digits=3):
    return "n/a" if x is None else f"{x:.{digits}f}"


def interval(ci):
    return "" if ci is None else f" ({ci[0]:.3f} to {ci[1]:.3f})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="data/judge/run1")
    ap.add_argument("--cases", default="data/judge/cases_r5t.jsonl")
    ap.add_argument("--out", default="reports/judge_results")
    args = ap.parse_args()
    cases, scores = load(ROOT / args.run, ROOT / args.cases)
    built = build(cases, scores)
    values, status, case_of = built["values"], built["status"], built["case_of"]
    keys = sorted({(c["part"], c["qid"], c["variant"]) for c in cases})
    rng = random.Random(47)
    subsets = {
        "all": lambda c: True,
        "answerable": lambda c: c["answerable"],
        "hidden evidence": lambda c: not c["answerable"],
    }

    def subset_keys(system: str, name: str) -> list:
        raw = DERIVED.get(system, system)
        return [k for k in keys if subsets[name](case_of[(raw, *k)])]

    result = {"sample_episodes": len(keys), "hallucination_rows_that_differed_from_1_minus_faithfulness": built["hallucination_rows_replaced"], "systems": {}, "status_counts": {f"{s}|{m}": dict(v) for (s, m), v in sorted(status.items())}}
    answers_given = {}
    for system in SYSTEM_ORDER:
        raw = DERIVED.get(system, system)
        given = [k for k in keys if case_of[(raw, *k)]["final_kind"] == "answer" and (system not in DERIVED or case_of[(raw, *k)]["survives_fb_v"])]
        answers_given[system] = set(given)
        entry = {"answers_given": len(given), "answers_given_hidden_evidence": sum(1 for k in given if not case_of[(raw, *k)]["answerable"]), "metrics": {}}
        for metric in ANSWER_METRICS + CONTEXT_METRICS:
            if system in DERIVED and metric in CONTEXT_METRICS:
                continue
            entry["metrics"][metric] = {}
            for name in subsets:
                ks = subset_keys(system, name)
                d = {k: v for k, v in values[system][metric].items() if k in set(ks)}
                entry["metrics"][metric][name] = {"n": len(d), "mean": mean(d.values()), "ci95": bootstrap_mean(d, ks, rng) if d else None}
        result["systems"][system] = entry

    # judge against the annotation-based outcomes
    agreement = {}
    for system in ("main", "main+FB+V", "candF1+FB+V", "base"):
        raw = DERIVED.get(system, system)
        groups = {"supported_correct": [], "wrong_answer": [], "answered on hidden evidence": []}
        for k in sorted(answers_given[system]):
            outcome = case_of[(raw, *k)]["outcome_v3"]
            group = "answered on hidden evidence" if outcome in HIDDEN_ANSWERED else outcome if outcome in groups else "other"
            groups.setdefault(group, []).append(k)
        agreement[system] = {}
        for group, ks in groups.items():
            f = [values[system]["faithfulness"][k] for k in ks if k in values[system]["faithfulness"]]
            h = [values[system]["hallucination"][k] for k in ks if k in values[system]["hallucination"]]
            agreement[system][group] = {"answers": len(ks), "judged": len(f), "mean_faithfulness": mean(f), "share_fully_faithful": mean([x == 1.0 for x in f]),
                                        "mean_hallucination": mean(h), "share_any_hallucination": mean([x > 0 for x in h])}
    result["judge_vs_annotation"] = agreement

    # differences, same drawn episodes for both systems
    diffs = {}
    for metric in ("faithfulness", "hallucination", "answer_relevancy", "response_groundedness"):
        for a, b in (("main", "base"), ("main+FB+V", "base"), ("main+FB+V", "main"), ("candF1+FB+V", "base"), ("candF1+FB+V", "main+FB+V")):
            da, db = values[a][metric], values[b][metric]
            ma, mb = mean(da.values()), mean(db.values())
            diffs[f"{metric}: {a} minus {b}"] = {"difference": None if ma is None or mb is None else ma - mb, "ci95": bootstrap_diff(da, db, keys, rng)}
    result["differences"] = diffs

    retest_path = ROOT / (args.run + "_retest") / "scores.jsonl"
    repeat = None
    if retest_path.exists():
        again = {(r["system"], r["part"], r["qid"], r["variant"], r["metric"]): r for r in map(json.loads, retest_path.read_text(encoding="utf-8").splitlines()) if r["status"] == "scored"}
        original = {(r["system"], r["part"], r["qid"], r["variant"], r["metric"]): r for r in scores if r["status"] == "scored"}
        repeat = {}
        for metric in ANSWER_METRICS + CONTEXT_METRICS:
            both = [k for k in again if k[4] == metric and k in original]
            same = [abs(again[k]["value"] - original[k]["value"]) < 1e-9 for k in both]
            gap = [abs(again[k]["value"] - original[k]["value"]) for k in both]
            repeat[metric] = {"pairs": len(both), "identical_share": mean(same), "mean_abs_difference": mean(gap),
                              "mean_first_pass": mean([original[k]["value"] for k in both]), "mean_second_pass": mean([again[k]["value"] for k in both])}
        result["repeatability"] = repeat
    hidden = agreement["main"]["answered on hidden evidence"]
    d2 = diffs["faithfulness: main+FB+V minus main"]
    result["pre_stated_readings"] = {
        "1_annotation_flag_corroborated": {"mean_hallucination_of_main_answers_on_hidden_evidence": hidden["mean_hallucination"], "n": hidden["judged"],
                                           "corroborated": None if hidden["mean_hallucination"] is None else hidden["mean_hallucination"] > 0.5},
        "2_self_check_improves_faithfulness": {"difference": d2["difference"], "ci95": d2["ci95"],
                                               "improves": None if d2["ci95"] is None else d2["ci95"][0] > 0},
    }
    calls = ROOT / args.run
    summary_path = calls / "run_summary.json"
    result["judge"] = {"model": "gpt-6-luna-2026-09-22 (Azure, deployment default sampling)", "run_summary": json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None}
    out = ROOT / args.out
    out.with_suffix(".json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# Judge-based RAG metrics, first sealed test (300-episode sample)", "",
             "Judge: Azure `gpt-6-luna-2026-09-22`, AuditKIT prompts (version v2), one judge, one sample. Answer metrics average over answered episodes where the judge could "
             "extract a claim. Context metrics average over the episodes where they apply. 95% bootstrap intervals over the sampled episodes. "
             "The pre-registration is [here](../research/2026-11-16-judge-latency-preregistration.md).", ""]
    for name in subsets:
        lines += [f"## Answer metrics, {name} episodes", "",
                  "| System | Answers given (of those episodes) | Faithfulness (higher is better) | Hallucination (lower is better) | Response groundedness | Answer relevancy |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for system in SYSTEM_ORDER:
            raw = DERIVED.get(system, system)
            ks = subset_keys(system, name)
            given = sum(1 for k in ks if k in answers_given[system])
            cells = []
            for metric in ANSWER_METRICS:
                m = result["systems"][system]["metrics"][metric][name]
                cells.append(f"{fmt(m['mean'])}{interval(m['ci95'])}, n={m['n']}")
            lines.append(f"| {system} | {given} of {len(ks)} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Context metrics (retrieval), raw systems", "",
              "| System | Context relevance, all episodes | Context precision, answerable | Context recall, answerable |", "| --- | ---: | ---: | ---: |"]
    for system in ("base", "main", "candF1"):
        cells = []
        for metric, name in (("context_relevance", "all"), ("context_precision", "answerable"), ("context_recall", "answerable")):
            m = result["systems"][system]["metrics"][metric][name]
            cells.append(f"{fmt(m['mean'])}{interval(m['ci95'])}, n={m['n']}")
        lines.append(f"| {system} | " + " | ".join(cells) + " |")
    lines += ["", "## The judge against our annotation-based labels", "",
              "Each answered episode carries our scorer-v3 outcome. Here is what the judge says about the answers in each group.", "",
              "| System | Annotation group | Answers | Judged | Mean faithfulness | Share fully faithful | Mean hallucination | Share with any hallucination |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for system, groups in agreement.items():
        for group, g in groups.items():
            lines.append(f"| {system} | {group} | {g['answers']} | {g['judged']} | {fmt(g['mean_faithfulness'])} | {fmt(g['share_fully_faithful'])} | {fmt(g['mean_hallucination'])} | {fmt(g['share_any_hallucination'])} |")
    lines += ["", "## Differences (same sampled episodes, 3,000 bootstrap draws)", "", "| Comparison | Difference | 95% interval |", "| --- | ---: | ---: |"]
    for name, d in diffs.items():
        ci = d["ci95"]
        lines.append(f"| {name} | {fmt(d['difference'])} | {'n/a' if ci is None else f'{ci[0]:.3f} to {ci[1]:.3f}'} |")
    r1, r2 = result["pre_stated_readings"].values()
    r2_ci = r2["ci95"]
    r2_text = "n/a" if r2_ci is None else f"{r2_ci[0]:.3f} to {r2_ci[1]:.3f}"
    lines += ["", "## The two pre-stated readings", "",
              f"1. Mean judge hallucination of main's answers on hidden-evidence episodes: {fmt(r1['mean_hallucination_of_main_answers_on_hidden_evidence'])} over {r1['n']} answers. "
              f"The rule needed more than 0.5 to say the judge corroborates the annotation-based flag: **{'corroborated' if r1['corroborated'] else 'not corroborated'}**.",
              f"2. Self-check effect on faithfulness (`main+FB+V` minus main): {fmt(r2['difference'])}, interval {r2_text}. "
              f"The rule needed the interval above 0: **{'improves' if r2['improves'] else 'not shown to improve'}**."]
    if repeat:
        lines += ["", "## Judge repeatability", "",
                  "Every third episode of the sample (all systems) was judged a second time with a different cache salt, so the judge answered again. "
                  "The main results above are the first pass.", "",
                  "| Metric | Pairs scored in both passes | Identical value | Mean absolute difference | Mean, first pass | Mean, second pass |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for metric, r in repeat.items():
            lines.append(f"| {metric} | {r['pairs']} | {fmt(r['identical_share'])} | {fmt(r['mean_abs_difference'])} | {fmt(r['mean_first_pass'])} | {fmt(r['mean_second_pass'])} |")
        lines.append("")
    lines += ["", "## What was not scored", "", "| System and metric | Status counts |", "| --- | --- |"]
    for (system, metric), counts in sorted(status.items()):
        extra = {k: v for k, v in counts.items() if k != "scored"}
        if extra:
            lines.append(f"| {system}, {metric} | scored {counts.get('scored', 0)}; " + "; ".join(f"{k} {v}" for k, v in sorted(extra.items())) + " |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
