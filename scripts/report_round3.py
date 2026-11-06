"""Third-round test report (plan section 24) -> reports/round3_<tag>.md/.json and, for the fresh test, docs/figures/round3.svg.

  python scripts/report_round3.py --eval <dir> --eval <dir> --systems main,candC --suite test        # fresh test set r3t_*
  python scripts/report_round3.py --eval <dir> --systems main,candC --suite informed                  # first-round benchmark sets, re-run once

Reads <system>_bench_<set>/lenient/{scores.jsonl,summary.json}. The primary result is the macro average over the components of
"ends well" (each component over all its scored episodes) and the difference second system minus first, with a 95% interval from
resampling whole questions inside each component. The safety check is the pooled unsupported-answer rate over all hidden-evidence episodes.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from bench_report import AUDIT, AUDIT_LABEL, cell, metrics, rate
from ragharness import scorecard

ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    "test": [("r3t_hotpot", "HotpotQA (fresh)"), ("r3t_2wiki", "2WikiMultihopQA (fresh)"), ("r3t_musique", "MuSiQue 2-hop (fresh)"),
             ("r3t_squad2", "SQuAD 2.0 (fresh questions)"), ("r3t_rgb", "RGB (fresh)"), ("r3t_rgb_int", "RGB integration (fresh)")],
    "informed": [("2wiki", "2WikiMultihopQA"), ("musique", "MuSiQue 2-hop"), ("squad2", "SQuAD 2.0"), ("rgb", "RGB"), ("rgb_int", "RGB integration")],
}
LABEL = {"base": "Tiny Aya base", "main": "main", "candA": "candA", "candC": "candC", "candD": "candD", "control": "control"}
SAFE_MARGIN = 0.05


def unsupported(s: dict) -> bool:
    return s["outcome"] in ("unsupported_correct", "unsupported_wrong")


def clusters(a: list[dict], b: list[dict], fn) -> dict[str, list[tuple[float, float]]]:
    ka, kb = {(s["qid"], s["variant"]): s for s in a}, {(s["qid"], s["variant"]): s for s in b}
    out: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for k in ka.keys() & kb.keys():
        r = fn(ka[k], kb[k])
        if r is not None:
            out[k[0]].append(r)
    return out


def boot_macro(per_comp: dict[str, dict[str, list[tuple[float, float]]]], reps: int = 4000, seed: int = 13) -> dict:
    """Macro average of (b rate - a rate) over components, resampling questions inside each component."""
    def diff(sample: dict[str, list[list[tuple[float, float]]]]) -> float:
        ds = []
        for rows in sample.values():
            flat = [p for r in rows for p in r]
            ds.append(sum(y for _, y in flat) / len(flat) - sum(x for x, _ in flat) / len(flat))
        return sum(ds) / len(ds)

    full = {c: list(cl.values()) for c, cl in per_comp.items() if cl}
    point = diff(full)
    rng = random.Random(seed)
    boots = sorted(diff({c: [rows[rng.randrange(len(rows))] for _ in rows] for c, rows in full.items()}) for _ in range(reps))
    return {"difference": round(point, 4), "ci95": [round(boots[int(0.025 * reps)], 4), round(boots[int(0.975 * reps) - 1], 4)]}


def boot_pooled(cl: dict[str, list[tuple[float, float]]], reps: int = 4000, seed: int = 13) -> dict:
    rows = list(cl.values())
    flat = lambda rs: [p for r in rs for p in r]
    diff = lambda rs: (lambda f: sum(y for _, y in f) / len(f) - sum(x for x, _ in f) / len(f))(flat(rs))
    rng = random.Random(seed)
    boots = sorted(diff([rows[rng.randrange(len(rows))] for _ in rows]) for _ in range(reps))
    f = flat(rows)
    return {"before": round(sum(x for x, _ in f) / len(f), 4), "after": round(sum(y for _, y in f) / len(f), 4), "n": len(f),
            "difference": round(diff(rows), 4), "ci95": [round(boots[int(0.025 * reps)], 4), round(boots[int(0.975 * reps) - 1], 4)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="append", required=True)
    ap.add_argument("--systems", required=True, help="comma list; the first is the reference (main), the second is compared with it")
    ap.add_argument("--suite", choices=list(SUITES), default="test")
    args = ap.parse_args()
    roots = [Path(e) for e in args.eval]
    systems = args.systems.split(",")
    sets = SUITES[args.suite]
    data, audit = {}, {}
    for b, _ in sets:
        for s in systems:
            for r in roots:
                d = r / f"{s}_bench_{b}" / "lenient"
                if (d / "scores.jsonl").exists():
                    sc, summ = scorecard.load(d)
                    data[(b, s)], audit[(b, s)] = sc, summ["auditkit"]["all"]
                    break
    have = [(b, t) for b, t in sets if all((b, s) in data for s in systems)]
    ref, new = systems[0], systems[1] if len(systems) > 1 else None

    good = lambda x, y: (float(x["good"]), float(y["good"]))
    unsup = lambda x, y: (float(unsupported(x)), float(unsupported(y))) if not x["answerable"] else None
    fa = lambda x, y: (float(x["outcome"] == "false_abstain"), float(y["outcome"] == "false_abstain")) if x["answerable"] and x["variant"] == "open" else None
    result: dict = {"suite": args.suite, "systems": systems, "components": {}}
    lines = [f"# Third round, {'fresh test set' if args.suite == 'test' else 'informed re-run of the first-round benchmarks'} (lenient reading)", "",
             "Plan section 24. " + ("Opened once; no choice depends on these numbers." if args.suite == "test" else
                                    "These sets informed the design of round 3, so this is a before and after, not a test; nothing was chosen with it."), ""]
    head = "| set (episodes) | " + " | ".join(LABEL[s] for s in systems) + (f" | {LABEL[new]} minus {LABEL[ref]} |" if new else " |")
    lines += [head, "| --- |" + " ---: |" * len(systems) + (" --- |" if new else "")]
    per_ends, per_unsup_cl = {}, {}
    for b, title in have:
        m = {s: metrics(data[(b, s)]) for s in systems}
        key = "ends well (all scored episodes)"
        row = f"| {title} ({m[ref][key]['n']}) | " + " | ".join(cell(m[s][key]) for s in systems)
        if new:
            cl = clusters(data[(b, ref)], data[(b, new)], good)
            per_ends[b] = cl
            p = boot_pooled(cl)
            row += f" | {p['difference']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]"
            result["components"][b] = {"metrics": m, "ends_well_paired": p}
            per_unsup_cl[b] = clusters(data[(b, ref)], data[(b, new)], unsup)
        else:
            result["components"][b] = {"metrics": m}
        lines.append(row + " |")
    macro = {s: sum(metrics(data[(b, s)])["ends well (all scored episodes)"]["value"] for b, _ in have) / len(have) for s in systems}
    result["macro"] = {s: round(v, 4) for s, v in macro.items()}
    if new:
        mc = boot_macro(per_ends)
        result["macro_paired"] = mc
        lines.append(f"| **macro average of {len(have)} sets** | " + " | ".join(f"{macro[s]:.3f}" for s in systems) + f" | **{mc['difference']:+.3f} [{mc['ci95'][0]:+.3f}, {mc['ci95'][1]:+.3f}]** |")
        pooled_unsup = boot_pooled({f"{b}:{q}": v for b, cl in per_unsup_cl.items() for q, v in cl.items()})
        pooled_fa = boot_pooled({f"{b}:{q}": v for b in [b for b, _ in have] for q, v in clusters(data[(b, ref)], data[(b, new)], fa).items()})
        result["pooled_unsupported_answers"], result["pooled_false_abstain_open"] = pooled_unsup, pooled_fa
        safe = pooled_unsup["after"] <= pooled_unsup["before"] + SAFE_MARGIN
        better = mc["ci95"][0] > 0
        result["criteria"] = {"primary_difference_ci_above_zero": better, "pooled_unsupported_answers_within_0.05_of_reference": safe,
                              "recommend_as_default": better and safe}
        lines += ["", "## Safety and refusals", "",
                  "| pooled over all sets | " + LABEL[ref] + " | " + LABEL[new] + " | difference [95% CI] | episodes |", "| --- | ---: | ---: | --- | ---: |",
                  f"| unsupported answers, hidden-evidence episodes (lower is better) | {pooled_unsup['before']:.3f} | {pooled_unsup['after']:.3f} | {pooled_unsup['difference']:+.3f} [{pooled_unsup['ci95'][0]:+.3f}, {pooled_unsup['ci95'][1]:+.3f}] | {pooled_unsup['n']} |",
                  f"| false abstentions, open search (lower is better) | {pooled_fa['before']:.3f} | {pooled_fa['after']:.3f} | {pooled_fa['difference']:+.3f} [{pooled_fa['ci95'][0]:+.3f}, {pooled_fa['ci95'][1]:+.3f}] | {pooled_fa['n']} |", "",
                  f"Pre-registered criteria: primary difference with its 95% interval above zero: **{'met' if better else 'not met'}**; pooled unsupported-answer rate at most {SAFE_MARGIN} above the reference's: **{'met' if safe else 'not met'}**. "
                  f"Recommended as the default model: **{'yes' if better and safe else 'no'}**."]
    for b, title in have:
        m = result["components"][b]["metrics"]
        lines += ["", f"## {title}", "", "| metric | " + " | ".join(LABEL[s] for s in systems) + " |", "| --- |" + " ---: |" * len(systems)]
        for name in m[ref]:
            row = [m[s][name] for s in systems]
            if all(r["n"] == 0 for r in row):
                continue
            lines.append(f"| {name} | " + " | ".join(cell(r) if r.get("k") is not None else f"{r['value']}" for r in row) + " |")
        lines += ["", "| AuditKIT (all episodes) | " + " | ".join(LABEL[s] for s in systems) + " |", "| --- |" + " ---: |" * len(systems)]
        for k in AUDIT:
            lines.append(f"| {AUDIT_LABEL.get(k, k)} | " + " | ".join("n/a" if audit[(b, s)].get(k) is None else f"{audit[(b, s)][k]:.3f}" for s in systems) + " |")
    tag = "test" if args.suite == "test" else "informed"
    (ROOT / "reports" / f"round3_{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (ROOT / "reports" / f"round3_{tag}.json").write_text(json.dumps(result, indent=1), encoding="utf-8", newline="\n")
    print("\n".join(lines[: 14 + (len(have) if new else 0) + 12]))


if __name__ == "__main__":
    main()
