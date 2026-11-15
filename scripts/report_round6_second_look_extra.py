"""Post hoc numbers for the second-look test, written after the registered report was produced (not part of the pre-registered decision).

  python scripts/report_round6_second_look_extra.py --run runs/kaggle/ats-r6-system-sl/out [--out reports/round6_second_look_extra]

1. The submitted system (main + FB + V) against refusing every question, with the same paired cluster bootstrap as the registered report.
2. The four macro components for each system next to the same components on the opened Round 5 test (`reports/reference_policies.json`).
   Pooled "correct and cited" is not comparable between the two tests because their mix of SQuAD, HotpotQA and RGB episodes differs; the components are.
"""

import argparse
import json
from pathlib import Path

import report_azure_compare as azure
import report_round6_second_look as R

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="reports/round6_second_look_extra")
    args = ap.parse_args()
    root = ROOT / args.run
    systems, _ = R.build_systems(root)
    indexed, keys = R.shared_keys(systems)
    pairs = [(R.REFERENCE, "always refuse"), (R.HEADLINE, "always refuse")]
    comparisons = R.cluster_intervals(indexed, keys, pairs, R.cluster_groups())
    stats = {name: azure.r5_stats(rows, keys) for name, rows in indexed.items()}
    r5 = json.loads((ROOT / "reports/reference_policies.json").read_text(encoding="utf-8"))["sealed_r5test"]
    components = ("single", "compound", "hotpot", "rgb")
    episodes = {c: sum(1 for k in keys if indexed[R.ACTOR][k]["comp"] == c) for c in components}
    report = {"note": "post hoc, not part of the pre-registered decision", "episodes_per_component": episodes,
              "comparisons": comparisons,
              "round6_components": {name: {c: s["components"][c] for c in components} for name, s in stats.items()},
              "round5_components": {name: {c: r5[name]["components"][c] for c in components} for name in ("main", "main+FB+V", "always_refuse")}}
    out = ROOT / args.out
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pct = lambda v: f"{100 * v:.1f}%"
    ci = lambda m: f"{100 * m['difference']:+.1f} pp [{100 * m['ci95'][0]:+.1f}, {100 * m['ci95'][1]:+.1f}]"
    lines = ["# Round 6 second-look test: post hoc numbers", "", "Written after the registered report. Not part of the decision.", "",
             "## The submitted system against refusing every question (paired, same draws)", "",
             "| Comparison | Macro | Correct and cited | Hidden-answer rate |", "| --- | ---: | ---: | ---: |"]
    for name, c in comparisons.items():
        lines.append(f"| {name} | {ci(c['macro'])} | {ci(c['supported_correct'])} | {ci(c['unsupported_rate'])} |")
    lines += ["", "## Success per component, Round 6 test and the opened Round 5 test", "",
              f"Episodes per component on the Round 6 test: {episodes}.", "",
              "| System | Test | SQuAD single | SQuAD two-part | HotpotQA | RGB |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for name in ("main+FB+V", "always refuse"):
        r5_name = "always_refuse" if name == "always refuse" else name
        lines.append(f"| {name} | Round 6 | " + " | ".join(pct(report["round6_components"][name][c]) for c in components) + " |")
        lines.append(f"| {name} | Round 5 | " + " | ".join(pct(report["round5_components"][r5_name][c]) for c in components) + " |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(f"Required saved artifact is missing: {exc.filename or exc}") from exc
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
