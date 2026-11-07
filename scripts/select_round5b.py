"""Round 5b development selection (research/2026-11-07-round5b-preregistration.md), scorer version 2, fail-closed and the checker V on every system.

  python scripts/select_round5b.py --eval <dev eval root> [--eval ...] --cases <cases> [...] --verdicts <folder> [...] --out reports/round5b_selection

Same inputs and the same statistics as scripts/select_round5.py. What changes is the gate, written down before the new checkpoints were trained:
  1. abstention retained   the model's own abstentions (before FB and V) on hidden-evidence development episodes are at least 0.75 of main's share
  2. unsupported pooled    unsupported-answer rate on hidden-evidence episodes (with FB and V) is at most main's + 0.03
  3. unsupported HotpotQA  at most main's + 2 episodes on HotpotQA dev
  4. supported correct     at least main's + 0.05
  5. invalid turns         before FB, at most main's + 1 point
  6. HotpotQA success      at least main's - 3 points
  7. macro                 at least main's + 3 points
The highest macro among the eligible wins; macro ties within 0.005 go to fewer unsupported answers, then to the name. Nothing here reads a test set.
"""

import argparse
import json
from pathlib import Path

import rescore_round5 as RS
import select_round5 as S

ALL = "candE1,candE2,candF1,candF2,candG1,candG2,candH1,candH2,candI1,candI2,candK1,candK2"


def hidden_rows(bench, hotpot):
    return [r for r in bench if not r["answerable"] and not r["excluded_v2"]] + [r for r in hotpot if r["variant"] in ("open", "support_masked") and not r["answerable"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="append", required=True)
    ap.add_argument("--candidates", default=ALL)
    ap.add_argument("--references", default="main,candA")
    ap.add_argument("--out", default="reports/round5b_selection")
    ap.add_argument("--cases", action="append", default=[])
    ap.add_argument("--verdicts", action="append", default=[])
    args = ap.parse_args()
    cases, verdict = S.load_v(args.cases, args.verdicts) if args.cases else ({}, {})
    runs = {}
    for root in args.eval:
        for folder in sorted(Path(root).rglob("lenient")):
            if (folder / "transcripts.jsonl").exists():
                runs[folder.parent.name] = folder
    records = {}
    for name in args.references.split(",") + args.candidates.split(","):
        if f"{name}_bench_r4dev" not in runs or f"{name}_dev" not in runs:
            print(f"missing development runs for {name}; skipped")
            continue
        _, _, bench, _ = RS.rescore_run(runs[f"{name}_bench_r4dev"])
        _, _, hotpot, _ = RS.rescore_run(runs[f"{name}_dev"])
        hidden = hidden_rows(bench, hotpot)
        raw_abstain = sum(r["final"] == "abstain" for r in hidden) / len(hidden)
        raw_invalid = sum(r["final"] == "invalid" for r in bench) / len(bench)
        bench, hotpot = S.fail_closed(bench), S.fail_closed(hotpot)
        if cases:
            bench, hotpot = S.apply_v(bench, f"{name}_bench_r4dev", cases, verdict), S.apply_v(hotpot, f"{name}_dev", cases, verdict)
        records[name] = S.stats(bench, hotpot)
        records[name].update(invalid_before_fb=raw_invalid, own_abstain_share=raw_abstain, hidden_all=len(hidden))
    base = records["main"]
    eligible = []
    for name in args.candidates.split(","):
        if name not in records:
            continue
        r = records[name]
        gates = {
            "abstention_retained_0.75": r["own_abstain_share"] >= 0.75 * base["own_abstain_share"],
            "unsupported_rate_within_3pts": r["unsupported_rate"] <= base["unsupported_rate"] + 0.03,
            "hotpot_unsupported_within_2": r["hotpot_unsupported"] <= base["hotpot_unsupported"] + 2,
            "supported_correct_plus_5pts": r["supported_correct"] >= base["supported_correct"] + 0.05,
            "invalid_not_more_than_plus_1pt": r["invalid_before_fb"] <= base["invalid_before_fb"] + 0.01,
            "hotpot_success_not_lower_by_3pts": r["hotpot_success"] >= base["hotpot_success"] - 0.03,
            "macro_plus_3pts": r["macro"] >= base["macro"] + 0.03,
        }
        r["gates"], r["eligible"] = gates, all(gates.values())
        if r["eligible"]:
            eligible.append(name)
    winner = None
    if eligible:
        best = max(records[n]["macro"] for n in eligible)
        tied = [n for n in eligible if records[n]["macro"] >= best - 0.005]
        winner = min(tied, key=lambda n: (records[n]["unsupported"], n))
    result = {"selected": winner, "scorer": "v2", "fail_closed_applied": True, "checker_applied": bool(cases), "records": records,
              "note": "Development selection only; at most the winner is opened on the sealed final test."}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Round 5b development selection (scorer v2, fail-closed and the checker on every system)", "",
             "| system | macro | single | compound | HotpotQA | supported correct | unsupported / hidden | HotpotQA unsupported | own abstentions on hidden | invalid before FB | eligible |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for n, r in records.items():
        lines.append(f"| {n} | {r['macro']:.3f} | {r['single']:.3f} | {r['compound']:.3f} | {r['hotpot_success']:.3f} | {r['supported_correct']:.3f} | "
                     f"{r['unsupported']}/{r['hidden_n']} | {r['hotpot_unsupported']} | {r['own_abstain_share']:.3f} | {r['invalid_before_fb']:.3f} | {r.get('eligible', 'reference')} |")
    lines += ["", f"Selected for the sealed final test: **{winner or 'none'}**.", ""]
    for n, r in records.items():
        if "gates" in r:
            lines.append(f"- {n}: " + ", ".join(f"{g}={'yes' if ok else 'NO'}" for g, ok in r["gates"].items()))
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
