"""Round 5 development selection (research/2026-11-04-fine-tuning-for-rag.md section 27, priority 2), on scorer version 2.

  python scripts/select_round5.py --eval <dev eval root> [--eval ...] --candidates candG1,candG2,candH1,candH2 --out reports/round5_selection

Every system is read from its two development runs (<system>_bench_r4dev and <system>_dev), rescored under version 2, and passed through the
fail-closed repair FB (an invalid turn ends as an abstention) on both sides of every comparison. A candidate is eligible only if all six
section-27 gates hold against main; among the eligible, the highest macro wins, ties go to fewer unsupported answers, then to candG.
Nothing here reads a test set.
"""

import argparse
import json
from pathlib import Path

import rescore_round5 as RS


def fail_closed(rows):
    out = []
    for r in rows:
        if r["final"] != "invalid":
            out.append(r)
            continue
        o = "false_abstain" if r["answerable"] else "correct_abstain"
        out.append({**r, "outcome_v2": o, "good_v2": not r["answerable"], "strong_v2": not r["answerable"], "final": "abstain"})
    return out


def load_v(cases_files, verdict_dirs):
    """The answer checker V (priority 1): deterministic flags from the cases, main-adapter wording-A probabilities from the verdicts."""
    cases, verdict = {}, {}
    for f in cases_files:
        for c in RS.read(f):
            cases[(c["run"], c["qid"], c["variant"])] = c
    for d in verdict_dirs:
        for r in RS.read(Path(d) / "verdicts_A.jsonl"):
            verdict[(r["run"], r["qid"], r["variant"])] = r["p_pos"]
    return cases, verdict


def apply_v(rows, run, cases, verdict):
    """Every answer V blocks ends as an abstention (the repair is applied to every system alike)."""
    out = []
    for r in rows:
        if r["final"] == "answer":
            key = (run, r["qid"], r["variant"])
            c = cases[key]
            blocked = c["deterministic"]["cites_unshown"] or c["deterministic"]["answer_part_absent_from_cited"] or key not in verdict or verdict[key] < 0.5
            if blocked:
                r = {**r, "outcome_v2": "false_abstain" if r["answerable"] else "correct_abstain", "good_v2": not r["answerable"],
                     "strong_v2": not r["answerable"], "final": "abstain"}
        out.append(r)
    return out


def stats(bench, hotpot):
    v = "v2"
    bench = [r for r in bench if not r["excluded_v2"]]
    hidden = [r for r in bench if not r["answerable"]]
    ans = [r for r in bench if r["answerable"]]
    hp = [r for r in hotpot if r["variant"] in ("open", "support_masked")]
    hp_hidden = [r for r in hp if not r["answerable"]]
    rate = lambda xs, f: sum(f(x) for x in xs) / len(xs) if xs else None
    uns = lambda rr: sum(r[f"outcome_{v}"].startswith("unsupported_") for r in rr)
    single, compound = (rate([r for r in bench if r["qtype"] == k], lambda r: r["strong_v2"]) for k in ("single", "compound"))
    hotpot_success = rate(hp, lambda r: r["good_v2"])
    return {
        "macro": (single + compound + hotpot_success) / 3, "single": single, "compound": compound, "hotpot_success": hotpot_success,
        "supported_correct": rate(ans, lambda r: r["strong_v2"]), "unsupported": uns(hidden), "hidden_n": len(hidden), "unsupported_rate": uns(hidden) / len(hidden),
        "unsupported_by_family": {f: uns([r for r in hidden if ("partial" in r["variant"]) == p]) for f, p in (("partial", True), ("fully_missing", False))},
        "hotpot_unsupported": uns(hp_hidden), "false_abstain": rate(ans, lambda r: r[f"outcome_{v}"] == "false_abstain"),
        "n": len(bench), "n_hotpot": len(hp),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="append", required=True)
    ap.add_argument("--candidates", default="candG1,candG2,candH1,candH2")
    ap.add_argument("--references", default="main,candA")
    ap.add_argument("--out", default="reports/round5_selection")
    ap.add_argument("--no-fb", action="store_true")
    ap.add_argument("--cases", action="append", default=[], help="verifier cases files; with --verdicts, V is applied to every system")
    ap.add_argument("--verdicts", action="append", default=[], help="folders holding verdicts_A.jsonl from the main-adapter checker")
    args = ap.parse_args()
    cases, verdict = load_v(args.cases, args.verdicts) if args.cases else ({}, {})
    runs = {}
    for root in args.eval:
        for folder in sorted(Path(root).rglob("lenient")):
            if (folder / "transcripts.jsonl").exists():
                runs[folder.parent.name] = folder
    records, raw_invalid = {}, {}
    for name in args.references.split(",") + args.candidates.split(","):
        if f"{name}_bench_r4dev" not in runs or f"{name}_dev" not in runs:
            print(f"missing development runs for {name}; skipped")
            continue
        _, _, bench, _ = RS.rescore_run(runs[f"{name}_bench_r4dev"])
        _, _, hotpot, _ = RS.rescore_run(runs[f"{name}_dev"])
        raw_invalid[name] = sum(r["final"] == "invalid" for r in bench) / len(bench)
        if not args.no_fb:
            bench, hotpot = fail_closed(bench), fail_closed(hotpot)
        if cases:
            bench, hotpot = apply_v(bench, f"{name}_bench_r4dev", cases, verdict), apply_v(hotpot, f"{name}_dev", cases, verdict)
        records[name] = stats(bench, hotpot)
        records[name]["invalid_before_fb"] = raw_invalid[name]
    base = records["main"]
    eligible = []
    for name in args.candidates.split(","):
        if name not in records:
            continue
        r = records[name]
        gates = {
            "fewer_unsupported_pooled": r["unsupported"] < base["unsupported"],
            "no_more_unsupported_in_either_family": all(r["unsupported_by_family"][k] <= base["unsupported_by_family"][k] for k in base["unsupported_by_family"]),
            "hotpot_unsupported_not_more": r["hotpot_unsupported"] <= base["hotpot_unsupported"],
            "supported_correct_not_lower": r["supported_correct"] >= base["supported_correct"],
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
        winner = min(tied, key=lambda n: (records[n]["unsupported"], n.startswith("candH"), n))
    result = {"selected": winner, "fail_closed_applied": not args.no_fb, "checker_applied": bool(cases), "scorer": "v2", "records": records,
              "note": "Development selection only; a winner goes to the sealed final test, nothing is promoted from here."}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Round 5 development selection (scorer v2, fail-closed applied to every system)" if not args.no_fb else "# Round 5 development selection (scorer v2)", "",
             "| system | macro | single | compound | HotpotQA | supported correct | unsupported / hidden | HotpotQA unsupported | invalid before FB | eligible |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for n, r in records.items():
        lines.append(f"| {n} | {r['macro']:.3f} | {r['single']:.3f} | {r['compound']:.3f} | {r['hotpot_success']:.3f} | {r['supported_correct']:.3f} | "
                     f"{r['unsupported']}/{r['hidden_n']} | {r['hotpot_unsupported']} | {r['invalid_before_fb']:.3f} | {r.get('eligible', 'reference')} |")
    lines += ["", f"Selected for the sealed final test: **{winner or 'none'}**.", ""]
    for n, r in records.items():
        if "gates" in r:
            lines.append(f"- {n}: " + ", ".join(f"{g}={'yes' if ok else 'NO'}" for g, ok in r["gates"].items()))
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
