"""Round 5 final-test report (research/2026-11-07-round5-preregistration.md): scorer v2, wrappers applied offline, paired intervals.

  python scripts/report_round5.py --eval <root holding <system>_bench_r5t_{squad,hotpot,rgb}> --systems base,main,candA,candG2 \
      --cases <cases file> --verdicts <folder with verdicts_A.jsonl> --compare main+FB/main --compare main+FB+V/main+FB --out reports/round5_final

A configuration is `system`, `system+FB` (invalid turn ends as an abstention) or `system+FB+V` (also: answers the checker blocks end as abstentions).
Macro = mean of the component success rates found (SQuAD single, SQuAD compound, HotpotQA part, RGB part) under scorer v2 (a correct answer that
cites every required aspect, or an abstention on hidden evidence). Unsupported rate = unsupported answers / hidden-evidence episodes, pooled.
Intervals: whole questions resampled within each part, 3,000 draws, seed 47, the same draws for both sides of a comparison.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import rescore_round5 as RS
import select_round5 as SEL

COMPONENTS = ("single", "compound", "hotpot", "rgb")


def component(part: str, qtype: str) -> str:
    return "hotpot" if part == "r5t_hotpot" else "rgb" if part == "r5t_rgb" else qtype


def load_system(runs, system, fb, v, cases, verdict):
    rows = []
    for part in sorted({p for (s, p) in runs if s == system}):
        _, _, rr, _ = RS.rescore_run(runs[(system, part)])
        if fb:
            rr = SEL.fail_closed(rr)
        if v:
            rr = SEL.apply_v(rr, f"{system}_bench_{part}", cases, verdict)
        rows += [{**r, "part": part, "comp": component(part, r["qtype"])} for r in rr]
    return rows


def vectors(rows, keys):
    """Per-episode numerators for every component and the unsupported rate, aligned to `keys` order."""
    by = {(r["part"], r["qid"], r["variant"]): r for r in rows}
    comps = sorted({r["comp"] for r in rows}, key=COMPONENTS.index)
    out = np.zeros((len(keys), 2 * len(comps) + 4))
    for i, k in enumerate(keys):
        r = by[k]
        if r["excluded_v2"]:
            continue
        c = comps.index(r["comp"])
        out[i, 2 * c], out[i, 2 * c + 1] = r["strong_v2"], 1
        j = 2 * len(comps)
        if not r["answerable"]:
            out[i, j + 1] = 1
            out[i, j] = r["outcome_v2"].startswith("unsupported_")
        else:
            out[i, j + 2], out[i, j + 3] = r["strong_v2"], 1
    return comps, out


def stats_from(total, comps):
    n = len(comps)
    comp = [total[2 * c] / total[2 * c + 1] if total[2 * c + 1] else float("nan") for c in range(n)]
    j = 2 * n
    return {"macro": float(np.nanmean(comp)), "components": dict(zip(comps, comp)),
            "unsupported": int(total[j]), "hidden_n": int(total[j + 1]), "unsupported_rate": float(total[j] / max(1, total[j + 1])),
            "supported_correct": float(total[j + 2] / max(1, total[j + 3]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="append", required=True)
    ap.add_argument("--systems", required=True)
    ap.add_argument("--cases", action="append", default=[])
    ap.add_argument("--verdicts", action="append", default=[])
    ap.add_argument("--compare", action="append", default=[], help="A/B, each side like main+FB+V")
    ap.add_argument("--configs", default="", help="comma list of configurations to tabulate; default every system raw, +FB and (if checked) +FB+V")
    ap.add_argument("--out", default="reports/round5_final")
    ap.add_argument("--reps", type=int, default=3000)
    args = ap.parse_args()
    runs = {}
    for root in args.eval:
        for folder in sorted(Path(root).rglob("lenient")):
            if (folder / "transcripts.jsonl").exists() and "_bench_" in folder.parent.name:
                system, part = folder.parent.name.split("_bench_", 1)
                runs[(system, part)] = folder
    cases, verdict = SEL.load_v(args.cases, args.verdicts) if args.cases else ({}, {})
    systems = args.systems.split(",")
    names = [c for c in args.configs.split(",") if c] or [n for s in systems for n in (s, f"{s}+FB") + ((f"{s}+FB+V",) if cases else ())]
    data = {}
    for name in dict.fromkeys(names + [x for pair in args.compare for x in pair.split("/")]):
        system, *flags = name.split("+")
        try:
            data[name] = load_system(runs, system, "FB" in flags, "V" in flags, cases, verdict)
        except KeyError as e:
            print(f"skipping {name}: no checker verdict or case for {e}")
    keys = sorted({(r["part"], r["qid"], r["variant"]) for r in next(iter(data.values()))})
    for name, rows in data.items():
        assert {(r["part"], r["qid"], r["variant"]) for r in rows} == set(keys), f"{name} covers different episodes"
    clusters = {}
    for i, k in enumerate(keys):
        clusters.setdefault(k[0], {}).setdefault(k[1], []).append(i)
    table = {}
    for name, rows in data.items():
        comps, vec = vectors(rows, keys)
        s = stats_from(vec.sum(axis=0), comps)
        raw = {(r["part"], r["qid"], r["variant"]): r for r in load_system(runs, name.split("+")[0], False, False, {}, {})} if True else {}
        s["invalid_before_repair"] = sum(r["final"] == "invalid" for r in raw.values()) / len(raw)
        s["false_abstain"] = sum(r["outcome_v2"] == "false_abstain" for r in rows if r["answerable"] and not r["excluded_v2"]) / max(1, sum(1 for r in rows if r["answerable"] and not r["excluded_v2"]))
        s["n"] = int(sum(1 for r in rows if not r["excluded_v2"]))
        table[name] = s
    comparisons = {}
    rng = np.random.default_rng(47)
    for pair in args.compare:
        a, b = pair.split("/")
        (ca, va), (cb, vb) = vectors(data[a], keys), vectors(data[b], keys)
        diffs = {"macro": [], "unsupported_rate": [], "supported_correct": []}
        for _ in range(args.reps):
            idx = []
            for part in sorted(clusters):
                ids = sorted(clusters[part])
                for q in rng.choice(len(ids), size=len(ids)):
                    idx += clusters[part][ids[q]]
            sa, sb = stats_from(va[idx].sum(axis=0), ca), stats_from(vb[idx].sum(axis=0), cb)
            diffs["macro"].append(sa["macro"] - sb["macro"])
            diffs["unsupported_rate"].append(sa["unsupported_rate"] - sb["unsupported_rate"])
            diffs["supported_correct"].append(sa["supported_correct"] - sb["supported_correct"])
        comparisons[pair] = {k: {"difference": table[a][k] - table[b][k] if k != "supported_correct" else table[a][k] - table[b][k],
                                 "ci95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]} for k, v in diffs.items()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"systems": table, "comparisons": comparisons, "episodes": len(keys)}, indent=2), encoding="utf-8")
    comps_all = sorted({c for s in table.values() for c in s["components"]}, key=COMPONENTS.index)
    lines = ["# Round 5 final test (scorer v2)", "", f"{len(keys)} episodes. Macro = mean of the component rates.", "",
             "| configuration | macro | " + " | ".join(comps_all) + " | unsupported / hidden | supported correct | false refusals | invalid before repair |",
             "| --- | ---: | " + " | ".join("---:" for _ in comps_all) + " | ---: | ---: | ---: | ---: |"]
    for name, s in table.items():
        lines.append(f"| {name} | {s['macro']:.3f} | " + " | ".join(f"{s['components'].get(c, float('nan')):.3f}" for c in comps_all) +
                     f" | {s['unsupported']}/{s['hidden_n']} | {s['supported_correct']:.3f} | {s['false_abstain']:.3f} | {s['invalid_before_repair']:.3f} |")
    lines += ["", "Paired differences (first minus second), 95% intervals:", ""]
    for pair, c in comparisons.items():
        lines.append(f"- {pair}: macro {c['macro']['difference']:+.3f} [{c['macro']['ci95'][0]:+.3f}, {c['macro']['ci95'][1]:+.3f}]; unsupported rate "
                     f"{c['unsupported_rate']['difference']:+.3f} [{c['unsupported_rate']['ci95'][0]:+.3f}, {c['unsupported_rate']['ci95'][1]:+.3f}]; supported correct "
                     f"{c['supported_correct']['difference']:+.3f} [{c['supported_correct']['ci95'][0]:+.3f}, {c['supported_correct']['ci95'][1]:+.3f}]")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
