"""Aggregate the general-capability check (research/2026-11-16-general-capability-preregistration.md, section 4).

  python scripts/report_general.py --run runs/kaggle/ats-general-t4x2/out/general --out reports/general_capability

Per benchmark: accuracy of base, main and candF1 with 95% bootstrap intervals over questions, and the paired difference of each adapter from the base model
(3,000 draws, seed 47, the same resampled questions for both systems). The registered rule: an adapter is retained on MMLU, ARC-Challenge and ARC-Easy when
the lower end of its paired interval is at least -3.0 points. TruthfulQA and the Global-MMLU-Lite sets have no rule. The report refuses to run if a system is
missing a question, so a partial run cannot be mistaken for a result.
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCHES = {"mmlu": "MMLU (English)", "arc_c": "ARC-Challenge (English)", "arc_e": "ARC-Easy (English)", "truthfulqa": "TruthfulQA MC1 (English)",
           "gmmlu_en": "Global-MMLU-Lite, English", "gmmlu_hi": "Global-MMLU-Lite, Hindi", "gmmlu_es": "Global-MMLU-Lite, Spanish"}
RULE_BENCHES = ("mmlu", "arc_c", "arc_e")
SYSTEMS = ("base", "main", "candF1")
MARGIN = -0.03


def load(run: Path, bench: str, expected: int) -> dict:
    by_system = {s: {} for s in SYSTEMS}
    for line in (run / f"{bench}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            by_system[r["system"]][r["id"]] = r["correct"]
    ids = sorted(by_system["base"])
    for s, d in by_system.items():
        if len(d) != expected or sorted(d) != ids:
            raise SystemExit(f"{bench}: system {s} has {len(d)} of {expected} questions")
    return {s: np.array([by_system[s][i] for i in ids], dtype=np.float64) for s in SYSTEMS}


def bootstrap(arrays: dict, draws: int = 3000, seed: int = 47, chunk: int = 250):
    """Means of each system and paired differences from base over `draws` resamples of the questions."""
    n = len(arrays["base"])
    rng = np.random.default_rng(seed)
    means = {s: [] for s in SYSTEMS}
    for start in range(0, draws, chunk):
        idx = rng.integers(0, n, size=(min(chunk, draws - start), n))
        for s in SYSTEMS:
            means[s].append(arrays[s][idx].mean(axis=1))
    means = {s: np.concatenate(v) for s, v in means.items()}
    interval = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    return ({s: interval(means[s]) for s in SYSTEMS},
            {s: interval(means[s] - means["base"]) for s in SYSTEMS if s != "base"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/kaggle/ats-general-t4x2/out/general")
    ap.add_argument("--out", default="reports/general_capability")
    args = ap.parse_args()
    manifest = json.loads((ROOT / "data/general/manifest.json").read_text(encoding="utf-8"))
    result = {"protocol": "zero-shot letter scoring, plain text, no chat template, 4-bit; base = adapters switched off", "margin_points": MARGIN * 100, "benchmarks": {}, "rules": {}}
    for bench in BENCHES:
        arrays = load(ROOT / args.run, bench, manifest[bench]["questions"])
        own, paired = bootstrap(arrays)
        acc = {s: float(arrays[s].mean()) for s in SYSTEMS}
        result["benchmarks"][bench] = {"title": BENCHES[bench], "questions": len(arrays["base"]), "license": manifest[bench]["license"],
                                       "accuracy": acc, "accuracy_ci95": own,
                                       "difference_from_base": {s: {"difference": acc[s] - acc["base"], "ci95": paired[s]} for s in paired}}
    for adapter in ("main", "candF1"):
        result["rules"][adapter] = {b: {"lower_end": result["benchmarks"][b]["difference_from_base"][adapter]["ci95"][0],
                                        "retained": result["benchmarks"][b]["difference_from_base"][adapter]["ci95"][0] >= MARGIN} for b in RULE_BENCHES}
    summaries = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / args.run).glob("summary_*.json"))]
    result["environment"] = {"gpu": summaries[0]["gpu"], "torch": summaries[0]["torch"], "model_source": summaries[0]["model_source"],
                             "seconds": {f"{r['bench']}/{r['system']}": r["seconds"] for s in summaries for r in s["results"]}} if summaries else None
    (ROOT / args.out).with_suffix(".json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

    pct = lambda x: f"{x * 100:.1f}"
    lines = ["# General-capability retention check", "",
             "Zero-shot multiple choice, plain text, no chat template, 4-bit, one run per system. The same prompt goes to all three systems; `base` is Tiny Aya Global with the adapters "
             "switched off. Accuracy in percent, 95% bootstrap interval over questions in brackets. Absolute scores are not comparable with published ones (other prompts and protocols); "
             "only differences within this table carry meaning. The pre-registration is [here](../research/2026-11-16-general-capability-preregistration.md).", "",
             "| Benchmark (language, role) | Questions | Base | Main (submitted) | candF1 (alternative) |", "| --- | ---: | ---: | ---: | ---: |"]
    for b, r in result["benchmarks"].items():
        cells = [f"{pct(r['accuracy'][s])} [{pct(r['accuracy_ci95'][s][0])}, {pct(r['accuracy_ci95'][s][1])}]" for s in SYSTEMS]
        lines.append(f"| {r['title']} | {r['questions']:,} | " + " | ".join(cells) + " |")
    lines += ["", "## Difference from the base model (points, paired 95% interval)", "", "| Benchmark | Main minus base | candF1 minus base |", "| --- | ---: | ---: |"]
    for b, r in result["benchmarks"].items():
        d = r["difference_from_base"]
        lines.append(f"| {r['title']} | " + " | ".join(f"{d[a]['difference'] * 100:+.1f} [{d[a]['ci95'][0] * 100:+.1f}, {d[a]['ci95'][1] * 100:+.1f}]" for a in ("main", "candF1")) + " |")
    lines += ["", f"## Registered rule: the lower end of the interval must be at least {MARGIN * 100:.1f} points", "",
              "| Benchmark | Main: lower end | Main | candF1: lower end | candF1 |", "| --- | ---: | --- | ---: | --- |"]
    for b in RULE_BENCHES:
        m, c = result["rules"]["main"][b], result["rules"]["candF1"][b]
        lines.append(f"| {BENCHES[b]} | {m['lower_end'] * 100:+.1f} | {'retained' if m['retained'] else 'not retained'} | {c['lower_end'] * 100:+.1f} | {'retained' if c['retained'] else 'not retained'} |")
    (ROOT / args.out).with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
