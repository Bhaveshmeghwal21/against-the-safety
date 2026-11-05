"""ROI comparison (optional in the competition, preferred): what the fine-tuning costs and what a query costs afterwards.

  python scripts/roi.py --split test --run base=runs/eval/base_test/lenient --run main=runs/eval/main_test/lenient \
      --run control=runs/eval/control_test/lenient --train main=runs/kaggle/ats-train-both/out/main/train_report.json \
      --train control=runs/kaggle/ats-train-both/out/control/train_report.json

Only measured quantities enter the table: seconds per episode (amortized over each batched model call), model calls and tokens
per episode, and the outcome rates the scorer produced. Prices are an input (`--price`, default 0.35 USD per T4 GPU-hour, the
Google Cloud on-demand list price; rented T4s range from 0.16 to 0.59 in 2026). Training ran on Kaggle's free quota, so the real
training cost was zero; the table also shows what the same GPU-hours cost at the reference price.

Cost per supported-correct answer = (cost per episode) / (rate of end-to-end good outcomes). It divides by the headline rate,
so a system that is cheap but wrong costs more per good outcome than one that is slower and right.
"""

import argparse
import json
from pathlib import Path

from ragharness import scorecard

ROOT = Path(__file__).resolve().parents[1]
HEADLINE = ("open", "support_masked")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--train", action="append", default=[], help="name=train_report.json")
    ap.add_argument("--price", type=float, default=0.35, help="USD per T4 GPU-hour")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    runs = {n: Path(p) for n, p in (r.split("=", 1) for r in args.run)}
    train = {n: json.loads(Path(p).read_text(encoding="utf-8")) for n, p in (r.split("=", 1) for r in args.train)}
    per_s = args.price / 3600.0

    rows = {}
    for name, d in runs.items():
        scores, summary = scorecard.load(d)
        head = [s for s in scores if s["variant"] in HEADLINE]
        ans = [s for s in head if s["answerable"]]
        good = sum(1 for s in head if s["good"])
        n = len(head)
        sec = sum(s["seconds"] for s in head) / n
        rows[name] = {
            "episodes": n, "seconds": sec, "calls": sum(s["model_calls"] for s in head) / n,
            "in_tok": sum(s["input_tokens"] for s in head) / n, "out_tok": sum(s["output_tokens"] for s in head) / n,
            "good_rate": good / n, "supported_correct": sum(1 for s in ans if s["outcome"] == "supported_correct") / max(1, len(ans)),
            "needs_review": sum(1 for s in head if not s["good"]) / n,
            "cost_per_1k": sec * per_s * 1000,
            "cost_per_good": (sec * per_s) / (good / n) if good else None,
        }

    lines = [f"# ROI ({args.split}, headline episodes: open and support-masked)", "",
             f"Reference price {args.price:.2f} USD per T4 GPU-hour (range 0.16 to 0.59). Seconds are amortized wall time on a shared, batched T4; they depend on batch size and are for comparison between rows, not a latency promise.", "",
             "| system | seconds / episode | model calls / episode | input tokens / episode | output tokens / episode | end-to-end good | supported_correct (answerable) | episodes needing review per 1,000 | GPU cost per 1,000 episodes (USD) | GPU cost per good outcome (USD) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, r in rows.items():
        cpg = "n/a" if r["cost_per_good"] is None else f"{r['cost_per_good']:.5f}"
        lines.append(f"| {name} | {r['seconds']:.2f} | {r['calls']:.2f} | {r['in_tok']:.0f} | {r['out_tok']:.0f} | {r['good_rate']:.3f} | {r['supported_correct']:.3f} | "
                     f"{1000 * r['needs_review']:.0f} | {r['cost_per_1k']:.3f} | {cpg} |")
    if train:
        lines += ["", "## Training cost", "", "| run | rows trained | processed tokens | GPU-hours | actual cost (Kaggle free quota) | cost at reference price (USD) | peak GPU memory (GB) |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for name, t in train.items():
            hrs = t["wall_seconds"] / 3600
            lines.append(f"| {name} | {t['rows_trained']} | {t['processed_tokens']:,} | {hrs:.2f} | 0.00 | {hrs * args.price:.2f} | {t['peak_gpu_memory_gb']} |")
        if "main" in rows and "base" in rows and "main" in train:
            gain = rows["main"]["good_rate"] - rows["base"]["good_rate"]
            cost = train["main"]["wall_seconds"] / 3600 * args.price
            lines += ["", "## Break-even", ""]
            if gain > 0:
                per_ep = rows["main"]["cost_per_1k"] / 1000
                lines.append(f"The tuned model produces {gain * 1000:.0f} more end-to-end good outcomes per 1,000 episodes than the base model. Training the main adapter costs {cost:.2f} USD at the reference price, "
                             f"which equals the GPU cost of about {cost / per_ep:,.0f} tuned-model episodes. Whether it pays back depends on what a bad outcome costs the operator, which this project did not measure. "
                             "Illustration only, with an assumed cost per avoided bad outcome:")
                lines += ["", "| assumed cost of one bad outcome (USD) | queries needed to recover the training cost |", "| ---: | ---: |"]
                for c in (0.10, 1.00, 5.00):
                    lines.append(f"| {c:.2f} | {cost / (c * gain):,.1f} |")
            else:
                lines.append("The tuned model did not improve the end-to-end good rate on these episodes; no break-even applies.")
    md = "\n".join(lines) + "\n"
    suffix = f"_{args.tag}" if args.tag else ""
    (ROOT / "reports" / f"roi_{args.split}{suffix}.md").write_text(md, encoding="utf-8", newline="\n")
    print(md)


if __name__ == "__main__":
    main()
