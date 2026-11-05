"""Dev-split table behind the epoch choice (plan section 18: pick the epoch with the higher main end-to-end good rate on dev).

  python scripts/epoch_table.py --run "main, epoch 1=runs/.../main_adapter_ep1_dev/lenient" --run "main, epoch 2=runs/.../main_dev/lenient" ...

Writes reports/dev_epoch_selection.md. Every column is a separate rate with its denominator.
"""

import argparse
from pathlib import Path

from ragharness import scorecard

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", default="reports/dev_epoch_selection.md")
    args = ap.parse_args()
    cols = ["system", "end-to-end good", "supported_correct (answerable)", "false_abstain (answerable)", "correct_abstain (unanswerable)",
            "unsupported answer (unanswerable)", "invalid turn", "mean searches"]
    lines = ["# Dev split: epoch selection (headline episodes: open and support-masked)", "",
             "| " + " | ".join(cols) + " |", "| --- |" + " ---: |" * (len(cols) - 1)]
    for item in args.run:
        name, path = item.split("=", 1)
        scores, _ = scorecard.load(Path(path))
        head = [s for s in scores if s["variant"] in ("open", "support_masked")]
        ans = [s for s in head if s["answerable"]]
        una = [s for s in head if not s["answerable"]]
        cell = lambda rows, pred: scorecard.fmt(scorecard._rate(rows, pred))
        lines.append("| " + " | ".join([
            name, cell(head, lambda s: s["good"]), cell(ans, lambda s: s["outcome"] == "supported_correct"), cell(ans, lambda s: s["outcome"] == "false_abstain"),
            cell(una, lambda s: s["outcome"] == "correct_abstain"), cell(una, lambda s: s["final"] == "answer"), cell(head, lambda s: s["outcome"] == "invalid"),
            f"{sum(s['n_searches'] for s in head) / len(head):.2f}"]) + " |")
    (ROOT / args.out).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
