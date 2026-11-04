"""Build the before-and-after scorecard from run_eval outputs.

  python scripts/scorecard.py --split test --tag lenient --before base --after main --control control \
      --run base=runs/eval/base_test/lenient --run main=runs/eval/main_test/lenient --run control=runs/eval/control_test/lenient

Writes reports/scorecard_<split>[_<tag>].json and .md. Run once per parse mode (lenient = first valid action, strict = protocol compliance). Use dev numbers for choices and test only once.
"""

import argparse
import json
from pathlib import Path

from ragharness import scorecard

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--run", action="append", required=True, help="name=directory (repeat)")
    ap.add_argument("--before", default="base")
    ap.add_argument("--after", default="main")
    ap.add_argument("--control", action="append", default=[])
    ap.add_argument("--product", action="store_true", help="the --run directories are product-suite runs: print the per-kind table instead")
    ap.add_argument("--tag", default="", help="suffix for the output files, e.g. lenient or strict")
    args = ap.parse_args()
    systems = {}
    for item in args.run:
        name, path = item.split("=", 1)
        systems[name] = Path(path)
    if args.product:
        md = "# Nimbus Home support suite (fictional, 24 tasks, exploratory)\n\n" + scorecard.product_table(systems) + "\n"
        tag = args.tag or "all"
        (ROOT / "reports" / f"scorecard_product_{tag}.md").write_text(md, encoding="utf-8", newline="\n")
        print(md)
        return
    card = scorecard.build(systems, args.before, args.after, args.control)
    card["split"] = args.split
    suffix = f"_{args.tag}" if args.tag else ""
    out = ROOT / "reports"
    (out / f"scorecard_{args.split}{suffix}.json").write_text(json.dumps(card, indent=2), encoding="utf-8", newline="\n")
    md = scorecard.markdown(card, f"Scorecard ({args.split}): {args.after} against {args.before}")
    (out / f"scorecard_{args.split}{suffix}.md").write_text(md, encoding="utf-8", newline="\n")
    print(md)


if __name__ == "__main__":
    main()
