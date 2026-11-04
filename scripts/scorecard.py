"""Build the before-and-after scorecard from run_eval outputs.

  python scripts/scorecard.py --split test --before base --after main --control control --closed closed \
      --run base=runs/eval/base_test --run main=runs/eval/main_test --run control=runs/eval/control_test

Writes reports/scorecard_<split>.json and reports/scorecard_<split>.md. Use dev numbers for choices and test only once.
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
    args = ap.parse_args()
    systems = {}
    for item in args.run:
        name, path = item.split("=", 1)
        systems[name] = Path(path)
    card = scorecard.build(systems, args.before, args.after, args.control)
    card["split"] = args.split
    out = ROOT / "reports"
    (out / f"scorecard_{args.split}.json").write_text(json.dumps(card, indent=2), encoding="utf-8", newline="\n")
    md = scorecard.markdown(card, f"Scorecard ({args.split}): {args.after} against {args.before}")
    (out / f"scorecard_{args.split}.md").write_text(md, encoding="utf-8", newline="\n")
    print(md)


if __name__ == "__main__":
    main()
