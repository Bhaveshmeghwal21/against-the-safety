"""What each system puts in front of the user: the accuracy-coverage view of the headline episodes (open and support-masked).

  python scripts/tradeoff.py --split test --tag lenient --run base=runs/.../base_test/lenient --run main=... --run control=...

Per system, over the same headline episodes, every episode ends in exactly one of five ways:
  supported answer   answered, F1 >= 0.5 against gold, cited a shown gold passage
  bad answer         answered but wrong or unsupported (answerable), or answered when the evidence was missing (unanswerable)
  correct abstain    abstained when the evidence was missing
  false abstain      abstained although the evidence was reachable
  no valid final     unparseable turn, repeated query, or search past the budget
"precision of delivered answers" is supported answers divided by all answers delivered. Writes reports/tradeoff_<split>_<tag>.md.
"""

import argparse
from pathlib import Path

from ragharness import scorecard

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--run", action="append", required=True)
    args = ap.parse_args()
    names, rows = [], {}
    for item in args.run:
        name, path = item.split("=", 1)
        scores, _ = scorecard.load(Path(path))
        head = [s for s in scores if s["variant"] in ("open", "support_masked")]
        n = len(head)
        supported = sum(s["outcome"] == "supported_correct" for s in head)
        bad = sum(s["final"] == "answer" and s["outcome"] != "supported_correct" for s in head)
        c_abs = sum(s["outcome"] == "correct_abstain" for s in head)
        f_abs = sum(s["outcome"] == "false_abstain" for s in head)
        invalid = sum(s["outcome"] == "invalid" for s in head)
        assert supported + bad + c_abs + f_abs + invalid == n, "outcomes must partition the episodes"
        delivered = supported + bad
        una = [s for s in head if not s["answerable"]]
        names.append(name)
        rows[name] = (n, supported, bad, c_abs, f_abs, invalid, delivered, sum(s["final"] != "answer" for s in una), len(una))
    n_ans = sum(s["answerable"] for s in head)
    lines = [f"# Accuracy and coverage ({args.split}, headline episodes: {n} = {n_ans} answerable + {n - n_ans} with the evidence masked)", "",
             "Each episode lands in exactly one row. Counts, then per 100 episodes.", "",
             "| outcome | " + " | ".join(names) + " |", "| --- |" + " ---: |" * len(names)]
    labels = [("supported answer (good)", 1), ("bad answer: wrong, unsupported, or answered with the evidence missing", 2), ("correct abstain (good)", 3),
              ("false abstain", 4), ("no valid final turn", 5)]
    for label, i in labels:
        lines.append(f"| {label} | " + " | ".join(f"{rows[x][i]} ({100 * rows[x][i] / rows[x][0]:.1f})" for x in names) + " |")
    lines.append("| answers delivered | " + " | ".join(f"{rows[x][6]} ({100 * rows[x][6] / rows[x][0]:.1f})" for x in names) + " |")
    lines.append("| precision of delivered answers (supported / delivered) | " + " | ".join(f"{rows[x][1] / rows[x][6]:.3f}" if rows[x][6] else "n/a" for x in names) + " |")
    lines.append("| evidence missing: no answer delivered (explicit abstain, or the episode ended without a valid final turn) | " + " | ".join(f"{rows[x][7]}/{rows[x][8]} ({100 * rows[x][7] / rows[x][8]:.1f}%)" for x in names) + " |")
    lines.append("| evidence missing: explicit abstain | " + " | ".join(f"{rows[x][3]}/{rows[x][8]} ({100 * rows[x][3] / rows[x][8]:.1f}%)" for x in names) + " |")
    # every evidence state separately, because the headline mixes two of them
    all_scores = {}
    for item in args.run:
        name, path = item.split("=", 1)
        all_scores[name] = scorecard.load(Path(path))[0]
    variants = sorted({s["variant"] for s in all_scores[names[0]]})
    lines += ["", "## By evidence state (all evaluated episodes)", "", "Each cell: supported answer / bad answer / correct abstain / false abstain / no valid final turn, out of n.", "",
              "| evidence state | answerable | n | " + " | ".join(names) + " |", "| --- | --- | ---: |" + " --- |" * len(names)]
    for v in variants:
        cells = []
        for name in names:
            ss = [s for s in all_scores[name] if s["variant"] == v]
            k = [sum(s["outcome"] == "supported_correct" for s in ss), sum(s["final"] == "answer" and s["outcome"] != "supported_correct" for s in ss),
                 sum(s["outcome"] == "correct_abstain" for s in ss), sum(s["outcome"] == "false_abstain" for s in ss), sum(s["outcome"] == "invalid" for s in ss)]
            cells.append(" / ".join(map(str, k)))
        first = [s for s in all_scores[names[0]] if s["variant"] == v]
        lines.append(f"| {v} | {"yes" if first[0]["answerable"] else "no"} | {len(first)} | " + " | ".join(cells) + " |")
    md = "\n".join(lines) + "\n"
    suffix = f"_{args.tag}" if args.tag else ""
    (ROOT / "reports" / f"tradeoff_{args.split}{suffix}.md").write_text(md, encoding="utf-8", newline="\n")
    print(md)


if __name__ == "__main__":
    main()
