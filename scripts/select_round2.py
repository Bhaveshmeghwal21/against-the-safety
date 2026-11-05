"""Apply the second-round selection rule (plan section 20) to development results -> reports/round2_selection.md.

  python scripts/select_round2.py --run main=<dir> --run candA=<dir> --run candB=<dir>

Each directory holds a lenient-reading scores.jsonl. The rule is fixed in plan section 20; this script only applies it.
"""

import argparse
import json
from pathlib import Path

from ragharness import scorecard

ROOT = Path(__file__).resolve().parents[1]
MAX_UNSUPPORTED = 11  # of the 73 unanswerable dev headline episodes (0.15)
MAX_INVALID_RATE = 0.10


def stats(path: Path) -> dict:
    scores, _ = scorecard.load(path)
    head = [s for s in scores if s["variant"] in ("open", "support_masked")]
    ans = [s for s in scores if s["answerable"]]
    un = [s for s in head if not s["answerable"]]
    return {
        "n": len(head), "good": sum(bool(s["good"]) for s in head),
        "unsupported": sum(s["final"] == "answer" for s in un), "n_unanswerable": len(un),
        "invalid": sum(s["outcome"] == "invalid" for s in head),
        "supported": sum(s["outcome"] == "supported_correct" for s in ans), "n_answerable": len(ans),
        "false_abstain": sum(s["outcome"] == "false_abstain" for s in ans),
        "correct_abstain": sum(s["outcome"] == "correct_abstain" for s in un),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="name=directory; names main, candA, candB")
    args = ap.parse_args()
    st = {}
    for item in args.run:
        name, path = item.split("=", 1)
        st[name] = stats(Path(path))
    for name, s in st.items():
        s["eligible"] = s["unsupported"] <= MAX_UNSUPPORTED and s["invalid"] / s["n"] <= MAX_INVALID_RATE
    cands = [c for c in ("candA", "candB") if c in st]
    order = {"candA": 0, "candB": 1}
    key = lambda c: (st[c]["good"], st[c]["supported"] / st[c]["n_answerable"], -order[c])
    eligible = [c for c in cands if st[c]["eligible"]]
    best = max(eligible or cands, key=key)
    model = best if st[best]["eligible"] and st[best]["good"] >= st["main"]["good"] else "main"

    lines = [
        "# Second-round selection on the development split (rule fixed in plan section 20)", "",
        f"Eligible: unsupported answers on unanswerable headline episodes at most {MAX_UNSUPPORTED} of 73, and at most "
        f"{MAX_INVALID_RATE:.0%} of headline episodes without a valid final turn.",
        "Best candidate: the eligible one with the higher end-to-end good rate (ties: higher supported-correct rate, then candA).",
        "Second-round model: the best candidate if its end-to-end good rate is at least main's, otherwise main.", "",
        "| system | end-to-end good (headline) | supported correct (answerable) | false abstain (answerable) | correct abstain (unanswerable headline) | unsupported answers (unanswerable headline) | no valid final turn (headline) | eligible |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, s in st.items():
        lines.append(
            f"| {name} | {s['good'] / s['n']:.3f} ({s['good']}/{s['n']}) | {s['supported'] / s['n_answerable']:.3f} ({s['supported']}/{s['n_answerable']}) "
            f"| {s['false_abstain'] / s['n_answerable']:.3f} ({s['false_abstain']}/{s['n_answerable']}) "
            f"| {s['correct_abstain'] / s['n_unanswerable']:.3f} ({s['correct_abstain']}/{s['n_unanswerable']}) "
            f"| {s['unsupported']}/{s['n_unanswerable']} | {s['invalid']}/{s['n']} | {'yes' if s['eligible'] else 'no'} |"
        )
    lines += ["", f"**Best candidate: {best}**{'' if st[best]['eligible'] else ' (no candidate was eligible; the higher-scoring one is still evaluated on test2 as a disclosed secondary)'}.",
              f"**Second-round model: {model}.** Both {best} and main are evaluated on test2.", ""]
    (ROOT / "reports" / "round2_selection.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    (ROOT / "reports" / "round2_selection.json").write_text(json.dumps({"stats": st, "best_candidate": best, "round2_model": model}, indent=2), encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
