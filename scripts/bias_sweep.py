"""Decoding-time abstain-bias sweep on the development split -> reports/bias_sweep.md.

  python scripts/bias_sweep.py --run -16=<dir> --run -3=<dir> --run 0=<dir> ...

`abstain_bias` is added to the logit of the token that follows "<" when it spells "abstain" (it changes no weights). Each --run is
bias=<directory holding scores.jsonl> (the lenient reading). The selection rule was fixed before the sweep: the bias with the highest
development end-to-end good rate, ties broken toward the bias closest to 0. The test split is not used here.
"""

import argparse
from pathlib import Path

from ragharness import scorecard

ROOT = Path(__file__).resolve().parents[1]


def fmt(b: float) -> str:
    return '0' if b == 0 else f'{b:+g}'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="bias=directory")
    ap.add_argument("--out", default="reports/bias_sweep.md")
    args = ap.parse_args()
    rows = []
    for item in args.run:
        b, path = item.split("=", 1)
        scores, _ = scorecard.load(Path(path))
        head = [s for s in scores if s["variant"] in ("open", "support_masked")]
        n = len(head)
        sup = sum(s["outcome"] == "supported_correct" for s in head)
        cab = sum(s["outcome"] == "correct_abstain" for s in head)
        fab = sum(s["outcome"] == "false_abstain" for s in head)
        inv = sum(s["outcome"] == "invalid" for s in head)
        ans_ok = [s for s in head if s["answerable"]]
        unans = [s for s in head if not s["answerable"]]
        bad_unans = sum(s["final"] == "answer" for s in unans)
        rows.append(dict(b=float(b), n=n, good=sup + cab, sup=sup, cab=cab, fab=fab, inv=inv,
                         n_ans=len(ans_ok), n_un=len(unans), bad_un=bad_unans,
                         fab_ans=sum(s["outcome"] == "false_abstain" for s in ans_ok),
                         sup_ans=sum(s["outcome"] == "supported_correct" for s in ans_ok)))
    rows.sort(key=lambda r: r["b"])
    best = max(r["good"] for r in rows)
    chosen = min((r for r in rows if r["good"] == best), key=lambda r: abs(r["b"]))
    lines = [
        f"# Abstain-bias sweep (development split, lenient reading, headline episodes: {rows[0]['n']})",
        "",
        "The bias is added at decoding time to the logit of `abstain` after `<`; no weights change. Negative values make abstaining less likely.",
        "Rule fixed before the sweep: highest development end-to-end good rate, ties toward the bias closest to 0.",
        f"Selected: **{fmt(chosen['b'])}** (good {chosen['good']}/{chosen['n']}). The test split was not run at other biases, so the test result for the trained adapter (bias 0) stays the only test number.",
        "",
        "| bias | end-to-end good | supported answer (answerable) | false abstain (answerable) | correct abstain (unanswerable) | answered anyway (unanswerable) | no valid final turn |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        mark = " (selected)" if r is chosen else ""
        lines.append(
            f"| {fmt(r['b'])}{mark} | {r['good'] / r['n']:.3f} ({r['good']}/{r['n']}) | {r['sup_ans'] / r['n_ans']:.3f} ({r['sup_ans']}/{r['n_ans']}) "
            f"| {r['fab_ans'] / r['n_ans']:.3f} ({r['fab_ans']}/{r['n_ans']}) | {r['cab'] / r['n_un']:.3f} ({r['cab']}/{r['n_un']}) "
            f"| {r['bad_un'] / r['n_un']:.3f} ({r['bad_un']}/{r['n_un']}) | {r['inv'] / r['n']:.3f} ({r['inv']}/{r['n']}) |"
        )
    out = ROOT / args.out
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
