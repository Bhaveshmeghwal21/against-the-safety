"""Figure for round 3: "ends well" per set for the reference and the candidate, from reports/round3_<tag>.json.

  python scripts/make_round3_figure.py --tag test        # fresh test set     -> docs/figures/round3_test.svg
  python scripts/make_round3_figure.py --tag informed    # first-round sets    -> docs/figures/round3_informed.svg
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TITLES = {
    "r3t_hotpot": "HotpotQA", "r3t_2wiki": "2WikiMultihopQA", "r3t_musique": "MuSiQue 2-hop", "r3t_squad2": "SQuAD 2.0", "r3t_rgb": "RGB", "r3t_rgb_int": "RGB integration",
    "2wiki": "2WikiMultihopQA", "musique": "MuSiQue 2-hop", "squad2": "SQuAD 2.0", "rgb": "RGB", "rgb_int": "RGB integration",
}
COLOURS = {"base": "#9e9e9e", "main": "#2e7d32", "candA": "#81c784", "candC": "#1565c0", "candD": "#64b5f6", "control": "#f9a825"}
LABEL = {"base": "Tiny Aya base", "main": "main", "candA": "candA", "candC": "candC (patch)", "candD": "candD (careful patch)", "control": "control"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", choices=["test", "informed"], default="test")
    args = ap.parse_args()
    rep = json.loads((ROOT / "reports" / f"round3_{args.tag}.json").read_text(encoding="utf-8"))
    systems = rep["systems"]
    key = "ends well (all scored episodes)"
    comps = [(b, TITLES.get(b, b)) for b in rep["components"]]
    left, top, bar_h, gap, chart_w, width = 170, 62, 15, 14, 520, 800
    rows = len(comps) + 1
    height = top + rows * (len(systems) * bar_h + gap) + 56
    title = "Ends well on a fresh test set (round 3)" if args.tag == "test" else "Ends well on the first-round benchmark sets, before and after round 3"
    sub = ("Fresh questions; nothing was chosen with them. Supported answer when answerable, abstention when the documents cannot answer." if args.tag == "test"
           else "These sets informed the design of round 3, so this is a before and after, not a test.")
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="Segoe UI, Helvetica, Arial, sans-serif">',
           f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
           f'<text x="{left}" y="24" font-size="16" font-weight="600" fill="#1f2328">{title}</text>',
           f'<text x="{left}" y="42" font-size="12" fill="#57606a">{sub}</text>']
    y = top
    entries = [(TITLES.get(b, b), {s: rep["components"][b]["metrics"][s][key]["value"] for s in systems}) for b, _ in comps]
    entries.append(("macro average", rep["macro"]))
    for i, (title_, vals) in enumerate(entries):
        bold = i == len(entries) - 1
        out.append(f'<text x="{left - 10}" y="{y + len(systems) * bar_h / 2 + 4}" font-size="13" font-weight="{700 if bold else 600}" text-anchor="end" fill="#1f2328">{title_}</text>')
        for s in systems:
            v = vals[s]
            w = chart_w * v
            out.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar_h - 2}" fill="{COLOURS[s]}"/>')
            out.append(f'<text x="{left + w + 4:.1f}" y="{y + bar_h - 4}" font-size="11" fill="#1f2328">{v:.2f}</text>')
            y += bar_h
        y += gap
    x = left
    for s in systems:
        out.append(f'<rect x="{x}" y="{y + 8}" width="12" height="12" fill="{COLOURS[s]}"/>')
        out.append(f'<text x="{x + 17}" y="{y + 19}" font-size="12" fill="#1f2328">{LABEL[s]}</text>')
        x += 17 + 7 * len(LABEL[s]) + 22
    out.append("</svg>")
    p = ROOT / "docs" / "figures" / f"round3_{args.tag}.svg"
    p.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    print("wrote", p)


if __name__ == "__main__":
    main()
