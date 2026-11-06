"""Figure for round 3: "ends well" per set and the pooled unsupported-answer rate, from reports/round3_<tag>.json.

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
LABEL = {"base": "Tiny Aya base", "main": "main", "candA": "candA (round 2)", "candC": "candC (patch)", "candD": "candD (round 3 patch)", "control": "control"}
ENDS = "ends well (all scored episodes)"
UNSUP = "answers when the documents cannot support it (lower is better)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", choices=["test", "informed"], default="test")
    args = ap.parse_args()
    rep = json.loads((ROOT / "reports" / f"round3_{args.tag}.json").read_text(encoding="utf-8"))
    systems = rep["systems"]
    comps = list(rep["components"])
    left, top, bar_h, gap, chart_w, width = 170, 78, 15, 14, 520, 800
    height = top + (len(comps) + 2) * (len(systems) * bar_h + gap) + 80
    title = "Round 3 on a fresh test set" if args.tag == "test" else "Round 3 on the first-round benchmark sets"
    sub1 = ("Fresh questions; nothing was chosen with them." if args.tag == "test" else "These sets informed the design of round 3: a before and after, not a test.")
    sub2 = "Ends well = a supported answer when the documents hold it, an abstention when they do not."
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="Segoe UI, Helvetica, Arial, sans-serif">',
           f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
           f'<text x="{left}" y="24" font-size="16" font-weight="600" fill="#1f2328">{title}</text>',
           f'<text x="{left}" y="42" font-size="12" fill="#57606a">{sub1}</text>',
           f'<text x="{left}" y="58" font-size="12" fill="#57606a">{sub2}</text>']
    entries = [(TITLES.get(b, b), {s: rep["components"][b]["metrics"][s][ENDS]["value"] for s in systems}, False) for b in comps]
    entries.append(("macro average", rep["macro"], True))
    pooled = {}
    for s in systems:
        k = sum(rep["components"][b]["metrics"][s][UNSUP]["k"] for b in comps)
        n = sum(rep["components"][b]["metrics"][s][UNSUP]["n"] for b in comps)
        pooled[s] = k / n
    y = top
    for title_, vals, bold in entries:
        out.append(f'<text x="{left - 10}" y="{y + len(systems) * bar_h / 2 + 4}" font-size="13" font-weight="{700 if bold else 600}" text-anchor="end" fill="#1f2328">{title_}</text>')
        for s in systems:
            w = chart_w * vals[s]
            out.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar_h - 2}" fill="{COLOURS[s]}"/>')
            out.append(f'<text x="{left + w + 4:.1f}" y="{y + bar_h - 4}" font-size="11" fill="#1f2328">{vals[s]:.2f}</text>')
            y += bar_h
        y += gap
    out.append(f'<line x1="20" y1="{y}" x2="{width - 20}" y2="{y}" stroke="#d0d7de"/>')
    y += 14
    out.append(f'<text x="{left}" y="{y}" font-size="12" fill="#57606a">Safety: answers given when no passage in reach holds the answer, all sets pooled (lower is better)</text>')
    y += 8
    out.append(f'<text x="{left - 10}" y="{y + len(systems) * bar_h / 2 + 4}" font-size="13" font-weight="700" text-anchor="end" fill="#b42318">unsupported answers</text>')
    for s in systems:
        w = chart_w * pooled[s]
        out.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar_h - 2}" fill="{COLOURS[s]}"/>')
        out.append(f'<text x="{left + w + 4:.1f}" y="{y + bar_h - 4}" font-size="11" fill="#1f2328">{pooled[s]:.2f}</text>')
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
