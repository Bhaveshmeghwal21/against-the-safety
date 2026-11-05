"""Stacked-bar SVG of how every headline episode ends, per system (no plotting library needed).

  python scripts/make_figure.py --run base=<dir> --run main=<dir> --run control=<dir> --out docs/figures/outcomes_test.svg

Each --run is name=<directory holding scores.jsonl>. The five outcomes partition the headline episodes (open and support-masked)
exactly as in scripts/tradeoff.py.
"""

import argparse
from pathlib import Path

from ragharness import scorecard

ROOT = Path(__file__).resolve().parents[1]

OUTCOMES = [  # key, legend label, colour
    ("supported", "supported answer", "#2e7d32"),
    ("c_abs", "correct abstain", "#81c784"),
    ("f_abs", "false abstain", "#f9a825"),
    ("bad", "wrong or unsupported answer", "#c62828"),
    ("invalid", "no valid final turn", "#9e9e9e"),
]


def counts(path: Path) -> dict:
    scores, _ = scorecard.load(path)
    head = [s for s in scores if s["variant"] in ("open", "support_masked")]
    c = {
        "supported": sum(s["outcome"] == "supported_correct" for s in head),
        "bad": sum(s["final"] == "answer" and s["outcome"] != "supported_correct" for s in head),
        "c_abs": sum(s["outcome"] == "correct_abstain" for s in head),
        "f_abs": sum(s["outcome"] == "false_abstain" for s in head),
        "invalid": sum(s["outcome"] == "invalid" for s in head),
    }
    assert sum(c.values()) == len(head), "outcomes must partition the episodes"
    c["n"] = len(head)
    return c


def svg(rows: list[tuple[str, dict]], title: str) -> str:
    left, bar_w, bar_h, gap, top = 140, 600, 34, 18, 58
    width = left + bar_w + 30
    legend_y = top + len(rows) * (bar_h + gap) + 8
    height = legend_y + 40
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'font-family="Segoe UI, Helvetica, Arial, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="26" font-size="16" font-weight="600" fill="#1f2328">{title}</text>',
        f'<text x="{left}" y="45" font-size="12" fill="#57606a">Each bar is the same {rows[0][1]["n"]} test episodes. Green is a good outcome.</text>',
    ]
    for i, (name, c) in enumerate(rows):
        y = top + i * (bar_h + gap)
        out.append(f'<text x="{left - 12}" y="{y + bar_h / 2 + 5}" font-size="14" font-weight="600" text-anchor="end" fill="#1f2328">{name}</text>')
        x = float(left)
        for key, _, colour in OUTCOMES:
            w = bar_w * c[key] / c["n"]
            if w <= 0:
                continue
            out.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{bar_h}" fill="{colour}"/>')
            if w >= 26:
                ink = "#1f2328" if key in ("c_abs", "f_abs") else "#ffffff"
                out.append(f'<text x="{x + w / 2:.1f}" y="{y + bar_h / 2 + 5}" font-size="13" text-anchor="middle" fill="{ink}">{c[key]}</text>')
            x += w
    x, ly = float(left), legend_y
    for key, label, colour in OUTCOMES:
        item_w = 17 + 7.0 * len(label) + 18
        if x + item_w > left + bar_w + 20:
            x, ly = float(left), ly + 22
        out.append(f'<rect x="{x:.0f}" y="{ly}" width="12" height="12" fill="{colour}"/>')
        out.append(f'<text x="{x + 17:.0f}" y="{ly + 11}" font-size="12" fill="#1f2328">{label}</text>')
        x += item_w
    out.append("</svg>")
    out[0] = out[0].replace(f'height="{height}" viewBox="0 0 {width} {height}"', f'height="{ly + 26}" viewBox="0 0 {width} {ly + 26}"')
    out[1] = f'<rect width="{width}" height="{ly + 26}" fill="#ffffff"/>'
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="name=directory")
    ap.add_argument("--title", default="How each test episode ends")
    ap.add_argument("--out", default="docs/figures/outcomes_test.svg")
    args = ap.parse_args()
    rows = []
    for item in args.run:
        name, path = item.split("=", 1)
        rows.append((name, counts(Path(path))))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg(rows, args.title), encoding="utf-8", newline="\n")
    for name, c in rows:
        print(name, {k: c[k] for k, _, _ in OUTCOMES}, "n", c["n"])


if __name__ == "__main__":
    main()
