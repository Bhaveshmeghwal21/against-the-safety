"""Figure for round 5: the sealed final test, from reports/round5b_final.json (written by scripts/report_round5.py; it holds the round-5 systems and the round-5b candidate).

  python scripts/make_round5_figure.py        # -> docs/figures/round5_final.svg

Top block: macro success and its four components (higher is better). Bottom block: what the repairs cost and gain
(answers although the evidence was hidden, supported correct answers, refusals of answerable episodes).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ["base", "main", "main+FB", "main+FB+V", "candA+FB+V", "candF1+FB+V"]
COLOURS = {"base": "#9e9e9e", "main": "#2e7d32", "main+FB": "#26a69a", "main+FB+V": "#1565c0", "candA+FB+V": "#9575cd", "candF1+FB+V": "#ef6c00"}
LABEL = {"base": "Tiny Aya base", "main": "main", "main+FB": "main + fail-closed",
         "main+FB+V": "main + fail-closed + self-check (submitted)", "candA+FB+V": "candA + both repairs (reference)", "candF1+FB+V": "candF1 + both repairs (alternative, opened once in round 5b)"}
COMPONENTS = [("single", "SQuAD single"), ("compound", "SQuAD two-part"), ("hotpot", "HotpotQA"), ("rgb", "RGB")]


def main() -> None:
    rep = json.loads((ROOT / "reports" / "round5b_final.json").read_text(encoding="utf-8"))
    table = rep["systems"]
    missing = [s for s in SYSTEMS if s not in table]
    assert not missing, f"not in reports/round5b_final.json: {missing}"
    left, top, bar_h, gap, chart_w, width = 200, 84, 14, 12, 500, 800
    top_rows = [(title, {s: table[s]["components"][key] for s in SYSTEMS}, False) for key, title in COMPONENTS]
    top_rows.append(("macro success", {s: table[s]["macro"] for s in SYSTEMS}, True))
    bottom_rows = [
        ("answers although the evidence was hidden (lower is better)", {s: table[s]["unsupported"] / table[s]["hidden_n"] for s in SYSTEMS}, "#b42318"),
        ("supported correct answers, answerable episodes (higher is better)", {s: table[s]["supported_correct"] for s in SYSTEMS}, "#1f2328"),
        ("refusals of answerable episodes (lower is better)", {s: table[s]["false_abstain"] for s in SYSTEMS}, "#1f2328"),
    ]
    block = len(SYSTEMS) * bar_h + gap
    height = top + len(top_rows) * block + 40 + len(bottom_rows) * (block + 14) + 120
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="Segoe UI, Helvetica, Arial, sans-serif">',
           f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
           f'<text x="{left}" y="24" font-size="16" font-weight="600" fill="#1f2328">Round 5 on a sealed test of 938 fresh episodes</text>',
           f'<text x="{left}" y="42" font-size="12" fill="#57606a">SQuAD, HotpotQA and RGB questions no earlier round used; scorer version 2.</text>',
           f'<text x="{left}" y="58" font-size="12" fill="#57606a">Success = a correct, fully cited answer, or an abstention when the evidence is hidden.</text>',
           f'<text x="{left}" y="74" font-size="12" fill="#57606a">Macro success is the mean of the four component rates.</text>']

    def bars(y: float, vals: dict) -> float:
        for s in SYSTEMS:
            w = chart_w * vals[s]
            out.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar_h - 2}" fill="{COLOURS[s]}"/>')
            out.append(f'<text x="{left + w + 4:.1f}" y="{y + bar_h - 4}" font-size="11" fill="#1f2328">{vals[s]:.2f}</text>')
            y += bar_h
        return y

    y = top
    for title, vals, bold in top_rows:
        out.append(f'<text x="{left - 10}" y="{y + len(SYSTEMS) * bar_h / 2 + 4}" font-size="13" font-weight="{700 if bold else 600}" text-anchor="end" fill="#1f2328">{title}</text>')
        y = bars(y, vals) + gap
    out.append(f'<line x1="20" y1="{y}" x2="{width - 20}" y2="{y}" stroke="#d0d7de"/>')
    y += 22
    out.append(f'<text x="{left}" y="{y - 6}" font-size="12" fill="#57606a">What the repairs gain and what they cost</text>')
    for title, vals, colour in bottom_rows:
        y += 4
        out.append(f'<text x="{left}" y="{y + 8}" font-size="12" font-weight="600" fill="{colour}">{title}</text>')
        y += 14
        y = bars(y, vals) + gap
    x, row_y = left, y + 4
    for s in SYSTEMS:
        text_w = 6.4 * len(LABEL[s])
        if x + 17 + text_w > width - 10:
            x, row_y = left, row_y + 20
        out.append(f'<rect x="{x}" y="{row_y}" width="12" height="12" fill="{COLOURS[s]}"/>')
        out.append(f'<text x="{x + 17}" y="{row_y + 11}" font-size="12" fill="#1f2328">{LABEL[s]}</text>')
        x += 17 + text_w + 18
    out.append(f'<text x="{left}" y="{row_y + 38}" font-size="11" fill="#57606a">The repairs cut unsupported answers by trading them for refusals; supported correct answers do not rise (0.290 to 0.275).</text>')
    out.append(f'<text x="{left}" y="{row_y + 54}" font-size="11" fill="#57606a">Opened for base, main and candA, then once for candF1 (round 5b). The submitted system is unchanged.</text>')
    out.append("</svg>")
    p = ROOT / "docs" / "figures" / "round5_final.svg"
    p.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    print("wrote", p, "height", height)


if __name__ == "__main__":
    main()
