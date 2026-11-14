"""Figures for the README and the Article, drawn from the saved report files.

  python scripts/make_readme_figures.py        # -> docs/figures/{headline,tradeoff,pipeline,toolchain,outcomes,trace_examples,rounds}.svg

Every number comes from reports/reference_policies.json (local systems and the refuse-everything line, 934 episodes of the round-5
test, scorer v3) or reports/fair_rerun.json (the hosted models in the format-aware rerun, 925 shared episodes). The trace
cards read the packaged-system check rows when they are on disk (runs/ is git-ignored) and otherwise reuse the committed
docs/examples/trace_examples.json, which this script writes the first time it can.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
FONT = "Segoe UI, Helvetica, Arial, sans-serif"
INK, MUTED, RULE = "#1f2328", "#57606a", "#d0d7de"

BASE, ADAPTER, MAIN, CAND, GPT, DEEPSEEK = "#9e9e9e", "#81c784", "#1565c0", "#ef6c00", "#8e6bbf", "#00897b"
GREEN, RED, GREY = "#2e7d32", "#b42318", "#6e7781"


def load(name: str) -> dict:
    return json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))


def systems() -> list[dict]:
    """The six systems of the headline chart, each with the four numbers it shows."""
    ref = load("reference_policies.json")["sealed_r5test"]
    fair = load("fair_rerun.json")["systems"]

    def local(key: str) -> dict:
        s = ref[key]
        return {"macro": s["macro"], "correct_cited": s["supported_correct"], "hidden_answers": s["unsupported_rate"], "correct_among": s["correct_among_answered"],
                "hidden": f"{s['unsupported']}/{s['hidden_n']}"}

    def hosted(key: str) -> dict:
        s = fair[key]
        return {"macro": s["macro"], "correct_cited": s["supported_correct"], "hidden_answers": s["unsupported_rate"], "correct_among": s["correct_among_answered"],
                "hidden": f"{s['unsupported']}/{s['hidden_n']}"}

    return [
        {"key": "base", "label": "Tiny Aya base", "colour": BASE, **local("base")},
        {"key": "main_raw", "label": "main adapter alone", "colour": ADAPTER, **local("main")},
        {"key": "main", "label": "main + both repairs (submitted)", "colour": MAIN, **local("main+FB+V")},
        {"key": "candF1", "label": "candF1 + both repairs (alternative)", "colour": CAND, **local("candF1+FB+V")},
        {"key": "gpt", "label": "GPT-6 Luna*", "colour": GPT, **hosted("gpt_fair+FB")},
        {"key": "deepseek", "label": "DeepSeek V4 Flash*", "colour": DEEPSEEK, **hosted("deepseek_fair+FB")},
    ]


def refuse_everything() -> float:
    return load("reference_policies.json")["sealed_r5test"]["always_refuse"]["macro"]


def svg_open(w: int, h: int) -> list[str]:
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="{FONT}">',
            f'<rect width="{w}" height="{h}" fill="#ffffff"/>']


def text(x, y, s, size=12, fill=INK, weight=400, anchor="start", extra="") -> str:
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" {extra}>{escape(str(s))}</text>'


def write(name: str, parts: list[str]) -> None:
    parts.append("</svg>")
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8", newline="\n")
    print("wrote", path.relative_to(ROOT))


# ---------------------------------------------------------------- headline


def headline() -> None:
    sy = systems()
    panels = [
        ("Macro success", "higher is better", "macro", True),
        ("Correct and cited answers", "share of answerable episodes, higher is better", "correct_cited", False),
        ("Answers although the evidence was hidden", "share of hidden-evidence episodes, lower is better", "hidden_answers", False),
        ("Correct among the answers shown", "higher is better", "correct_among", False),
    ]
    w, pw, ph, bar, left, top = 820, 380, 168, 17, 20, 78
    h = top + 2 * ph + 100
    out = svg_open(w, h)
    out.append(text(20, 26, "Tiny Aya with search, citations and two repairs, against the base model and two hosted models", 16, weight=600))
    out.append(text(20, 46, "Sealed round-5 test, scorer v3. Local rows: 934 episodes. Hosted rows (*): 925 shared episodes, told the search limit and answer format.", 11.5, MUTED))
    chart_w = 290
    for i, (title, sub, key, refline) in enumerate(panels):
        x0, y0 = left + (i % 2) * (pw + 20), top + (i // 2) * ph
        out.append(text(x0, y0 + 14, title, 13, weight=600))
        out.append(text(x0, y0 + 29, sub, 11, MUTED))
        y = y0 + 40
        for s in sy:
            value = s[key]
            width = chart_w * value
            out.append(f'<rect x="{x0}" y="{y}" width="{width:.1f}" height="{bar - 3}" fill="{s["colour"]}"/>')
            label = f"{value * 100:.1f}%" + (f"  ({s['hidden']})" if key == "hidden_answers" else "")
            out.append(text(x0 + width + 5, y + bar - 5, label, 10.5))
            y += bar
        if refline:
            x = x0 + chart_w * refuse_everything()
            out.append(f'<line x1="{x:.1f}" y1="{y0 + 36}" x2="{x:.1f}" y2="{y + 2}" stroke="{INK}" stroke-width="1.2" stroke-dasharray="4 3"/>')
            out.append(text(x + 4, y + 14, f"refusing every question scores {refuse_everything() * 100:.1f}%", 10.5, MUTED))
    out.append(f'<line x1="20" y1="{top + 2 * ph + 4}" x2="{w - 20}" y2="{top + 2 * ph + 4}" stroke="{RULE}"/>')
    x, y = 20, top + 2 * ph + 24
    for s in sy:
        label = s["label"]
        width = 18 + 6.4 * len(label) + 20
        if x + width > w - 10:
            x, y = 20, y + 20
        out.append(f'<rect x="{x}" y="{y - 10}" width="12" height="12" fill="{s["colour"]}"/>')
        out.append(text(x + 17, y, label, 11.5))
        x += width
    out.append(text(20, y + 24, "Bars in every panel follow the legend order. Hidden-evidence answers are support flags from annotations, not human-checked hallucinations.", 10.5, MUTED))
    write("headline.svg", out)


# ---------------------------------------------------------------- trade-off


def tradeoff() -> None:
    ref = load("reference_policies.json")["sealed_r5test"]
    fair = load("fair_rerun.json")["systems"]
    points = [  # (label, x = answers on hidden evidence, y = correct and cited, colour, dx, dy, anchor)
        ("Tiny Aya base", ref["base"]["unsupported_rate"], ref["base"]["supported_correct"], BASE, -10, -12, "end"),
        ("main adapter alone", ref["main"]["unsupported_rate"], ref["main"]["supported_correct"], ADAPTER, 11, 4, "start"),
        ("main + both repairs", ref["main+FB+V"]["unsupported_rate"], ref["main+FB+V"]["supported_correct"], MAIN, 11, 18, "start"),
        ("candF1 + both repairs", ref["candF1+FB+V"]["unsupported_rate"], ref["candF1+FB+V"]["supported_correct"], CAND, 11, -2, "start"),
        ("GPT-6 Luna*", fair["gpt_fair+FB"]["unsupported_rate"], fair["gpt_fair+FB"]["supported_correct"], GPT, 11, 4, "start"),
        ("DeepSeek V4 Flash*", fair["deepseek_fair+FB"]["unsupported_rate"], fair["deepseek_fair+FB"]["supported_correct"], DEEPSEEK, 11, 4, "start"),
        ("refuse every question", 0.0, 0.0, INK, 11, -4, "start"),
    ]
    w, h, left, top, pw, ph = 800, 520, 80, 64, 660, 380
    xmax, ymax = 0.8, 0.9
    px = lambda v: left + pw * v / xmax  # noqa: E731
    py = lambda v: top + ph - ph * v / ymax  # noqa: E731
    out = svg_open(w, h)
    out.append(text(20, 26, "More correct answers, or fewer answers when the evidence is missing", 16, weight=600))
    out.append(text(20, 46, "Up and to the left is better. Same sealed round-5 test as the headline chart; hosted models (*) on the 925-episode format-aware rerun.", 11.5, MUTED))
    out.append(f'<rect x="{px(0):.1f}" y="{py(ymax):.1f}" width="{px(0.2) - px(0):.1f}" height="{py(0.4) - py(ymax):.1f}" fill="#e8f5e9"/>')
    out.append(text(px(0) + 8, py(ymax) + 16, "few unsupported answers,", 11, GREEN, 600))
    out.append(text(px(0) + 8, py(ymax) + 30, "many correct ones", 11, GREEN, 600))
    for v in (0, 0.2, 0.4, 0.6, 0.8):
        out.append(f'<line x1="{px(v):.1f}" y1="{top}" x2="{px(v):.1f}" y2="{top + ph}" stroke="{RULE}" stroke-width="0.8"/>')
        out.append(text(px(v), top + ph + 18, f"{v * 100:.0f}%", 11, MUTED, anchor="middle"))
    for v in (0, 0.2, 0.4, 0.6, 0.8):
        out.append(f'<line x1="{left}" y1="{py(v):.1f}" x2="{left + pw}" y2="{py(v):.1f}" stroke="{RULE}" stroke-width="0.8"/>')
        out.append(text(left - 8, py(v) + 4, f"{v * 100:.0f}%", 11, MUTED, anchor="end"))
    out.append(text(left + pw / 2, top + ph + 42, "Answers given although the evidence was hidden (share of hidden-evidence episodes; lower is better)", 12, anchor="middle"))
    out.append(f'<text transform="translate(24 {top + ph / 2}) rotate(-90)" font-size="12" fill="{INK}" text-anchor="middle">Correct and cited (share of answerable episodes)</text>')
    # arrow: what the repairs do to the adapter
    a, b = points[1], points[2]
    out.append(f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{MAIN}"/></marker></defs>')
    out.append(f'<line x1="{px(a[1]) - 8:.1f}" y1="{py(a[2]):.1f}" x2="{px(b[1]) + 9:.1f}" y2="{py(b[2]) - 1:.1f}" stroke="{MAIN}" stroke-width="1.6" marker-end="url(#arrow)"/>')
    out.append(text(px(0.205), py(0.33), "blue arrow: what the two repairs do to the adapter", 11, MAIN, 600))
    for label, x, y, colour, dx, dy, anchor in points:
        out.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="7" fill="{colour}" stroke="#ffffff" stroke-width="1.5"/>')
        out.append(text(px(x) + dx, py(y) + dy, f"{label} ({x * 100:.0f}%, {y * 100:.0f}%)", 11.5, INK, 600 if "repairs" in label else 400, anchor))
    out.append(text(20, h - 14, "Hosted models answer more questions correctly and also answer more often when the evidence is missing. Both axes are annotation-based.", 11, MUTED))
    write("tradeoff.svg", out)


# ---------------------------------------------------------------- pipeline


def wrap(s: str, n: int) -> list[str]:
    return textwrap.wrap(s, n, break_long_words=False) or [""]


def box(out, x, y, w, h, title, body, fill, stroke, n=24, size=11) -> None:
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
    out.append(text(x + w / 2, y + 22, title, 13, weight=700, anchor="middle"))
    yy = y + 42
    for line in [l for part in body for l in wrap(part, n)]:
        out.append(text(x + w / 2, yy, line, size, MUTED, anchor="middle"))
        yy += 15


def arrow(out, x1, y1, x2, y2, colour=GREY) -> None:
    out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colour}" stroke-width="1.6" marker-end="url(#head)"/>')


def pipeline() -> None:
    w, h = 820, 424
    out = svg_open(w, h)
    out.append(f'<defs><marker id="head" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="{GREY}"/></marker></defs>')
    out.append(text(20, 26, "How one question is answered", 16, weight=600))
    out.append(text(20, 46, "Trained with QLoRA through AgentTune (steps 2 and 3). The two repairs (steps 4 and 5) need no training.", 11.5, MUTED))
    bw, bh, y, gap = 140, 214, 78, 16
    xs = [20 + i * (bw + gap) for i in range(5)]
    box(out, xs[0], y, bw, bh, "1  Question", ["A user question and a set of documents, indexed for search."], "#f6f8fa", RULE, 19)
    box(out, xs[1], y, bw, bh, "2  Search loop", ["Tiny Aya calls search_corpus.", "Up to 3 searches, 3 passages each, 800 characters per passage."], "#e8f1fb", MAIN, 19)
    box(out, xs[2], y, bw, bh, "3  Draft ending", ["<answer> with chunk citations,", "or <abstain> saying what is missing."], "#e8f1fb", MAIN, 19)
    box(out, xs[3], y, bw, bh, "4  Fail-closed", ["A rollout with no valid ending (repeated query, search over budget, bad format) becomes a refusal."], "#fff4e5", CAND, 19)
    box(out, xs[4], y, bw, bh, "5  Self-check", ["The same model reads the question, the cited passages and the answer.", "The answer is shown only if P(SUPPORTED) is at least 0.5, the cited passage was shown, and the answer text appears in it."], "#fff4e5", CAND, 19)
    for i in range(4):
        arrow(out, xs[i] + bw + 1, y + bh / 2, xs[i + 1] - 3, y + bh / 2)
    yo = y + bh + 48
    rx, rw = xs[2] - 6, 2 * bw + gap + 12
    out.append(f'<rect x="{rx}" y="{yo}" width="{rw}" height="46" rx="23" fill="#f6f8fa" stroke="{GREY}" stroke-width="1.4"/>')
    out.append(text(rx + rw / 2, yo + 20, "Refusal: the documents don't say", 12.5, INK, 700, "middle"))
    out.append(text(rx + rw / 2, yo + 36, "from the model itself, fail-closed, or the self-check", 11, MUTED, anchor="middle"))
    ax = xs[4] + 8
    out.append(f'<rect x="{ax}" y="{yo}" width="{bw - 8}" height="46" rx="23" fill="#e8f5e9" stroke="{GREEN}" stroke-width="1.4"/>')
    out.append(text(ax + (bw - 8) / 2, yo + 20, "Answer with", 12.5, GREEN, 700, "middle"))
    out.append(text(ax + (bw - 8) / 2, yo + 36, "citations", 12.5, GREEN, 700, "middle"))
    arrow(out, xs[2] + bw / 2, y + bh + 1, xs[2] + bw / 2, yo - 3)
    arrow(out, xs[3] + bw / 2, y + bh + 1, xs[3] + bw / 2, yo - 3)
    arrow(out, xs[4] + 28, y + bh + 1, rx + rw - 14, yo - 3)
    arrow(out, xs[4] + bw / 2 + 24, y + bh + 1, xs[4] + bw / 2 + 24, yo - 3, GREEN)
    out.append(text(20, h - 16, "Steps 2 and 3 are one trained LoRA adapter on Tiny Aya Global; steps 4 and 5 are code.", 11, MUTED))
    write("pipeline.svg", out)


# ---------------------------------------------------------------- real traces


CARD_KINDS = [
    ("answered", "Answered with citations", GREEN, lambda r, q: r["decision"] == "answer"),
    ("refused_hidden", "Refused: evidence hidden from search", GREY, lambda r, q: r["decision"] == "abstain" and r["model_final"] == "abstain" and r["variant"] == "support_masked"),
    ("self_check", "Answer withheld by the self-check", CAND, lambda r, q: r["repaired_by"] == "self_check"),
    ("fail_closed", "Refused by the fail-closed rule", CAND, lambda r, q: r["repaired_by"] == "fail_closed"),
]


def collect_trace_examples() -> list[dict]:
    cache = ROOT / "docs" / "examples" / "trace_examples.json"
    source = ROOT / "runs" / "account2_round5" / "ats-check-system-r5" / "out" / "system_check" / "system_r5t_hotpot.jsonl"
    if not source.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in source.read_text(encoding="utf-8").splitlines() if l.strip()]
    qs = {q["qid"]: q for q in (json.loads(l) for l in (ROOT / "data/bench/r5t_hotpot/questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    titles = {c["chunk_id"]: c["title"] for c in json.loads((ROOT / "data/bench/r5t_hotpot/corpus.json").read_text(encoding="utf-8"))["chunks"]}
    examples = []
    for kind, label, colour, pred in CARD_KINDS:
        r = next(x for x in rows if pred(x, qs[x["qid"]]))
        q = qs[r["qid"]]
        examples.append({"kind": kind, "label": label, "question": q["question"], "gold": q["answer"], "variant": r["variant"], "queries": r["queries"], "answer": r["answer"],
                         "cited": [titles.get(c, c) for c in r["citations"]], "p_supported": r["p_supported"], "reason": r["reason"], "model_final": r["model_final"]})
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(examples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return examples


def trace_examples() -> None:
    ex = collect_trace_examples()
    colours = {k: c for k, _, c, _ in CARD_KINDS}
    cw, ch, gx, gy, top = 390, 216, 20, 16, 70
    w, h = 820, top + 2 * ch + gy + 56
    out = svg_open(w, h)
    out.append(text(20, 26, "Four recorded episodes of the packaged system", 16, weight=600))
    out.append(text(20, 46, "HotpotQA part of the sealed test, selection rule: the first episode of each kind in file order.", 11.5, MUTED))
    for i, e in enumerate(ex):
        x, y = 20 + (i % 2) * (cw + gx), top + (i // 2) * (ch + gy)
        colour = colours[e["kind"]]
        out.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="8" fill="#ffffff" stroke="{colour}" stroke-width="1.6"/>')
        out.append(f'<rect x="{x}" y="{y}" width="{cw}" height="26" rx="8" fill="{colour}"/><rect x="{x}" y="{y + 14}" width="{cw}" height="12" fill="{colour}"/>')
        out.append(text(x + 12, y + 18, e["label"], 12.5, "#ffffff", 700))
        yy = y + 46
        out.append(text(x + 12, yy, "Question", 10.5, MUTED, 600))
        for line in wrap(e["question"], 62)[:3]:
            yy += 15
            out.append(text(x + 12, yy, line, 11.5))
        yy += 20
        out.append(text(x + 12, yy, "Searches", 10.5, MUTED, 600))
        for qline in e["queries"][:3]:
            yy += 15
            out.append(text(x + 12, yy, "• " + (qline if len(qline) <= 64 else qline[:61] + "..."), 11))
        yy += 20
        if e["kind"] == "answered":
            out.append(text(x + 12, yy, f"Answer: {e['answer']}   (P(SUPPORTED) {e['p_supported']:.2f})", 11.5, GREEN, 600))
            yy += 15
            out.append(text(x + 12, yy, "Cited: " + "; ".join(e["cited"])[:60], 11, MUTED))
        elif e["kind"] == "self_check":
            out.append(text(x + 12, yy, f"Answer withheld: P(SUPPORTED) {e['p_supported']:.2f} is below 0.5", 11.5, CAND, 600))
        elif e["kind"] == "fail_closed":
            out.append(text(x + 12, yy, "No valid ending: " + e["reason"].replace("invalid turn: ", ""), 11.5, CAND, 600))
        else:
            out.append(text(x + 12, yy, "The model said the documents did not hold the answer.", 11.5, INK, 600))
        out.append(text(x + 12, y + ch - 12, "Reference answer (not shown to the model): " + (e["gold"] if len(e["gold"]) <= 40 else e["gold"][:37] + "..."), 10.5, MUTED))
    out.append(text(20, h - 14, "Questions come from HotpotQA (CC BY-SA 4.0). The withheld and fail-closed cases do not say whether the model's own answer would have been right.", 10.5, MUTED))
    write("trace_examples.svg", out)


# ---------------------------------------------------------------- rounds


def rounds() -> None:
    ref = load("reference_policies.json")["sealed_r5test"]
    rows = [
        ("Round 1", "Main adapter, with refusal examples, against a control without them", "Adopted: main", MAIN),
        ("Round 2", "candA: fewer refusal examples in training", "Alternative: fewer refusals, more unsupported answers", CAND),
        ("Round 3", "A SQuAD patch for single-passage and two-part questions", "Rejected: failed its safety guard (6 to 31 unsupported SQuAD answers of 109)", RED),
        ("Round 4", "Matched-evidence curriculum (candE, candF)", "No checkpoint met the release rules; candF1 kept as an alternative", CAND),
        ("Round 5", "Fail-closed rule and self-check on the output", f"Adopted: macro {ref['main']['macro'] * 100:.1f}% to {ref['main+FB+V']['macro'] * 100:.1f}%, no retraining", MAIN),
        ("Round 5", "Preference training (DPO) on 335 pairs", "Rejected: the checkpoints stopped abstaining", RED),
        ("Round 5b", "DPO again, with abstention pairs added", "Rejected: one eligible checkpoint, candF1, missed the safety bound", RED),
        ("Round 6", "Trained answer checkers V2m and V2b", "Rejected: no threshold met the hidden-evidence limit", RED),
    ]
    w, rh, top = 820, 50, 66
    h = top + len(rows) * rh + 24
    out = svg_open(w, h)
    out.append(text(20, 26, "What we tried, and what we kept", 16, weight=600))
    out.append(text(20, 46, "Every rule was written down before the data it applied to. Details in docs/HISTORY.md.", 11.5, MUTED))
    for i, (rnd, what, result, colour) in enumerate(rows):
        y = top + i * rh
        out.append(f'<line x1="20" y1="{y}" x2="{w - 20}" y2="{y}" stroke="{RULE}" stroke-width="0.8"/>')
        out.append(f'<circle cx="34" cy="{y + 26}" r="6" fill="{colour}"/>')
        out.append(text(52, y + 22, rnd, 12.5, weight=700))
        out.append(text(52, y + 38, what, 11.5, MUTED))
        out.append(f'<rect x="470" y="{y + 9}" width="330" height="32" rx="6" fill="{colour}" fill-opacity="0.12" stroke="{colour}" stroke-width="1"/>')
        for j, line in enumerate(wrap(result, 52)[:2]):
            out.append(text(480, y + 24 + j * 13 - (6 if len(wrap(result, 52)) > 1 else 0) + 6, line, 11, INK, 600 if j == 0 else 400))
    write("rounds.svg", out)


def toolchain() -> None:
    """Where the two Lexsi libraries sit in the project, in the order the work runs."""
    w, h = 820, 530
    out = svg_open(w, h)
    out.append(f'<defs><marker id="head" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="{GREY}"/></marker></defs>')
    out.append(text(20, 26, "Where AgentTune and AuditKIT fit", 16, weight=600))
    out.append(text(20, 46, "Blue boxes are AgentTune, green boxes are AuditKIT, grey boxes are code we wrote.", 11.5, MUTED))
    bw, bh, gap, n = 177, 176, 24, 25
    xs = [20 + i * (bw + gap) for i in range(4)]
    y1 = 70
    y2 = y1 + bh + 48
    A_FILL, A_EDGE, K_FILL, K_EDGE = "#e8f1fb", MAIN, "#e8f5e9", GREEN
    box(out, xs[0], y1, bw, bh, "1  Documents", ["Public question sets (HotpotQA, SQuAD, RGB) and our fictional Nimbus manuals, cut into short passages."], "#f6f8fa", RULE, n)
    box(out, xs[1], y1, bw, bh, "2  Search tool", ["SQLiteFTSBackend is a keyword index. SearchCorpusTool is the search_corpus call the model makes; we subclassed it to hide chosen passages."], A_FILL, A_EDGE, n)
    box(out, xs[2], y1, bw, bh, "3  Practice runs", ["A scripted policy solves training questions with that tool. Each run is saved as an EventLog."], A_FILL, A_EDGE, n)
    box(out, xs[3], y1, bw, bh, "4  Training", ["create_distill_trainer turns the EventLogs into training rows. Our QLoRA factory trains the adapter on one T4."], A_FILL, A_EDGE, n)
    box(out, xs[3], y2, bw, bh, "5  Test runs", ["The trained model answers new questions with the same tool. Each run is saved as a TrajectoryStore record."], A_FILL, A_EDGE, n)
    box(out, xs[2], y2, bw, bh, "6  Wrap each run", ["Our adapter turns every run into an ak.Sample: a RAGTrace (what it searched, read and cited) and a RAGCase (the right behaviour)."], "#f6f8fa", RULE, n)
    box(out, xs[1], y2, bw, bh, "7  Score", ["ak.evaluate computes seven scores: evidence-set recall, citation support, abstention, tool-call validity, redundant calls, loop detection, freshness."], K_FILL, K_EDGE, n)
    box(out, xs[0], y2, bw, bh, "8  Decide", ["Scorecards and our own scorer decide what we keep. The rules were written before the sealed test was opened."], "#f6f8fa", RULE, n)
    for i in range(3):
        arrow(out, xs[i] + bw + 1, y1 + bh / 2, xs[i + 1] - 3, y1 + bh / 2)
    arrow(out, xs[3] + bw / 2, y1 + bh + 1, xs[3] + bw / 2, y2 - 3)
    for i in (3, 2, 1):
        arrow(out, xs[i] - 1, y2 + bh / 2, xs[i - 1] + bw + 3, y2 + bh / 2)
    out.append(text(20, h - 30, "AgentTune covers steps 2 to 5 (build, record, train). AuditKIT covers step 7 (score). Step 6 is the glue between them.", 11, MUTED))
    out.append(text(20, h - 12, "Preference training (DPO) used our own short loop, because AgentTune's DPO trainer ran out of memory on a T4.", 11, MUTED))
    write("toolchain.svg", out)


def outcomes() -> None:
    """The four things that can happen to a question, with the submitted system's rates on the sealed test."""
    ref = load("reference_policies.json")["sealed_r5test"]["main+FB+V"]
    answerable_other = 1 - ref["supported_correct"] - ref["false_abstain"]
    hidden_refused = ref["hidden_n"] - ref["unsupported"]
    w, h, cw, ch, gap = 820, 436, 300, 120, 12
    out = svg_open(w, h)
    out.append(text(20, 26, "The four things that can happen to a question", 16, weight=600))
    out.append(text(20, 46, "Percentages are for the submitted system on the sealed test (934 episodes). The rows show whether the documents hold the answer.", 11.5, MUTED))
    x0, y0 = 190, 100
    out.append(text(x0 + cw / 2, y0 - 14, "The system answers", 13, weight=700, anchor="middle"))
    out.append(text(x0 + cw + gap + cw / 2, y0 - 14, "The system refuses", 13, weight=700, anchor="middle"))
    out.append(text(20, y0 + ch / 2 - 4, "The documents", 13, weight=700))
    out.append(text(20, y0 + ch / 2 + 12, "hold the answer", 13, weight=700))
    out.append(text(20, y0 + ch + gap + ch / 2 - 4, "The documents do", 13, weight=700))
    out.append(text(20, y0 + ch + gap + ch / 2 + 12, "not hold it", 13, weight=700))
    cells = [
        (0, 0, "#e8f5e9", GREEN, "Correct and cited", f"{ref['supported_correct'] * 100:.1f}% of these questions",
         f"This is the goal. Another {answerable_other * 100:.1f}% are answered, but wrongly or without the right citation."),
        (1, 0, "#fff4e5", CAND, "Over-refusal", f"{ref['false_abstain'] * 100:.1f}% of these questions", "Safe but unhelpful. This is the price of the two repairs."),
        (0, 1, "#fdeceb", RED, "Unsupported answer", f"{ref['unsupported']} of {ref['hidden_n']} ({ref['unsupported_rate'] * 100:.1f}%)",
         "The system answers although the evidence was hidden. The answer may be made up. Lower is better."),
        (1, 1, "#e8f5e9", GREEN, "Correct refusal", f"{hidden_refused} of {ref['hidden_n']} ({hidden_refused / ref['hidden_n'] * 100:.1f}%)",
         "It says the documents do not hold the answer. This counts as success."),
    ]
    for col, row, fill, edge, title, figure, note in cells:
        x, y = x0 + col * (cw + gap), y0 + row * (ch + gap)
        out.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="8" fill="{fill}" stroke="{edge}" stroke-width="1.4"/>')
        out.append(text(x + 14, y + 26, title, 14, edge, 700))
        out.append(text(x + 14, y + 48, figure, 13, INK, 600))
        for j, line in enumerate(wrap(note, 44)):
            out.append(text(x + 14, y + 70 + j * 15, line, 11.5, MUTED))
    out.append(text(20, h - 34, "Macro success rewards the two green cells, averaged over four groups of questions. Refusing every question therefore scores 42.0%.", 11, MUTED))
    out.append(text(20, h - 16, "That is why we also report the share of correct, cited answers and the share of unsupported answers.", 11, MUTED))
    write("outcomes.svg", out)


def main() -> None:
    headline()
    tradeoff()
    pipeline()
    toolchain()
    outcomes()
    trace_examples()
    rounds()


if __name__ == "__main__":
    main()
