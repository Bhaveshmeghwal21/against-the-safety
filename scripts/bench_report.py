"""External benchmark report (plan section 22) -> reports/benchmarks_<parse>.md / .json and docs/figures/benchmarks.svg.

  python scripts/bench_report.py --eval runs/kaggle/ats-eval-bench-a/out/eval --eval runs/kaggle/ats-eval-bench-b/out/eval

Reads <system>_bench_<name>/<parse>/scores.jsonl and closed_bench_<name>/summary.json from every --eval directory.
"""

import argparse
import json
from pathlib import Path

from ragharness import scorecard

ROOT = Path(__file__).resolve().parents[1]
BENCHES = [("2wiki", "2WikiMultihopQA"), ("musique", "MuSiQue (2-hop)"), ("squad2", "SQuAD 2.0"), ("rgb", "RGB"), ("rgb_int", "RGB integration")]
SYSTEMS = ["base", "main", "candA", "control"]
LABEL = {"base": "Tiny Aya base", "main": "main", "candA": "candA", "control": "control"}
AUDIT = ["evidence_set_recall", "citation_support", "abstention", "tool_call_validity", "redundant_tool_calls", "agent_loop_detection"]
AUDIT_LABEL = {"redundant_tool_calls": "redundant_tool_calls (lower is better)", "agent_loop_detection": "agent_loop_detection (1.0 = loop-free)"}


def rate(rows, pred):
    n = len(rows)
    k = sum(1 for r in rows if pred(r))
    return {"k": k, "n": n, "value": round(k / n, 3) if n else None}


def cell(r) -> str:
    return "n/a" if not r or r["n"] == 0 else f"{r['value']:.3f} ({r['k']}/{r['n']})"


def metrics(scores: list[dict]) -> dict:
    v = lambda *names: [s for s in scores if s["variant"] in names]
    una = [s for s in scores if not s["answerable"]]
    acc = lambda rows: rate(rows, lambda s: s["final"] == "answer" and bool(s.get("answer_ok")))
    return {
        "ends well (all scored episodes)": rate(scores, lambda s: s["good"]),
        "supported correct, open search (answerable)": rate(v("open"), lambda s: s["outcome"] == "supported_correct"),
        "false abstain, open search (lower is better)": rate(v("open"), lambda s: s["outcome"] == "false_abstain"),
        "false abstain with the full evidence already shown, open search (post hoc; lower is better)": rate(v("open"), lambda s: s["outcome"] == "false_abstain" and bool(s["evidence_complete"])),
        "answer correct, citation not required, open search (post hoc)": acc(v("open")),
        "correct abstain, support masked": rate(v("support_masked"), lambda s: s["outcome"] == "correct_abstain"),
        "correct abstain, natural unanswerable (SQuAD 2.0)": rate(v("natural_unanswerable"), lambda s: s["outcome"] == "correct_abstain"),
        "answers when the documents cannot support it (lower is better)": rate(una, lambda s: s["final"] == "answer"),
        "RGB accuracy, noise 0 (answer correct, citation not required)": acc(v("rgb_noise_0")),
        "RGB accuracy, noise 0.8": acc(v("rgb_noise_0.8")),
        "RGB rejection rate, noise 1 (abstains)": rate(v("rgb_noise_1"), lambda s: s["final"] == "abstain"),
        "RGB integration accuracy, noise 0": acc(v("rgb_int_noise_0")),
        "RGB integration accuracy, noise 0.4": acc(v("rgb_int_noise_0.4")),
        "invalid or rule-breaking turns (lower is better)": rate(scores, lambda s: s["outcome"] == "invalid"),
        "mean searches per episode": {"value": round(sum(s["n_searches"] for s in scores) / len(scores), 2) if scores else None, "k": None, "n": len(scores)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="append", required=True)
    ap.add_argument("--parse", default="lenient")
    args = ap.parse_args()
    roots = [Path(e) for e in args.eval]

    def find(name: str) -> Path | None:
        for r in roots:
            if (r / name).exists():
                return r / name
        return None

    data, closed, audit = {}, {}, {}
    for b, _ in BENCHES:
        for s in SYSTEMS:
            d = find(f"{s}_bench_{b}")
            if d and (d / args.parse / "scores.jsonl").exists():
                data[(b, s)] = scorecard.load(d / args.parse)[0]
                audit[(b, s)] = json.loads((d / args.parse / "summary.json").read_text(encoding="utf-8"))["auditkit"]["all"]
        d = find(f"closed_bench_{b}")
        if d and (d / "summary.json").exists():
            closed[b] = json.loads((d / "summary.json").read_text(encoding="utf-8"))["closed_book"]
    systems = [s for s in SYSTEMS if any((b, s) in data for b, _ in BENCHES)]
    good = lambda s: float(s["good"])

    lines = [f"# External benchmarks ({args.parse} reading)", "",
             "Evaluation only: nothing was trained on or chosen with these sets (plan section 22). Same harness as HotpotQA: shared index per benchmark, "
             "three searches, top 3, 800 characters shown. Not comparable with official leaderboard numbers.", ""]
    summary = ["| benchmark | " + " | ".join(LABEL[s] for s in systems) + " | main minus base, 95% CI | main minus candA, 95% CI | closed book |",
               "| --- |" + " ---: |" * len(systems) + " --- | --- | ---: |"]
    out_json: dict = {"parse": args.parse, "benchmarks": {}}
    pooled = {s: [] for s in systems}
    for b, title in BENCHES:
        have = [s for s in systems if (b, s) in data]
        if not have:
            continue
        m = {s: metrics(data[(b, s)]) for s in have}
        pairs = {}
        for other in ("base", "candA", "control"):
            if "main" in have and other in have:
                pairs[f"main vs {other}"] = scorecard.paired(data[(b, other)], data[(b, "main")], good)
        out_json["benchmarks"][b] = {"metrics": m, "paired_ends_well": pairs, "closed_book": closed.get(b),
                                     "auditkit": {s: {k: audit[(b, s)].get(k) for k in AUDIT} for s in have}}
        for s in have:
            pooled[s] += data[(b, s)]
        fmt_pair = lambda p: "n/a" if not p else f"{p['difference']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]"
        summary.append(f"| {title} | " + " | ".join(cell(m[s]["ends well (all scored episodes)"]) if s in m else "n/a" for s in systems)
                       + f" | {fmt_pair(pairs.get('main vs base'))} | {fmt_pair(pairs.get('main vs candA'))} | "
                       + (f"{closed[b]['closed_book_correct']:.3f}" if b in closed else "n/a") + " |")
        lines += [f"## {title}", "", "| metric | " + " | ".join(LABEL[s] for s in have) + " |", "| --- |" + " ---: |" * len(have)]
        for name in m[have[0]]:
            row = [m[s][name] for s in have]
            if all(r["n"] == 0 for r in row):
                continue
            lines.append(f"| {name} | " + " | ".join(cell(r) if r.get("k") is not None else f"{r['value']}" for r in row) + " |")
        if pairs:
            lines += ["", "| ends well, paired | before | after | difference | 95% CI | episodes / questions |", "| --- | ---: | ---: | ---: | --- | ---: |"]
            for label, p in pairs.items():
                if p:
                    lines.append(f"| {label} | {p['before']:.3f} | {p['after']:.3f} | {p['difference']:+.3f} | [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] | {p['n_episodes']} / {p['n_questions']} |")
        lines += ["", "| AuditKIT (all episodes) | " + " | ".join(LABEL[s] for s in have) + " |", "| --- |" + " ---: |" * len(have)]
        for k in AUDIT:
            lines.append(f"| {AUDIT_LABEL.get(k, k)} | " + " | ".join("n/a" if audit[(b, s)].get(k) is None else f"{audit[(b, s)][k]:.3f}" for s in have) + " |")
        if b in closed:
            lines += ["", f"Closed book (base model, question only): {closed[b]['closed_book_correct']:.3f} of {closed[b]['questions']} answerable questions."]
        lines.append("")

    pool_pairs = {f"main vs {o}": scorecard.paired(pooled[o], pooled["main"], good) for o in ("base", "candA", "control") if o in pooled and pooled[o] and pooled.get("main")}
    out_json["pooled"] = {"ends_well": {s: rate(pooled[s], lambda r: r["good"]) for s in systems}, "paired": pool_pairs}
    summary.append("| **all five, pooled** | " + " | ".join(cell(rate(pooled[s], lambda r: r["good"])) for s in systems) + " | "
                   + (f"{pool_pairs['main vs base']['difference']:+.3f} [{pool_pairs['main vs base']['ci95'][0]:+.3f}, {pool_pairs['main vs base']['ci95'][1]:+.3f}]" if "main vs base" in pool_pairs else "n/a")
                   + " | " + (f"{pool_pairs['main vs candA']['difference']:+.3f} [{pool_pairs['main vs candA']['ci95'][0]:+.3f}, {pool_pairs['main vs candA']['ci95'][1]:+.3f}]" if "main vs candA" in pool_pairs else "n/a") + " | |")
    lines[4:4] = ["## Summary: ends well (supported answer when answerable, abstain when the documents cannot answer)", ""] + summary + [""]
    (ROOT / "reports" / f"benchmarks_{args.parse}.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (ROOT / "reports" / f"benchmarks_{args.parse}.json").write_text(json.dumps(out_json, indent=1), encoding="utf-8", newline="\n")
    if args.parse == "lenient":
        figure(out_json, systems)
    print("\n".join(lines[:4 + len(summary) + 3]))


def figure(rep: dict, systems: list[str]) -> None:
    colours = {"base": "#9e9e9e", "main": "#2e7d32", "candA": "#81c784", "control": "#f9a825"}
    names = [(b, t) for b, t in BENCHES if b in rep["benchmarks"]]
    left, top, group_w, bar_h, gap = 150, 56, 4 * 16 + 14, 14, 12
    width, chart_w = 780, 560
    height = top + len(names) * (len(systems) * bar_h + gap) + 60
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="Segoe UI, Helvetica, Arial, sans-serif">',
           f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
           f'<text x="{left}" y="24" font-size="16" font-weight="600" fill="#1f2328">Ends well on five external benchmarks</text>',
           f'<text x="{left}" y="42" font-size="12" fill="#57606a">Supported answer when answerable, abstention when the documents cannot answer. Same harness for every system.</text>']
    y = top
    for b, title in names:
        out.append(f'<text x="{left - 10}" y="{y + len(systems) * bar_h / 2 + 4}" font-size="13" font-weight="600" text-anchor="end" fill="#1f2328">{title}</text>')
        for s in systems:
            r = rep["benchmarks"][b]["metrics"].get(s, {}).get("ends well (all scored episodes)")
            if r and r["value"] is not None:
                w = chart_w * r["value"]
                out.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar_h - 2}" fill="{colours[s]}"/>')
                out.append(f'<text x="{left + w + 4:.1f}" y="{y + bar_h - 4}" font-size="11" fill="#1f2328">{r["value"]:.2f}</text>')
            y += bar_h
        y += gap
    x = left
    for s in systems:
        out.append(f'<rect x="{x}" y="{y + 8}" width="12" height="12" fill="{colours[s]}"/>')
        out.append(f'<text x="{x + 17}" y="{y + 19}" font-size="12" fill="#1f2328">{LABEL[s]}</text>')
        x += 17 + 7 * len(LABEL[s]) + 22
    out.append("</svg>")
    p = ROOT / "docs" / "figures" / "benchmarks.svg"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
