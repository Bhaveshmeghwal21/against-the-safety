"""Apply the third-round selection rule (plan section 24) to the development results -> reports/round3_selection.md/.json.

  python scripts/select_round3.py --eval runs/kaggle/ats-patch-r3/out/eval --main-dev runs/kaggle/ats-eval-dev/out/eval/main_dev/lenient

Components (each scored by "ends well"): D1 HotpotQA dev headline (open + support masked); D2 single-passage questions (open + support masked);
D3 compound questions (open + support masked + part masked); D4 prompt variants (noisy + negatives). Reads <system>_dev/lenient and
<system>_bench_r3dev/lenient from the --eval directories; main's D1 comes from the first-round dev run (--main-dev), which has the same
headline episodes. The rule is fixed in plan section 24; this script only applies it.
"""

import argparse
import json
from pathlib import Path

from ragharness import scorecard

ROOT = Path(__file__).resolve().parents[1]
HEAD = ("open", "support_masked")
MAX_UNSUPPORTED_D1 = 11  # of the 73 unanswerable dev headline episodes
MARGIN_G1, MARGIN_G3 = 0.03, 0.03
MIN_GAIN, MAX_D1_LOSS = 0.05, 0.03
CANDIDATES = ("candC", "candD")
PREFER = ("candD", "candC")  # ties go to the first one listed


def components(dev: list[dict], r3: list[dict]) -> dict[str, list[dict]]:
    return {
        "D1 HotpotQA dev": [s for s in dev if s["variant"] in HEAD],
        "D2 single-passage": [s for s in r3 if s["qtype"] == "single" and s["variant"] in HEAD],
        "D3 compound": [s for s in r3 if s["qtype"] == "compound" and s["variant"] in HEAD + ("part_masked",)],
        "D4 prompt variants": [s for s in r3 if s["variant"] in ("prefill_noisy", "prefill_negatives")],
    }


def unsupported(s: dict) -> bool:
    return s["outcome"] in ("unsupported_correct", "unsupported_wrong")


def stats(comps: dict[str, list[dict]]) -> dict:
    allrows = [s for rows in comps.values() for s in rows]
    hidden = [s for s in allrows if not s["answerable"]]
    d1_hidden = [s for s in comps["D1 HotpotQA dev"] if not s["answerable"]]
    per = {}
    for name, rows in comps.items():
        ans, un = [s for s in rows if s["answerable"]], [s for s in rows if not s["answerable"]]
        per[name] = {
            "n": len(rows), "good": sum(bool(s["good"]) for s in rows), "rate": sum(bool(s["good"]) for s in rows) / len(rows),
            "supported_correct": [sum(s["outcome"] == "supported_correct" for s in ans), len(ans)],
            "false_abstain": [sum(s["outcome"] == "false_abstain" for s in ans), len(ans)],
            "correct_abstain": [sum(s["outcome"] == "correct_abstain" for s in un), len(un)],
            "unsupported": [sum(unsupported(s) for s in un), len(un)],
            "invalid": [sum(s["outcome"] == "invalid" for s in rows), len(rows)],
        }
    return {
        "components": per, "macro": sum(p["rate"] for p in per.values()) / len(per),
        "pooled_unsupported": [sum(unsupported(s) for s in hidden), len(hidden)],
        "d1_unsupported": [sum(unsupported(s) for s in d1_hidden), len(d1_hidden)],
        "pooled_invalid": [sum(s["outcome"] == "invalid" for s in allrows), len(allrows)],
    }


def find(roots: list[Path], name: str) -> Path | None:
    for r in roots:
        if (r / name / "lenient" / "scores.jsonl").exists():
            return r / name / "lenient"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="append", required=True)
    ap.add_argument("--main-dev", required=True, help="first-round dev run of the main adapter (its headline episodes are D1)")
    ap.add_argument("--candidates", default=",".join(CANDIDATES), help="comma list of candidate systems, in display order")
    ap.add_argument("--prefer", default=",".join(PREFER), help="comma list: on equal macro the first listed wins")
    ap.add_argument("--report", default="round3_selection", help="file stem written under reports/")
    args = ap.parse_args()
    candidates, tie_order = tuple(args.candidates.split(",")), {n: i for i, n in enumerate(args.prefer.split(","))}
    roots = [Path(e) for e in args.eval]
    st: dict[str, dict] = {}
    for sysname in ("main",) + candidates:
        dev_dir = Path(args.main_dev) if sysname == "main" else find(roots, f"{sysname}_dev")
        r3_dir = find(roots, f"{sysname}_bench_r3dev")
        if dev_dir is None or r3_dir is None:
            print(f"missing results for {sysname}; skipped")
            continue
        st[sysname] = stats(components(scorecard.load(dev_dir)[0], scorecard.load(r3_dir)[0]))
    main_s = st["main"]
    rate = lambda kn: kn[0] / kn[1] if kn[1] else 0.0
    for name, s in st.items():
        if name == "main":
            s["eligible"] = True
            continue
        s["g1"] = rate(s["pooled_unsupported"]) <= rate(main_s["pooled_unsupported"]) + MARGIN_G1
        s["g2"] = s["d1_unsupported"][0] <= MAX_UNSUPPORTED_D1
        s["g3"] = rate(s["pooled_invalid"]) <= rate(main_s["pooled_invalid"]) + MARGIN_G3
        s["eligible"] = s["g1"] and s["g2"] and s["g3"]
    cands = [c for c in candidates if c in st]
    eligible = [c for c in cands if st[c]["eligible"]]
    best = max(eligible, key=lambda c: (st[c]["macro"], -tie_order.get(c, 99))) if eligible else None
    d1 = lambda n: st[n]["components"]["D1 HotpotQA dev"]["rate"]
    replaces = bool(best) and st[best]["macro"] >= main_s["macro"] + MIN_GAIN and d1(best) >= d1("main") - MAX_D1_LOSS
    headline = best if replaces else "main"

    comp_names = list(main_s["components"])
    lines = ["# Third-round selection on the development set (rule fixed in plan section 24)", "",
             f"Eligible: (G1) pooled unsupported-answer rate at most {MARGIN_G1} above main's; (G2) at most {MAX_UNSUPPORTED_D1} of 73 unsupported answers on D1; "
             f"(G3) pooled invalid-turn rate at most {MARGIN_G3} above main's.",
             f"Winner: the eligible candidate with the highest macro average of the four components (ties: first in the preference list). It replaces main only if its macro is at least "
             f"main's plus {MIN_GAIN} and its D1 score is at least main's minus {MAX_D1_LOSS}.", "",
             "| system | " + " | ".join(f"{c} ends well" for c in comp_names) + " | macro | pooled unsupported answers | pooled invalid | G1 | G2 | G3 | eligible |",
             "| --- |" + " ---: |" * len(comp_names) + " ---: | ---: | ---: | --- | --- | --- | --- |"]
    for name, s in st.items():
        cells = " | ".join(f"{s['components'][c]['rate']:.3f} ({s['components'][c]['good']}/{s['components'][c]['n']})" for c in comp_names)
        yn = lambda k: "-" if name == "main" else ("yes" if s[k] else "no")
        lines.append(f"| {name} | {cells} | {s['macro']:.3f} | {s['pooled_unsupported'][0]}/{s['pooled_unsupported'][1]} | {s['pooled_invalid'][0]}/{s['pooled_invalid'][1]} "
                     f"| {yn('g1')} | {yn('g2')} | {yn('g3')} | {'yes' if s['eligible'] else 'no'} |")
    lines += ["", "Detail per component (supported correct / false abstain among answerable; correct abstain / unsupported answers among hidden-evidence; invalid turns):", ""]
    for name, s in st.items():
        for c, p in s["components"].items():
            f = lambda kn: f"{kn[0]}/{kn[1]}"
            lines.append(f"- {name}, {c}: supported {f(p['supported_correct'])}, false abstain {f(p['false_abstain'])}, correct abstain {f(p['correct_abstain'])}, unsupported {f(p['unsupported'])}, invalid {f(p['invalid'])}")
    lines += ["", f"**Winner: {best or 'none eligible'}.** Macro {('%.3f' % st[best]['macro']) if best else 'n/a'} against main's {main_s['macro']:.3f}; "
              f"D1 {('%.3f' % d1(best)) if best else 'n/a'} against {d1('main'):.3f}.",
              f"**Headline model after development: {headline}.** " + ("The winner replaces main." if replaces else "Main stays the headline model; the winner, if any, is published as a variant."),
              "If there is a winner, main and the winner are evaluated once on the fresh test set (r3t_*)."]
    (ROOT / "reports" / f"{args.report}.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (ROOT / "reports" / f"{args.report}.json").write_text(json.dumps({"stats": st, "winner": best, "headline": headline, "replaces_main": replaces}, indent=2), encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
