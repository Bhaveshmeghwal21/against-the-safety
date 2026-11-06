"""Priority 1 gate (research/2026-11-07-round5-preregistration.md): does the answer checker pass on development data?

  python scripts/analyze_verifier_round5.py --verify runs/account2_round5/ats-verify-r5-dev/out/verify --out reports/round5_verifier

Blocks an answer when a deterministic check fires or the model's probability of the positive word is below 0.5. Gate: at least 40% of main's
unsupported answers blocked and at most 10% of its supported-correct answers blocked; the winner has the largest difference of the two rates.
Also reports the effect of V on main's development statistics (fail-closed repair applied on both sides), computed from the saved traces.
"""

import argparse
import json
from pathlib import Path

import rescore_round5 as RS
import select_round5 as SEL

GATE_UNSUPPORTED, GATE_SUPPORTED = 0.40, 0.10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", required=True)
    ap.add_argument("--cases", default="data/processed/verifier/cases_dev.jsonl")
    ap.add_argument("--eval", default="runs/account2_round4/ats-eval-jobs-r4dev/out/eval")
    ap.add_argument("--out", default="reports/round5_verifier")
    args = ap.parse_args()
    cases = RS.read(args.cases)
    key = lambda r: (r["run"], r["qid"], r["variant"])
    label = lambda c: ("unsupported" if c["labels"]["outcome_v2"].startswith("unsupported_") and not c["labels"]["excluded_v2"]
                       else c["labels"]["outcome_v2"])
    groups = {g: [c for c in cases if label(c) == g] for g in ("unsupported", "supported_correct", "wrong_answer")}
    det = lambda c: c["deterministic"]["cites_unshown"] or c["deterministic"]["answer_part_absent_from_cited"]
    configs, table = {}, []
    for tag in ("base", "main"):
        for w in ("A", "B"):
            path = Path(args.verify) / tag / f"verdicts_{w}.jsonl"
            if not path.exists():
                continue
            p = {key(r): r for r in RS.read(path)}
            blocked = {key(c): det(c) or (key(c) not in p) or p[key(c)]["p_pos"] < 0.5 for c in cases}
            only_model = {key(c): (key(c) not in p) or p[key(c)]["p_pos"] < 0.5 for c in cases}
            share = lambda g, b: sum(b[key(c)] for c in groups[g]) / len(groups[g])
            row = {"config": f"{tag}-{w}", "unsupported_blocked": share("unsupported", blocked), "supported_correct_blocked": share("supported_correct", blocked),
                   "wrong_blocked": share("wrong_answer", blocked), "model_only_unsupported_blocked": share("unsupported", only_model),
                   "model_only_supported_correct_blocked": share("supported_correct", only_model), "model_only_wrong_blocked": share("wrong_answer", only_model),
                   "top1_is_candidate": sum(r["top1_is_candidate"] for r in p.values()) / max(1, len(p))}
            row["passes"] = row["unsupported_blocked"] >= GATE_UNSUPPORTED and row["supported_correct_blocked"] <= GATE_SUPPORTED
            row["score"] = row["unsupported_blocked"] - row["supported_correct_blocked"]
            configs[row["config"]] = blocked
            table.append(row)
    passing = [r for r in table if r["passes"]]
    winner = max(passing, key=lambda r: r["score"])["config"] if passing else None

    # effect of V on main's development statistics: blocked answers end as abstentions
    runs = {f.parent.name: f for f in Path(args.eval).rglob("lenient") if (f / "transcripts.jsonl").exists()}
    effect = None
    if winner:
        sets = {}
        for name in ("main_bench_r4dev", "main_dev"):
            _, _, rows, _ = RS.rescore_run(runs[name])
            rows = SEL.fail_closed(rows)
            sets[name] = rows
        before = SEL.stats(sets["main_bench_r4dev"], sets["main_dev"])
        after_sets = {}
        for name, rows in sets.items():
            new = []
            for r in rows:
                if r["final"] == "answer" and configs[winner].get((name, r["qid"], r["variant"])):
                    o = "false_abstain" if r["answerable"] else "correct_abstain"
                    r = {**r, "outcome_v2": o, "good_v2": not r["answerable"], "strong_v2": not r["answerable"], "final": "abstain"}
                new.append(r)
            after_sets[name] = new
        effect = {"main_with_FB": before, "main_with_FB_and_V": SEL.stats(after_sets["main_bench_r4dev"], after_sets["main_dev"]), "config": winner}
    out = Path(args.out)
    result = {"groups": {g: len(v) for g, v in groups.items()}, "configs": table, "selected": winner, "effect_on_main": effect,
              "gate": {"unsupported_blocked_at_least": GATE_UNSUPPORTED, "supported_correct_blocked_at_most": GATE_SUPPORTED}}
    out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    f = lambda x: f"{100 * x:.0f}%"
    lines = ["# Round 5 priority 1: the answer checker on development data", "",
             f"Cases: {result['groups']}. Gate: block at least {f(GATE_UNSUPPORTED)} of unsupported answers and at most {f(GATE_SUPPORTED)} of supported-correct answers (deterministic checks plus the model check).", "",
             "| configuration | unsupported blocked | supported correct blocked | wrong blocked | model only: unsupported / supported / wrong | first token is a candidate | passes |",
             "| --- | ---: | ---: | ---: | --- | ---: | --- |"]
    for r in table:
        lines.append(f"| {r['config']} | {f(r['unsupported_blocked'])} | {f(r['supported_correct_blocked'])} | {f(r['wrong_blocked'])} | "
                     f"{f(r['model_only_unsupported_blocked'])} / {f(r['model_only_supported_correct_blocked'])} / {f(r['model_only_wrong_blocked'])} | {f(r['top1_is_candidate'])} | {r['passes']} |")
    lines += ["", f"Selected: **{winner or 'none; V is dropped'}**."]
    if effect:
        b, a = effect["main_with_FB"], effect["main_with_FB_and_V"]
        lines += ["", "Main on development data (scorer v2, fail-closed applied):", "", "| | macro | supported correct | unsupported / hidden | false refusals | HotpotQA unsupported |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for n, s in (("FB", b), ("FB + V", a)):
            lines.append(f"| {n} | {s['macro']:.3f} | {s['supported_correct']:.3f} | {s['unsupported']}/{s['hidden_n']} | {s['false_abstain']:.3f} | {s['hotpot_unsupported']} |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
