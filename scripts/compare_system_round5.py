"""Compare the packaged system's decisions with the offline round-5 evaluation of main + fail-closed + the checker, episode by episode.

  python scripts/compare_system_round5.py --system runs/account2_round5/ats-check-system-r5/out/system_check/system_r5t_hotpot.jsonl \
      --eval runs/account2_round5/ats-eval-jobs-r5test-ref/out/eval --bench r5t_hotpot \
      --cases data/processed/verifier/cases_r5test_refs.jsonl --verdicts runs/account2_round5/ats-verify-main-r5-r5test-refs/out/verify/main \
      --out reports/round5_system_check

The two runs use the same model, the same greedy decoding and the same episodes, but the offline traces were generated in padded batches of up to 12
and the packaged system runs one episode at a time, so fp16 arithmetic can flip a token. This script counts how often that happens.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import rescore_round5 as RS
import select_round5 as S


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--cases", action="append", required=True)
    ap.add_argument("--verdicts", action="append", required=True)
    ap.add_argument("--out", default="reports/round5_system_check")
    args = ap.parse_args()
    run = f"main_bench_{args.bench}"
    folder = Path(args.eval) / run / "lenient"
    _, _, raw, _ = RS.rescore_run(folder)
    cases, verdict = S.load_v(args.cases, args.verdicts)
    offline_raw = {(r["qid"], r["variant"]): r["final"] for r in raw}
    final = S.apply_v(S.fail_closed(raw), run, cases, verdict)
    offline = {(r["qid"], r["variant"]): r["final"] for r in final}
    mine = {}
    for l in Path(args.system).read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            mine[(r["qid"], r["variant"])] = r
    keys = sorted(set(offline) & set(mine))
    assert keys, "no episodes in common"
    same_final = [k for k in keys if mine[k]["decision"] == offline[k]]
    same_raw = [k for k in keys if (mine[k]["model_final"]) == offline_raw[k]]
    both_answer = [k for k in keys if mine[k]["decision"] == "answer" and offline[k] == "answer"]
    report = {
        "bench": args.bench, "episodes_compared": len(keys), "only_in_offline": len(set(offline) - set(mine)), "only_in_system": len(set(mine) - set(offline)),
        "same_final_decision": len(same_final), "same_model_ending_before_repairs": len(same_raw),
        "decisions_system": dict(Counter(mine[k]["decision"] for k in keys)), "decisions_offline": dict(Counter(offline[k] for k in keys)),
        "repairs_system": dict(Counter(str(mine[k]["repaired_by"]) for k in keys)),
        "disagreements": [{"qid": k[0], "variant": k[1], "system": mine[k]["decision"], "offline": offline[k], "system_model_final": mine[k]["model_final"],
                           "offline_model_final": offline_raw[k], "p_supported": mine[k]["p_supported"]} for k in keys if k not in set(same_final)],
        "both_answer": len(both_answer),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "disagreements"}, indent=2))
    print(f"{len(report['disagreements'])} disagreements")


if __name__ == "__main__":
    main()
