"""Round 3 training sets (plan section 24): candC and candD, patches that continue from the main adapter.

Run (after prepare_round3.py --only train):  python scripts/make_patch_sets.py --tokenizer .cache/tiny-aya-global
Writes data/processed/training/{candC,candD}.jsonl (git-ignored) and reports/training_sets_round3.json (committed).
Each set = new-family rows (episodes_train3) + rehearsal rows (decisions from main.jsonl), at the same processed-token budget.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from ragharness import mixture, round3

ROOT = Path(__file__).resolve().parents[1]
SEED = 23
TRAINING = ROOT / "data" / "processed" / "training"


def read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def describe(rows: list[dict]) -> dict:
    new = [r for r in rows if r["meta"]["split"] == "train3"]
    toks = lambda rs: sum(r["meta"]["est_tokens"] for r in rs)
    return {
        **mixture.summarize(rows),
        "new_rows": len(new), "new_share_of_rows": round(len(new) / len(rows), 3), "new_share_of_tokens": round(toks(new) / toks(rows), 3),
        "new_by_family_and_kind": {f"{fam}:{k}": sum(1 for r in new if r["meta"]["family"] == fam and r["meta"]["kind"] == k)
                                   for fam in ("single", "compound", "hotpot") for k in ("search", "answer", "abstain")},
        "new_by_variant": dict(sorted(Counter(r["meta"]["variant"] for r in new).items())),
        "finals_by_searches_before": mixture.depth_table(rows),
        "abstain_share_of_finals": round(sum(r["meta"]["kind"] == "abstain" for r in rows) / sum(r["meta"]["kind"] != "search" for r in rows), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-tokens", type=int, default=2560)
    ap.add_argument("--tokenizer", default=".cache/tiny-aya-global")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(ROOT / args.tokenizer))
    new = mixture.annotate(read(ROOT / "data" / "processed" / "episodes" / "rows_train3.jsonl"), tok)
    main_rows = mixture.annotate(read(TRAINING / "main.jsonl"), tok)
    clash = {r["meta"]["parent_qid"] for r in new if r["meta"]["family"] != "hotpot"} & {r["meta"]["parent_qid"] for r in main_rows}
    assert not clash, "a SQuAD question id equals a HotpotQA question id"
    report = {"token_counts": "exact (Tiny Aya tokenizer)", "target_processed_tokens": round3.PATCH_TOKENS, "family_share_of_new_rows": round3.FAMILY_SHARE,
              "new_rows_available": len(new), "rehearsal_pool": "main.jsonl (1,611 rows the main adapter already trained on)", "main": describe(main_rows)}
    for name, cfg in round3.PATCHES.items():
        rows, info = round3.build_patch(new, main_rows, SEED, args.cap_tokens, round3.PATCH_TOKENS, cfg["new_share"], cfg["abstain_new"], cfg["abstain_replay"], round3.FAMILY_SHARE)
        keys = [(r["meta"]["parent_qid"], r["meta"]["variant"], r["meta"]["step"]) for r in rows]
        assert len(keys) == len(set(keys)), f"{name}: duplicate decisions"
        assert all(r["meta"]["est_tokens"] <= args.cap_tokens for r in rows)
        write(TRAINING / f"{name}.jsonl", rows)
        report[name] = {"config": cfg, **describe(rows), "fill": info}
    (ROOT / "reports" / "training_sets_round3.json").write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in ("config", "rows", "parents", "by_kind", "new_rows", "new_share_of_rows", "new_share_of_tokens", "processed_tokens_est", "supervised_tokens_est", "abstain_share_of_finals", "finals_by_searches_before", "new_by_family_and_kind")}) for k, v in report.items()}, indent=1))


if __name__ == "__main__":
    main()
