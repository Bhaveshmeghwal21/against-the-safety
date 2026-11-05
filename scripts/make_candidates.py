"""Second-round training sets (plan section 20): candA and candB, drawn from the same train decisions as main.

Run (after make_training_set.py):  python scripts/make_candidates.py --tokenizer .cache/tiny-aya-global
Writes data/processed/training/{candA,candB}.jsonl (git-ignored) and reports/training_sets_round2.json (committed).
main.jsonl, control.jsonl and dev.jsonl are read, never written.
"""

import argparse
import json
from pathlib import Path

from ragharness import corpus as corpus_mod, data, mixture

ROOT = Path(__file__).resolve().parents[1]
SEED = 13
TRAINING = ROOT / "data" / "processed" / "training"


def read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def protected_chunks() -> set[str]:
    """Support copies of dev, test and (if built) test2 questions: no training row may show one."""
    man = json.loads((ROOT / "data" / "manifests" / "split_manifest.json").read_text(encoding="utf-8"))
    splits = {n: data.read_questions(ROOT / "data" / "processed" / f"questions_{n}.jsonl") for n in ("train", "dev", "test")}
    corp = corpus_mod.build_corpus([q for n in splits for q in splits[n]], seed=man["seed"])
    out = {c for n in ("dev", "test") for q in splits[n] for c in corp.support_copies(q)}
    t2 = ROOT / "data" / "processed" / "questions_test2.jsonl"
    if t2.exists():
        q2 = data.read_questions(t2)
        corp2 = corpus_mod.extend_corpus(corp, q2)
        out |= {c for q in q2 for c in corp2.support_copies(q)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-tokens", type=int, default=2560)
    ap.add_argument("--tokenizer", default=".cache/tiny-aya-global")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(ROOT / args.tokenizer))
    pool = mixture.annotate(read(ROOT / "data" / "processed" / "episodes" / "rows_train.jsonl"), tok)
    pool, n_dropped = mixture.drop_overlap(pool, protected_chunks())
    main_rows = mixture.annotate(read(TRAINING / "main.jsonl"), tok)
    target = sum(r["meta"]["est_tokens"] for r in main_rows)

    a, info_a = mixture.build_cand_a(pool, SEED, args.cap_tokens, target)
    b, info_b = mixture.build_cand_b(pool, SEED, args.cap_tokens, target)
    for name, rows in (("candA", a), ("candB", b)):
        keys = [(r["meta"]["parent_qid"], r["meta"]["variant"], r["meta"]["step"]) for r in rows]
        assert len(keys) == len(set(keys)), f"{name}: duplicate decisions"
        assert all(r["meta"]["est_tokens"] <= args.cap_tokens for r in rows)
        write(TRAINING / f"{name}.jsonl", rows)

    report = {
        "token_counts": "exact (Tiny Aya tokenizer)",
        "pool_rows_after_drop": len(pool),
        "pool_rows_dropped_for_showing_dev_test_or_test2_support": n_dropped,
        "target_processed_tokens (main)": target,
        "main": {**mixture.summarize(main_rows), "finals_by_searches_before": mixture.depth_table(main_rows)},
        "candA": {**mixture.summarize(a), "finals_by_searches_before": mixture.depth_table(a), "fill": info_a,
                  "relative_token_difference_vs_main": round(info_a["tokens"] / target - 1, 4)},
        "candB": {**mixture.summarize(b), "finals_by_searches_before": mixture.depth_table(b), "fill": info_b,
                  "relative_token_difference_vs_main": round(info_b["tokens"] / target - 1, 4),
                  "search_rows_by_step": {s: sum(r["meta"]["kind"] == "search" and r["meta"]["step"] == s for r in b) for s in (0, 1, 2)}},
    }
    report["main"]["search_rows_by_step"] = {s: sum(r["meta"]["kind"] == "search" and r["meta"]["step"] == s for r in main_rows) for s in (0, 1, 2)}
    (ROOT / "reports" / "training_sets_round2.json").write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
