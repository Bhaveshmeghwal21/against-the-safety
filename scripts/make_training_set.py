"""Select the main and control training sets and the dev set from the built rows.

Run (after build_episodes.py):  python scripts/make_training_set.py [--tokenizer .cache/tiny-aya-global] [--cap-tokens 2560] [--max-ratio 1.0]
Writes data/processed/training/{main,control,dev}.jsonl (git-ignored) and reports/training_sets.json (committed).
"""

import argparse
import json
from pathlib import Path

from ragharness import corpus as corpus_mod, data, mixture

ROOT = Path(__file__).resolve().parents[1]
SEED = 13


def read(split: str) -> list[dict]:
    p = ROOT / "data" / "processed" / "episodes" / f"rows_{split}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-tokens", type=int, default=2560)
    ap.add_argument("--max-ratio", type=float, default=1.0)
    ap.add_argument("--tokenizer", default=None, help="HF id or path of the Tiny Aya tokenizer; counts become exact (needs the gate)")
    args = ap.parse_args()

    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer, revision="af89d219b53ed9b13b8a4645f9c8028973510324" if "/" in args.tokenizer and not Path(args.tokenizer).exists() else None)
    train = mixture.annotate(read("train"), tok)
    dev = mixture.annotate(read("dev"), tok)
    # Plan section 5.2: no SFT row may show a dev or test support passage, so document-level transfer holds for all of them.
    man = json.loads((ROOT / "data" / "manifests" / "split_manifest.json").read_text(encoding="utf-8"))
    splits = {n: data.read_questions(ROOT / "data" / "processed" / f"questions_{n}.jsonl") for n in ("train", "dev", "test")}
    corp = corpus_mod.build_corpus([q for n in splits for q in splits[n]], seed=man["seed"])
    protected = {c for n in ("dev", "test") for q in splits[n] for c in corp.support_copies(q)}
    train_before = len(train)
    train, n_dropped = mixture.drop_overlap(train, protected)
    over = [r for r in train if r["meta"]["est_tokens"] > args.cap_tokens]
    main_rows = mixture.build_main(train, SEED, args.cap_tokens, args.max_ratio)
    control_rows, control_info = mixture.build_control(train, main_rows, SEED, args.cap_tokens)
    dev_rows = [r for r in dev if r["meta"]["est_tokens"] <= args.cap_tokens]

    out = ROOT / "data" / "processed" / "training"
    write(out / "main.jsonl", main_rows)
    write(out / "control.jsonl", control_rows)
    write(out / "dev.jsonl", dev_rows)

    m, c = mixture.summarize(main_rows), mixture.summarize(control_rows)
    report = {
        "cap_tokens": args.cap_tokens,
        "token_counts": "exact (Tiny Aya tokenizer)" if tok is not None else "estimated from characters at 3.5 chars/token",
        "train_rows_total": train_before,
        "train_rows_dropped_for_showing_dev_or_test_support": n_dropped,
        "train_rows_after_drop": len(train),
        "train_rows_over_cap": len(over),
        "train_rows_over_cap_by_kind": {k: sum(r["meta"]["kind"] == k for r in over) for k in ("search", "answer", "abstain")},
        "main": m,
        "control": {**c, **control_info},
        "dev": mixture.summarize(dev_rows),
        "supervised_token_difference_control_vs_main": c["supervised_tokens_est"] - m["supervised_tokens_est"],
    }
    (ROOT / "reports" / ("training_sets.json" if tok is None else "training_sets_exact.json")).write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
