"""Sample HotpotQA splits and write the manifest plus processed question files.

Run (after `. scripts/env.ps1`):  python scripts/prepare_data.py
Downloads only two parquet shards into .cache/huggingface (about 190 MB).
"""

import argparse
import json
from pathlib import Path

from ragharness import data

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--n-dev", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=100)
    args = ap.parse_args()

    files = data.download(ROOT / ".cache" / "huggingface" / "hub")
    train_rows, rej_train = data.load_questions(files[data.TRAIN_FILE])
    val_rows, rej_val = data.load_questions(files[data.VAL_FILE])
    print(f"train pool {len(train_rows)} rows, test pool {len(val_rows)} rows, rejected {dict(rej_train)} {dict(rej_val)}")

    splits = data.sample_splits(train_rows, val_rows, args.n_train, args.n_dev, args.n_test, args.seed)
    man = data.manifest(splits, args.seed, {"train_pool": rej_train, "test_pool": rej_val})

    out = ROOT / "data"
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    (out / "manifests" / "split_manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8", newline="\n")
    for name, qs in splits.items():
        data.write_questions(out / "processed" / f"questions_{name}.jsonl", qs)
        print(name, man["splits"][name]["n"], man["splits"][name]["by_type"], man["splits"][name]["by_level"])


if __name__ == "__main__":
    main()
