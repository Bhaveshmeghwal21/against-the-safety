"""Fresh held-out split for the second round (plan section 20): 100 HotpotQA validation questions and their own index.

Run:  python scripts/prepare_test2.py
Writes data/processed/questions_test2.jsonl, data/index/corpus_test2.sqlite and data/manifests/test2_manifest.json.
Nothing of the first round is rewritten: the base corpus keeps every chunk ID, and the new paragraphs are appended.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from ragharness import corpus as corpus_mod, data

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    man = json.loads((ROOT / "data" / "manifests" / "split_manifest.json").read_text(encoding="utf-8"))
    idx = json.loads((ROOT / "data" / "manifests" / "index_manifest.json").read_text(encoding="utf-8"))
    splits = {n: data.read_questions(ROOT / "data" / "processed" / f"questions_{n}.jsonl") for n in ("train", "dev", "test")}
    base = corpus_mod.build_corpus([q for n in splits for q in splits[n]], seed=man["seed"])
    assert base.index_hash == idx["index_hash"], "base corpus differs from the first-round index"

    files = data.download(ROOT / ".cache" / "huggingface" / "hub")
    val_rows, rejected = data.load_questions(files[data.VAL_FILE])
    test2 = data.sample_test2(val_rows, splits, args.n, man["seed"])
    corp = corpus_mod.extend_corpus(base, test2)
    base_ids = {c["chunk_id"]: (c["title"], c["text"]) for c in base.chunks}
    assert all(base_ids[c["chunk_id"]] == (c["title"], c["text"]) for c in corp.chunks[: len(base.chunks)])
    support = {c for q in test2 for c in corp.support_copies(q)}
    assert not support & set(base_ids), "a test2 support passage is already in the first-round corpus"

    data.write_questions(ROOT / "data" / "processed" / "questions_test2.jsonl", test2)
    db = ROOT / "data" / "index" / "corpus_test2.sqlite"
    corpus_mod.build_index(corp, db)
    out = {
        "dataset": data.DATASET, "revision": data.REVISION, "file": data.VAL_FILE, "seed": man["seed"], "order_seed": man["seed"] + 4,
        "selection_rule": "type-stratified seeded order over the validation pool; skip test questions; skip any question whose support title "
                          "is the title of any paragraph already in the corpus (train, dev or test, support or distractor)",
        "n": len(test2), "by_type": dict(Counter(q.qtype for q in test2)), "by_level": dict(Counter(q.level for q in test2)),
        "ids": [q.qid for q in test2],
        "base_index_hash": base.index_hash, "index_hash": corp.index_hash,
        "base_chunks": len(base.chunks), "n_chunks": len(corp.chunks), "new_chunks": len(corp.chunks) - len(base.chunks),
        "load_rejections": dict(rejected),
    }
    (ROOT / "data" / "manifests" / "test2_manifest.json").write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in out.items() if k != "ids"}, indent=1))


if __name__ == "__main__":
    main()
