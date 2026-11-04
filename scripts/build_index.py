"""Build the shared SQLite FTS index over every sampled paragraph (train + dev + test).

Run (after `. scripts/env.ps1`):  python scripts/build_index.py
Writes data/index/corpus.sqlite (ignored by git) and data/manifests/index_manifest.json (committed).
"""

import json
from importlib.metadata import version
from pathlib import Path

from ragharness import corpus as corpus_mod
from ragharness import data

ROOT = Path(__file__).resolve().parents[1]


def load_all() -> dict[str, list[data.Question]]:
    return {n: data.read_questions(ROOT / "data" / "processed" / f"questions_{n}.jsonl") for n in ("train", "dev", "test")}


def main() -> None:
    manifest = json.loads((ROOT / "data" / "manifests" / "split_manifest.json").read_text(encoding="utf-8"))
    splits = load_all()
    allq = [q for n in ("train", "dev", "test") for q in splits[n]]
    corp = corpus_mod.build_corpus(allq, seed=manifest["seed"])
    corpus_mod.build_index(corp, ROOT / "data" / "index" / "corpus.sqlite")

    info = {
        "index_hash": corp.index_hash,
        "n_chunks": len(corp.chunks),
        "n_questions": len(allq),
        "paragraphs_before_dedup": sum(len(q.paragraphs) for q in allq),
        "backend": "agenttune.rag.retrieval.sqlite_fts.SQLiteFTSBackend",
        "match_all": False,
        "agenttune_version": version("agenttune"),
        "split_seed": manifest["seed"],
        "hotpotqa_revision": manifest["revision"],
    }
    (ROOT / "data" / "manifests" / "index_manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
