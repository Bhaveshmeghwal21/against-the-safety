"""Retrieval ceiling on the DEV split (plan section 4.3). Test questions are not scored here.

Run (after `. scripts/env.ps1` and build_index.py):  python scripts/retrieval_ceiling.py
"""

import json
import statistics
from pathlib import Path

from agenttune.rag.retrieval.sqlite_fts import SQLiteFTSBackend

from ragharness import ceiling, corpus as corpus_mod, data
from ragharness.retrieval import MaskedSearch

ROOT = Path(__file__).resolve().parents[1]
BUDGET = 3
TOP_KS = (3, 5, 8)


def main() -> None:
    man = json.loads((ROOT / "data" / "manifests" / "split_manifest.json").read_text(encoding="utf-8"))
    idx = json.loads((ROOT / "data" / "manifests" / "index_manifest.json").read_text(encoding="utf-8"))
    splits = {n: data.read_questions(ROOT / "data" / "processed" / f"questions_{n}.jsonl") for n in ("train", "dev", "test")}
    corp = corpus_mod.build_corpus([q for n in ("train", "dev", "test") for q in splits[n]], seed=man["seed"])
    assert corp.index_hash == idx["index_hash"], "index manifest does not match the processed questions"

    search = MaskedSearch(SQLiteFTSBackend(str(ROOT / "data" / "index" / "corpus.sqlite"), match_all=False))
    dev = splits["dev"]
    report = {"index_hash": corp.index_hash, "split": "dev", "n": len(dev), "budget": BUDGET, "results": {}}
    chars: list[int] = []
    for bridge_query in ("mention", "mention+question"):
        for top_k in TOP_KS:
            runs = [ceiling.run_oracle(q, corp, search, top_k, BUDGET, bridge_query) for q in dev]
            report["results"][f"{bridge_query}|top_k={top_k}"] = ceiling.summarize(runs, BUDGET)
            if bridge_query == "mention" and top_k == 3:
                chars = [c for r in runs for c in r.hit_chars]
    report["hit_chars_top_k3"] = {
        "mean": round(statistics.mean(chars)), "p95": sorted(chars)[int(0.95 * len(chars))], "max": max(chars),
    }

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "retrieval_ceiling_dev.json").write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")

    print(f"index {corp.index_hash[:12]}  chunks {len(corp.chunks)}  dev n={len(dev)}")
    print(f"{'setting':34s} {'budget':>6s} {'all':>7s} {'bridge':>7s} {'compar':>7s}   (complete support-set recall)")
    for setting, res in report["results"].items():
        for b in range(1, BUDGET + 1):
            r = res[f"searches<={b}"]
            print(f"{setting:34s} {b:6d} {r['all']['complete_recall']:7.2f} {r['bridge']['complete_recall']:7.2f} {r['comparison']['complete_recall']:7.2f}")
    print("passage chars per hit (top_k=3):", report["hit_chars_top_k3"])


if __name__ == "__main__":
    main()
