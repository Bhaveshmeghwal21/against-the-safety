"""CPU-only retrieval-score gate baseline: threshold fit on DEV, reported on TEST.

Each accepted episode contributes the BM25 score of its first search (query = question, with the episode's masks).
"Best of searches" takes the highest top-1 score over the scripted episode's searches.

Run (after build_episodes.py):  python scripts/gate_baseline.py
"""

import json
from pathlib import Path

from agenttune.rag.retrieval.sqlite_fts import SQLiteFTSBackend

from ragharness import builder, corpus as corpus_mod, data, gate
from ragharness.retrieval import MaskedSearch
from ragharness.tools import CorpusSearchTool

ROOT = Path(__file__).resolve().parents[1]


def collect(split, qs, corp, tool, search) -> list[gate.GateSample]:
    out = []
    for q in qs:
        specs = builder.make_specs(q, corp, search)
        for b in builder.build_question(q, split, corp, tool, search):
            if b.episode is None:
                continue
            spec = specs[b.variant]
            ep = b.episode
            first = gate.first_search_score(search, q.question, tool.top_k, spec.masked | spec.first_search_masked)
            best = max([h.score for s in ep.steps for h in s.hits[:1]] + [first])
            out.append(gate.GateSample(q.qid, b.variant, spec.answerable, first, best))
    return out


def main() -> None:
    man = json.loads((ROOT / "data" / "manifests" / "split_manifest.json").read_text(encoding="utf-8"))
    splits = {n: data.read_questions(ROOT / "data" / "processed" / f"questions_{n}.jsonl") for n in ("train", "dev", "test")}
    corp = corpus_mod.build_corpus([q for n in splits for q in splits[n]], seed=man["seed"])
    backend = SQLiteFTSBackend(str(ROOT / "data" / "index" / "corpus.sqlite"), match_all=False)
    tool, search = CorpusSearchTool(backend, top_k=3), MaskedSearch(backend)

    dev = collect("dev", splits["dev"], corp, tool, search)
    test = collect("test", splits["test"], corp, tool, search)
    report: dict = {"note": "decision baseline with an oracle reader; threshold fit on dev, applied to test"}
    for key in ("top1", "top1_best_of_searches"):
        tau, dev_m = gate.best_threshold(dev, key)
        test_m = gate.evaluate(test, tau, key)
        by_variant = {}
        for v in sorted({s.variant for s in test}):
            sub = [s for s in test if s.variant == v]
            by_variant[v] = {"n": len(sub), "answerable": sub[0].answerable,
                             "abstain_rate": round(sum(getattr(s, key) < tau for s in sub) / len(sub), 3)}
        report[key] = {"dev": dev_m, "test": test_m, "test_abstain_rate_by_variant": by_variant}
    (ROOT / "reports" / "gate_baseline.json").write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
