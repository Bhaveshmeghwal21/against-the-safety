"""Build the matched-state episodes for every split, record them through AgentTune, and write Tiny Aya rows.

Outputs (data/processed/episodes/, git-ignored; rebuild with this script):
  episodes_{split}.jsonl     one line per accepted episode: meta, user turn, steps, final turn
  eventlogs_{split}.jsonl    the same episodes as AgentTune EventLogs (AuditKIT reads these)
  rows_{split}.jsonl         one Tiny Aya prompt/completion row per decision, with meta
  trajectories.sqlite/.jsonl AgentTune TrajectoryStore export, the trace dataset for the HF collection
And reports/episode_stats.json (committed).

Run (after `. scripts/env.ps1`, prepare_data.py, build_index.py):  python scripts/build_episodes.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from agenttune.rag.retrieval.sqlite_fts import SQLiteFTSBackend
from agenttune.rag.storage.trajectory_store import TrajectoryStore

from ragharness import builder, corpus as corpus_mod, data, episodes, protocol, trainer
from ragharness.retrieval import MaskedSearch
from ragharness.tools import CorpusSearchTool

ROOT = Path(__file__).resolve().parents[1]
TOP_K, BUDGET = 3, 3


def main() -> None:
    man = json.loads((ROOT / "data" / "manifests" / "split_manifest.json").read_text(encoding="utf-8"))
    idx = json.loads((ROOT / "data" / "manifests" / "index_manifest.json").read_text(encoding="utf-8"))
    splits = {n: data.read_questions(ROOT / "data" / "processed" / f"questions_{n}.jsonl") for n in ("train", "dev", "test")}
    corp = corpus_mod.build_corpus([q for n in splits for q in splits[n]], seed=man["seed"])
    assert corp.index_hash == idx["index_hash"]
    backend = SQLiteFTSBackend(str(ROOT / "data" / "index" / "corpus.sqlite"), match_all=False)
    tool = CorpusSearchTool(backend, top_k=TOP_K)
    search = MaskedSearch(backend)
    system = protocol.system_prompt(tool)

    out = ROOT / "data" / "processed" / "episodes"
    out.mkdir(parents=True, exist_ok=True)
    store_path = out / "trajectories.sqlite"
    store_path.unlink(missing_ok=True)
    store = TrajectoryStore(str(store_path))
    run_id = "scripted-oracle-episodes"
    store.register_run(run_id, "eval", "scripted-oracle", config={"top_k": TOP_K, "budget": BUDGET, "index_hash": corp.index_hash})

    stats: dict = {"index_hash": corp.index_hash, "top_k": TOP_K, "budget": BUDGET, "splits": {}}
    for split, qs in splits.items():
        built: list[builder.Built] = []
        for q in qs:
            built.extend(builder.build_question(q, split, corp, tool, search, BUDGET))
        qmap = {q.qid: q for q in qs}
        accepted = [b for b in built if b.episode is not None]

        logs, metas = [], []
        with (out / f"episodes_{split}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for b in accepted:
                ep = b.episode
                meta = {"parent_qid": b.qid, "split": split, "state": b.state, "variant": b.variant, "answerable": b.answerable,
                        "qtype": qmap[b.qid].qtype, "n_searches": ep.n_searches, "final": ep.final.turn.kind}
                logs.append(episodes.to_eventlog(ep))
                metas.append(meta)
                store.store(episodes.to_record(ep, qmap[b.qid], run_id, "scripted-oracle", {**meta, "split": split}))
                f.write(json.dumps({
                    "meta": meta, "user": ep.user_text,
                    "steps": [{"assistant": s.text, "observation": s.observation} for s in ep.steps],
                    "support_masked": sorted(ep.masked), "first_search_masked": sorted(ep.first_search_masked),
                    "gold_answer": qmap[b.qid].answer,
                }, ensure_ascii=False) + "\n")
        episodes.write_eventlogs(out / f"eventlogs_{split}.jsonl", logs)

        collector = trainer.RowCollector(system, metas)
        result = trainer.build_trainer(logs, collector).train()
        with (out / f"rows_{split}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for r in collector.rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        by_state = Counter(b.state for b in built)
        by_variant_state = Counter(f"{b.variant}:{b.state}" for b in built)
        rejects = Counter(b.reject_reason for b in built if b.reject_reason)
        stats["splits"][split] = {
            "questions": len(qs),
            "unreachable_open": by_state.get("unreachable", 0) and sum(1 for b in built if b.variant == "open" and b.state == "unreachable"),
            "accepted_episodes": len(accepted),
            "by_state": dict(sorted(by_state.items())),
            "by_variant_and_state": dict(sorted(by_variant_state.items())),
            "reject_reasons": dict(rejects),
            "rows": result["n_rows"],
            "rows_by_kind": result["by_kind"],
            "searches_per_episode": dict(sorted(Counter(m["n_searches"] for m in metas).items())),
        }
        print(split, json.dumps(stats["splits"][split]["by_state"]), "rows", result["n_rows"], result["by_kind"])

    store.export_jsonl(str(out / "trajectories.jsonl"), run_id=run_id)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "episode_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in stats["splits"]["train"].items() if k != "by_variant_and_state"}, indent=1))
    print("store:", store.stats())


if __name__ == "__main__":
    main()
