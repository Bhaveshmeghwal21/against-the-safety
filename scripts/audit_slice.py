"""End-to-end CPU slice on DEV: AgentTune tool -> AgentTune trajectories -> AuditKIT RAG-stress metrics.

Two conditions per dev question with the scripted oracle policy (not a model):
  open            no masks; the oracle answers when it found the full support set, otherwise abstains
  support_masked  every gold support chunk is masked; the correct behaviour is to abstain
The open-condition evidence_set_recall must equal the independent ceiling number from retrieval_ceiling.py.

Run (after `. scripts/env.ps1`, prepare_data.py and build_index.py):  python scripts/audit_slice.py
"""

import json
from collections import defaultdict
from pathlib import Path

from agenttune.rag.retrieval.sqlite_fts import SQLiteFTSBackend
from agenttune.rag.storage.trajectory_store import TrajectoryStore
from auditkit.agent_eval import episodes_from_agenttune

from ragharness import audit, corpus as corpus_mod, data, episodes
from ragharness.tools import CorpusSearchTool

ROOT = Path(__file__).resolve().parents[1]
TOP_K, BUDGET = 3, 3


def main() -> None:
    man = json.loads((ROOT / "data" / "manifests" / "split_manifest.json").read_text(encoding="utf-8"))
    idx = json.loads((ROOT / "data" / "manifests" / "index_manifest.json").read_text(encoding="utf-8"))
    splits = {n: data.read_questions(ROOT / "data" / "processed" / f"questions_{n}.jsonl") for n in ("train", "dev", "test")}
    corp = corpus_mod.build_corpus([q for n in splits for q in splits[n]], seed=man["seed"])
    assert corp.index_hash == idx["index_hash"]
    chunks = audit.ChunkLookup(corp)
    tool = CorpusSearchTool(SQLiteFTSBackend(str(ROOT / "data" / "index" / "corpus.sqlite"), match_all=False), top_k=TOP_K)

    out = ROOT / "runs" / "slice_dev"
    out.mkdir(parents=True, exist_ok=True)
    store_path = out / "trajectories.sqlite"
    store_path.unlink(missing_ok=True)
    store = TrajectoryStore(str(store_path))
    run_id = "scripted-oracle-dev"
    store.register_run(run_id, "eval", "scripted-oracle", config={"top_k": TOP_K, "budget": BUDGET, "index_hash": corp.index_hash})

    eps, samples, logs = [], defaultdict(list), []
    for q in splits["dev"]:
        support = frozenset(corp.support_chunks(q).values())
        for cond, masked in (("open", frozenset()), ("support_masked", support)):
            ep = episodes.run_scripted(q, corp, tool, cond, masked, BUDGET)
            eps.append((q, ep))
            logs.append(episodes.to_eventlog(ep))
            store.store(episodes.to_record(ep, q, run_id, "scripted-oracle"))
            samples[cond].append(audit.to_sample(ep, q, corp, chunks, answerable=(cond == "open")))

    episodes.write_eventlogs(out / "eventlogs.jsonl", logs)
    store.export_jsonl(str(out / "trajectories.jsonl"), run_id=run_id)

    # AuditKIT reads AgentTune's own EventLog file with its importer.
    imported = episodes_from_agenttune(str(out / "eventlogs.jsonl"))
    assert len(imported) == len(eps), (len(imported), len(eps))
    n_calls = [sum(1 for e in ep.events if e.type == "tool_call") for ep in imported]
    assert n_calls == [ep.n_searches for _, ep in eps], "AuditKIT saw a different number of searches than we logged"

    report = {"index_hash": corp.index_hash, "split": "dev", "top_k": TOP_K, "budget": BUDGET, "policy": "scripted oracle (not a model)", "conditions": {}}
    for cond, ss in samples.items():
        res = audit.evaluate(ss, f"slice-{cond}")
        row = {"n": len(ss), "auditkit": {}, "unscored": res.unscored}
        for name in audit.SCORERS:
            s = res.stats.get(name)
            row["auditkit"][name] = None if s is None else round(s.mean, 3)
        mine = [audit.answer_scores(ep, q) for q, ep in eps if ep.condition == cond]
        row["answered"] = sum(m["answered"] for m in mine)
        row["abstained"] = sum(m["abstained"] for m in mine)
        row["invalid"] = sum(m["invalid"] for m in mine)
        f1s = [m["answer_f1"] for m in mine if m["answer_f1"] is not None]
        row["answer_f1_when_answered"] = round(sum(f1s) / len(f1s), 3) if f1s else None
        row["cited_not_observed"] = sum(m["cited_not_observed"] or 0 for m in mine)
        row["mean_searches"] = round(sum(m["n_searches"] for m in mine) / len(mine), 2)
        report["conditions"][cond] = row

    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "audit_slice_dev.json").write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))
    print("trajectory store:", store.stats())


if __name__ == "__main__":
    main()
