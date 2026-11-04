"""AgentTune EventLogs -> create_distill_trainer -> Tiny Aya prompt/completion rows, checked on CPU.

Uses scripted open-condition DEV episodes (no model, no training happens). Verifies every row against the pinned
Tiny Aya chat template, and reports row counts and rendered lengths.

Run (after `. scripts/env.ps1`, prepare_data.py and build_index.py):  python scripts/check_rows.py
"""

import json
import statistics
from pathlib import Path

from agenttune.rag.retrieval.sqlite_fts import SQLiteFTSBackend

from ragharness import corpus as corpus_mod, data, episodes, protocol, trainer
from ragharness.tools import CorpusSearchTool

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    man = json.loads((ROOT / "data" / "manifests" / "split_manifest.json").read_text(encoding="utf-8"))
    splits = {n: data.read_questions(ROOT / "data" / "processed" / f"questions_{n}.jsonl") for n in ("train", "dev", "test")}
    corp = corpus_mod.build_corpus([q for n in splits for q in splits[n]], seed=man["seed"])
    tool = CorpusSearchTool(SQLiteFTSBackend(str(ROOT / "data" / "index" / "corpus.sqlite"), match_all=False), top_k=3)
    system = protocol.system_prompt(tool)

    logs = [episodes.to_eventlog(episodes.run_scripted(q, corp, tool, "open")) for q in splits["dev"]]
    collector = trainer.RowCollector(system)
    t = trainer.build_trainer(logs, collector)
    preview = t.sft_dataset()  # AgentTune's GPU-free preview of the rows it will hand to the factory
    assert len(preview) == len(logs)
    result = t.train()  # Project.distill -> RowCollector

    chars = [len(c["prompt_text"]) + len(c["completion_text"]) for c in collector.checked]
    comp_chars = [len(c["completion_text"]) for c in collector.checked]
    print("student:", collector.student)
    print("AgentTune episodes:", len(preview), " prompt/completion rows:", result["n_rows"], " by kind:", result["by_kind"])
    print(f"rendered chars per row: mean {statistics.mean(chars):.0f}  max {max(chars)}  | completion chars mean {statistics.mean(comp_chars):.0f}")
    ex = next(c for c in collector.checked if c["kind"] == "answer")
    print("\n--- example answer row (completion only) ---")
    print(ex["completion_text"])
    ex = next(c for c in collector.checked if c["kind"] == "search")
    print("--- example search completion ---")
    print(ex["completion_text"])
    print("--- system prompt sent in the Developer Preamble ---")
    print(system)


if __name__ == "__main__":
    main()
