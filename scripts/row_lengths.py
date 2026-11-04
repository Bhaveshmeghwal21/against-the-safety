"""Rendered-length budget for Tiny Aya rows on DEV (characters only; token counts need the gated tokenizer).

Run (after `. scripts/env.ps1`, prepare_data.py and build_index.py):  python scripts/row_lengths.py
"""

import json
from collections import defaultdict
from pathlib import Path

from agenttune.rag.retrieval.sqlite_fts import SQLiteFTSBackend

from ragharness import corpus as corpus_mod, data, episodes, protocol, rows, trainer
from ragharness.tools import CorpusSearchTool

ROOT = Path(__file__).resolve().parents[1]
CHARS_PER_TOKEN = 4.0  # rough English estimate; replace with the real tokenizer count


def main() -> None:
    man = json.loads((ROOT / "data" / "manifests" / "split_manifest.json").read_text(encoding="utf-8"))
    splits = {n: data.read_questions(ROOT / "data" / "processed" / f"questions_{n}.jsonl") for n in ("train", "dev", "test")}
    corp = corpus_mod.build_corpus([q for n in splits for q in splits[n]], seed=man["seed"])
    tool = CorpusSearchTool(SQLiteFTSBackend(str(ROOT / "data" / "index" / "corpus.sqlite"), match_all=False), top_k=3)
    system = protocol.system_prompt(tool)

    empty = rows.render([{"role": "system", "content": system}, {"role": "user", "content": ""}], True)
    fixed = len(empty) - len(rows.PROMPT_END)
    no_system = rows.render([{"role": "user", "content": ""}], True)
    print(f"fixed prefix chars: {fixed}  (Tiny Aya default preamble {len(no_system) - len(rows.PROMPT_END)}, our system prompt {len(system)})")

    collector = trainer.RowCollector(system)
    trainer.build_trainer([episodes.to_eventlog(episodes.run_scripted(q, corp, tool, "open")) for q in splits["dev"]], collector).train()

    by_search = defaultdict(list)
    for d, c in zip(collector.rows, collector.checked):
        n_obs = sum(m["content"].startswith("Tool result") for m in d["prompt"] if m["role"] == "user")
        by_search[n_obs].append(len(c["prompt_text"]) + len(c["completion_text"]))
    print("rows by number of observations already in the prompt (chars: mean / max, est. tokens at 4 chars/token):")
    for k in sorted(by_search):
        v = by_search[k]
        print(f"  {k} observations: n={len(v):3d}  mean {sum(v)/len(v):6.0f}  max {max(v):6d}  ~{max(v)/CHARS_PER_TOKEN:5.0f} tokens max")
    over = sum(max(v) / CHARS_PER_TOKEN > 2048 for v in by_search.values())
    n_over = sum(1 for v in by_search.values() for x in v if x / CHARS_PER_TOKEN > 2048)
    print(f"rows over 2048 est. tokens: {n_over} of {sum(len(v) for v in by_search.values())}")


if __name__ == "__main__":
    main()
