"""AgentTune's `search_corpus` tool with per-episode masks and neutral output.

Subclasses `agenttune.rag.tools.search_corpus.SearchCorpusTool` so the tool name, description and schema are
AgentTune's. Two changes, both required by the plan (section 4.1/4.3):
- the stock tool prints `score=` and truncates each passage at 350 characters; this one prints neither;
- results go through the masked-search wrapper so missing-support episodes can hide chosen chunks.
The `[chunk_id=...]` marker is kept because AgentTune's rollout collects `retrieved_chunk_ids` from it.
"""

from __future__ import annotations

from agenttune.agentic.tools.base import ToolResult
from agenttune.rag.tools.search_corpus import SearchCorpusTool

from .retrieval import Hit, MaskedSearch, render_observation


class CorpusSearchTool(SearchCorpusTool):
    def __init__(self, backend, top_k: int = 3):
        super().__init__(backend, top_k=top_k)
        self._masked_search = MaskedSearch(backend)
        self.masked: frozenset[str] = frozenset()
        self.last_hits: list[Hit] = []

    def start_episode(self, masked: frozenset[str] = frozenset()) -> None:
        self.masked = frozenset(masked)
        self.last_hits = []

    def execute(self, query: str) -> ToolResult:
        try:
            hits = self._masked_search.search(query, self.top_k, self.masked)
        except Exception as e:  # same contract as the stock tool: failures are results, not crashes
            self.last_hits = []
            return ToolResult(success=False, output=None, error=str(e))
        self.last_hits = hits
        return ToolResult(
            success=True,
            output=render_observation(hits),
            metadata={"result_count": len(hits), "chunk_ids": [h.chunk_id for h in hits], "scores": [h.score for h in hits]},
        )
