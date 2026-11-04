"""AgentTune's `search_corpus` tool with per-episode masks and neutral output.

Subclasses `agenttune.rag.tools.search_corpus.SearchCorpusTool` so the tool name, description and schema are
AgentTune's. Two changes, both required by the plan (section 4.1/4.3):
- the stock tool prints `score=` and truncates each passage at 350 characters; this one prints neither;
- results go through the masked-search wrapper so missing-support episodes can hide chosen chunks.
The `[chunk_id=...]` marker is kept because AgentTune's rollout collects `retrieved_chunk_ids` from it.

Masks: `masked` applies to every search of the episode; `first_search_masked` applies to the first search only
(the "weak first results, support still reachable" look-alike).
"""

from __future__ import annotations

from agenttune.agentic.tools.base import ToolResult
from agenttune.rag.tools.search_corpus import SearchCorpusTool

from .retrieval import Hit, MaskedSearch, render_observation


class CorpusSearchTool(SearchCorpusTool):
    def __init__(self, backend, top_k: int = 3, max_chars: int | None = 800):
        super().__init__(backend, top_k=top_k)
        self._masked_search = MaskedSearch(backend)
        self.max_chars = max_chars
        self.masked: frozenset[str] = frozenset()
        self.first_search_masked: frozenset[str] = frozenset()
        self.shown: set[str] = set()
        self.n_calls = 0
        self.last_hits: list[Hit] = []

    def start_episode(
        self, masked: frozenset[str] = frozenset(), first_search_masked: frozenset[str] = frozenset(), shown: frozenset[str] = frozenset()
    ) -> None:
        """`shown` lists chunk IDs already in the conversation (pre-supplied evidence)."""
        self.masked = frozenset(masked)
        self.first_search_masked = frozenset(first_search_masked)
        self.shown = set(shown)
        self.n_calls = 0
        self.last_hits = []

    def execute(self, query: str) -> ToolResult:
        mask = self.masked | (self.first_search_masked if self.n_calls == 0 else frozenset())
        self.n_calls += 1
        try:
            hits = self._masked_search.search(query, self.top_k, mask)
        except Exception as e:  # same contract as the stock tool: failures are results, not crashes
            self.last_hits = []
            return ToolResult(success=False, output=None, error=str(e))
        self.last_hits = hits
        output = render_observation(hits, frozenset(self.shown), self.max_chars)
        self.shown.update(h.chunk_id for h in hits)
        return ToolResult(
            success=True,
            output=output,
            metadata={"result_count": len(hits), "chunk_ids": [h.chunk_id for h in hits], "scores": [h.score for h in hits]},
        )
