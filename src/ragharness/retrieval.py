"""Masked search over the shared index, plus the neutral rendering the model sees.

Per-episode masks (plan section 4.3): over-fetch `top_k + len(masked)` results, drop masked
chunk IDs, return the first `top_k`. Masking never changes the ranking of the other chunks.
BM25 scores are kept on `Hit.score` for logging but are never rendered to the model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    title: str
    text: str
    score: float  # higher is better; logged only


class MaskedSearch:
    def __init__(self, backend):
        self.backend = backend

    def search(self, query: str, top_k: int, masked: frozenset[str] = frozenset()) -> list[Hit]:
        raw = self.backend.search(query, top_k=top_k + len(masked))
        hits = [
            Hit(r["metadata"]["chunk_id"], r["source"], r["content"], float(r["score"]))
            for r in raw
            if r["metadata"]["chunk_id"] not in masked
        ]
        return hits[:top_k]


def render_observation(hits: list[Hit]) -> str:
    """Neutral tool result: chunk ID, title and passage text. No scores or ranks.

    The `[chunk_id=...]` marker matches the pattern AgentTune's rollout parses to collect `retrieved_chunk_ids`.
    """
    if not hits:
        return "No results found."
    return "\n\n".join(f"[chunk_id={h.chunk_id}] {h.title}\n{h.text}" for h in hits)
