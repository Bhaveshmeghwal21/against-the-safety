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


def open_verified_backend(db_path, corp):
    """Open the shared SQLite FTS index only if it holds exactly the corpus' chunks.

    AgentTune's backend turns a missing or empty database into "no results" instead of an error, so a bad copy of the
    index would silently make every question unreachable. This reads the index read-only, recomputes the corpus hash from
    the stored rows, and runs one probe search before returning the backend.
    """
    import hashlib
    import sqlite3
    import sys
    from pathlib import Path

    from agenttune.rag.retrieval.sqlite_fts import SQLiteFTSBackend

    p = Path(db_path)
    if not p.is_file() or p.stat().st_size == 0:
        raise RuntimeError(f"index missing or empty: {p} ({p.stat().st_size if p.exists() else 'absent'} bytes)")
    con = sqlite3.connect(f"file:{p.resolve().as_posix()}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT chunk_id, title, text FROM chunks ORDER BY chunk_id").fetchall()
    finally:
        con.close()
    digest = hashlib.sha256()
    for chunk_id, title, text in rows:
        digest.update(f"{chunk_id}\t{title}\t{text}\n".encode("utf-8"))
    if len(rows) != len(corp.chunks) or digest.hexdigest() != corp.index_hash:
        raise RuntimeError(f"index {p} does not match the corpus: {len(rows)} rows vs {len(corp.chunks)} chunks, hash {digest.hexdigest()[:12]} vs {corp.index_hash[:12]}")
    backend = SQLiteFTSBackend(str(p), match_all=False)
    probe = corp.chunks[0]
    if not any(r["metadata"]["chunk_id"] == probe["chunk_id"] for r in backend.search(probe["title"], top_k=10)):
        # AgentTune's backend reports every sqlite error as "no results"; run the query directly to see the real cause.
        con = sqlite3.connect(f"file:{p.resolve().as_posix()}?mode=ro", uri=True)
        try:
            options = [r[0] for r in con.execute("PRAGMA compile_options").fetchall()]
            try:
                con.execute("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH ?", ('"' + probe["title"].split()[0] + '"',)).fetchone()
                cause = "the raw FTS query works, so the backend's own query failed"
            except sqlite3.Error as e:
                cause = f"{type(e).__name__}: {e}"
        finally:
            con.close()
        raise RuntimeError(
            f"index {p} holds the chunks but full-text search returns nothing for {probe['title']!r}; {cause}; "
            f"sqlite {sqlite3.sqlite_version}, python {sys.version.split()[0]}, FTS5 compiled: {'ENABLE_FTS5' in options}"
        )
    return backend


def render_observation(hits: list[Hit], shown: frozenset[str] = frozenset(), max_chars: int | None = None) -> str:
    """Neutral tool result: chunk ID, title and passage text. No scores or ranks.

    `shown` holds chunk IDs already printed earlier in the episode; they appear as marker + title only, so the same
    passage is never paid for twice. `max_chars` is one uniform limit for every passage (support or not).
    The `[chunk_id=...]` marker matches the pattern AgentTune's rollout parses to collect `retrieved_chunk_ids`.
    """
    if not hits:
        return "No results found."
    parts = []
    for h in hits:
        if h.chunk_id in shown:
            parts.append(f"[chunk_id={h.chunk_id}] {h.title} (shown above)")
        else:
            parts.append(f"[chunk_id={h.chunk_id}] {h.title}\n{visible_text(h.text, max_chars)}")
    return "\n\n".join(parts)


def visible_text(text: str, max_chars: int | None) -> str:
    """The passage as the model sees it: cut at `max_chars` on a word boundary and marked with an ellipsis."""
    if max_chars is None or len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + " ..."
