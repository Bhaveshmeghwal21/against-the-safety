"""Shared retrieval corpus: the union of all sampled HotpotQA paragraphs (plan section 4.3).

Paragraphs are deduplicated by title + normalized text hash. Chunk IDs are neutral
(`c00001`...) and assigned in a seeded shuffle, so ID order carries no split, support or
question information. One paragraph is one chunk.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

from .data import Question


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def _key(title: str, text: str) -> str:
    return hashlib.sha1(f"{_norm(title)}\x00{_norm(text)}".encode("utf-8")).hexdigest()


@dataclass
class Corpus:
    chunks: list[dict]
    by_question: dict[str, dict[str, str]]  # qid -> paragraph title -> chunk_id
    index_hash: str

    def support_chunks(self, q: Question) -> dict[str, str]:
        return {t: self.by_question[q.qid][t] for t in q.support_titles}


def build_corpus(questions: list[Question], seed: int) -> Corpus:
    unique: dict[str, tuple[str, str]] = {}
    for q in questions:
        for p in q.paragraphs:
            unique.setdefault(_key(p.title, p.text), (p.title, p.text))
    keys = sorted(unique)
    random.Random(seed).shuffle(keys)
    chunk_id = {k: f"c{i + 1:05d}" for i, k in enumerate(keys)}

    chunks = [
        {"chunk_id": chunk_id[k], "doc_id": chunk_id[k], "title": unique[k][0], "text": unique[k][1], "chunk_index": 0}
        for k in sorted(keys, key=lambda k: chunk_id[k])
    ]
    by_question = {q.qid: {p.title: chunk_id[_key(p.title, p.text)] for p in q.paragraphs} for q in questions}

    digest = hashlib.sha256()
    for c in chunks:
        digest.update(f"{c['chunk_id']}\t{c['title']}\t{c['text']}\n".encode("utf-8"))
    return Corpus(chunks=chunks, by_question=by_question, index_hash=digest.hexdigest())


def build_index(corpus: Corpus, db_path: Path):
    """Create a fresh SQLite FTS5 index with AgentTune's backend, OR-semantics BM25 (match_all=False)."""
    from agenttune.rag.retrieval.sqlite_fts import SQLiteFTSBackend

    db_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(str(db_path) + suffix).unlink(missing_ok=True)
    backend = SQLiteFTSBackend(str(db_path), match_all=False)
    backend.index(corpus.chunks)
    return backend
