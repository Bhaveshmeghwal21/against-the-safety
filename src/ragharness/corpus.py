"""Shared retrieval corpus: the union of all sampled HotpotQA paragraphs (plan section 4.3).

Paragraphs are deduplicated by title + normalized text hash. Chunk IDs are neutral
(`c00001`...) and assigned in a seeded shuffle, so ID order carries no split, support or
question information. One paragraph is one chunk.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
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
    # Benchmarks only (plan section 22): qid -> other chunks that each hold the answer on their own (RGB's extra positive
    # passages). They are masked with the support and count as gold citations. Empty for HotpotQA.
    extra_copies: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self):
        self._by_id = {c["chunk_id"]: c for c in self.chunks}
        self._by_title: dict[str, list[str]] = {}
        for c in self.chunks:
            self._by_title.setdefault(c["title"].casefold(), []).append(c["chunk_id"])

    def chunk(self, chunk_id: str) -> dict:
        return self._by_id[chunk_id]

    def support_chunks(self, q: Question) -> dict[str, str]:
        return {t: self.by_question[q.qid][t] for t in q.support_titles}

    def support_copies(self, q: Question) -> frozenset[str]:
        """Every chunk that could stand in for a gold support paragraph: the chunk itself plus any other chunk
        carrying the same title (e.g. a differently worded copy of the same page)."""
        out: set[str] = set()
        for title, cid in self.support_chunks(q).items():
            out.update(self.title_copies(title, cid))
        out.update(self.extra_copies.get(q.qid, ()))
        return frozenset(out)

    def title_copies(self, title: str, chunk_id: str) -> frozenset[str]:
        return frozenset({chunk_id, *self._by_title.get(title.casefold(), [])})


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


def extend_corpus(base: Corpus, questions: list[Question], seed: int = 20261105) -> Corpus:
    """The base corpus plus the paragraphs of new questions, without renumbering anything (plan section 20).

    Paragraphs already in the base keep their chunk ID; new ones get IDs after the last base ID, in a seeded shuffle among
    themselves. Every training row, citation and index hash built on the base therefore stays valid.
    """
    known = {_key(c["title"], c["text"]): c["chunk_id"] for c in base.chunks}
    new: dict[str, tuple[str, str]] = {}
    for q in questions:
        for p in q.paragraphs:
            k = _key(p.title, p.text)
            if k not in known:
                new.setdefault(k, (p.title, p.text))
    keys = sorted(new)
    random.Random(seed).shuffle(keys)
    start = max(int(c["chunk_id"][1:]) for c in base.chunks)
    assert start + len(keys) <= 99999, "chunk IDs are five digits"
    ids = dict(known)
    ids.update({k: f"c{start + i + 1:05d}" for i, k in enumerate(keys)})
    chunks = list(base.chunks) + [
        {"chunk_id": ids[k], "doc_id": ids[k], "title": new[k][0], "text": new[k][1], "chunk_index": 0} for k in keys
    ]
    by_question = dict(base.by_question)
    by_question.update({q.qid: {p.title: ids[_key(p.title, p.text)] for p in q.paragraphs} for q in questions})
    digest = hashlib.sha256()
    for c in chunks:
        digest.update(f"{c['chunk_id']}\t{c['title']}\t{c['text']}\n".encode("utf-8"))
    return Corpus(chunks=chunks, by_question=by_question, index_hash=digest.hexdigest())


def corpus_hash(chunks: list[dict]) -> str:
    digest = hashlib.sha256()
    for c in chunks:
        digest.update(f"{c['chunk_id']}\t{c['title']}\t{c['text']}\n".encode("utf-8"))
    return digest.hexdigest()


def save_corpus(path: Path, corpus: Corpus) -> None:
    """Benchmarks (plan section 22) ship their corpus as one file instead of rebuilding it from source questions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {"index_hash": corpus.index_hash, "chunks": corpus.chunks, "by_question": corpus.by_question, "extra_copies": corpus.extra_copies}
    path.write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8", newline="\n")


def load_corpus(path: Path) -> Corpus:
    blob = json.loads(path.read_text(encoding="utf-8"))
    if corpus_hash(blob["chunks"]) != blob["index_hash"]:
        raise RuntimeError(f"{path}: chunks do not match the recorded index hash")
    return Corpus(chunks=blob["chunks"], by_question=blob["by_question"], index_hash=blob["index_hash"], extra_copies=blob.get("extra_copies", {}))
