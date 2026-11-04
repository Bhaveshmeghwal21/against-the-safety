"""Plan section 4.3 checks: backend edge cases, deterministic ranking, and masked-search behaviour."""

import json
from pathlib import Path

import pytest
from agenttune.rag.retrieval.sqlite_fts import SQLiteFTSBackend

from ragharness import corpus as corpus_mod, data
from ragharness.retrieval import MaskedSearch, render_observation

ROOT = Path(__file__).resolve().parents[1]

DOCS = [
    ("c1", "Aster Phone", "The Aster Phone has a battery covered for twelve months."),
    ("c2", "Aster Tablet", "The Aster Tablet has a screen covered for twenty-four months."),
    ("c3", "Warranty policy", "Warranty claims need a receipt. Batteries are covered for twelve months."),
    ("c4", "Shipping", "Orders ship within two days."),
    ("c5", "Aster Phone (duplicate)", "The Aster Phone has a battery covered for twelve months."),
]


@pytest.fixture()
def backend(tmp_path):
    b = SQLiteFTSBackend(str(tmp_path / "t.sqlite"), match_all=False)
    b.index([{"chunk_id": i, "doc_id": i, "title": t, "text": x, "chunk_index": 0} for i, t, x in DOCS])
    return b


def ids(results):
    return [r["metadata"]["chunk_id"] for r in results]


def test_paraphrase_finds_topic(backend):
    assert "c1" in ids(backend.search("how long is the phone battery covered", top_k=3))


def test_quoted_and_punctuated_queries_do_not_raise(backend):
    for q in ['"Aster Phone"', "Aster, Phone?!", "battery (twelve) months;", "***", "", "   ", "a' OR 1=1 --"]:
        assert isinstance(backend.search(q, top_k=3), list)


def test_no_result_query_is_empty(backend):
    assert backend.search("zzzxqv", top_k=3) == []


def test_duplicate_passages_both_returned(backend):
    got = ids(backend.search("Aster Phone battery covered twelve months", top_k=5))
    assert "c1" in got and "c5" in got


def test_ranking_is_deterministic(backend):
    first = ids(backend.search("battery covered twelve months", top_k=5))
    assert all(ids(backend.search("battery covered twelve months", top_k=5)) == first for _ in range(5))


def test_mask_removes_ids_and_keeps_order(backend):
    s = MaskedSearch(backend)
    plain = [h.chunk_id for h in s.search("battery covered twelve months", 4)]
    masked = frozenset({plain[0]})
    got = [h.chunk_id for h in s.search("battery covered twelve months", 3, masked)]
    assert plain[0] not in got
    assert got == [c for c in plain if c not in masked][:3]


def test_render_hides_scores():
    from ragharness.retrieval import Hit

    out = render_observation([Hit("c9", "T", "body text", 12.345)])
    assert "12.3" not in out and "score" not in out.lower() and out.startswith("[chunk_id=c9] T\nbody text")
    assert render_observation([]) == "No results found."


@pytest.mark.skipif(not (ROOT / "data" / "index" / "corpus.sqlite").exists(), reason="shared index not built")
def test_shared_index_masks_all_support_for_dev():
    man = json.loads((ROOT / "data" / "manifests" / "index_manifest.json").read_text(encoding="utf-8"))
    splits = {n: data.read_questions(ROOT / "data" / "processed" / f"questions_{n}.jsonl") for n in ("train", "dev", "test")}
    corp = corpus_mod.build_corpus([q for n in splits for q in splits[n]], seed=json.loads(
        (ROOT / "data" / "manifests" / "split_manifest.json").read_text(encoding="utf-8"))["seed"])
    assert corp.index_hash == man["index_hash"]
    s = MaskedSearch(SQLiteFTSBackend(str(ROOT / "data" / "index" / "corpus.sqlite"), match_all=False))
    for q in splits["dev"]:
        support = frozenset(corp.support_chunks(q).values())
        hits = s.search(q.question, 3, support)
        assert len(hits) == 3 and not support.intersection(h.chunk_id for h in hits)
