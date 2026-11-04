"""The rollout loop must reproduce the scripted training episodes and label bad turns explicitly."""

import json
from pathlib import Path

import pytest
from agenttune.rag.retrieval.sqlite_fts import SQLiteFTSBackend

from ragharness import builder, corpus as corpus_mod, data, protocol
from ragharness.retrieval import MaskedSearch
from ragharness.rollout import ReplayGenerator, rollout
from ragharness.tools import CorpusSearchTool

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "data" / "index" / "corpus.sqlite").exists(), reason="index not built")


@pytest.fixture(scope="module")
def env():
    man = json.loads((ROOT / "data" / "manifests" / "split_manifest.json").read_text(encoding="utf-8"))
    splits = {n: data.read_questions(ROOT / "data" / "processed" / f"questions_{n}.jsonl") for n in ("train", "dev", "test")}
    corp = corpus_mod.build_corpus([q for n in splits for q in splits[n]], seed=man["seed"])
    backend = SQLiteFTSBackend(str(ROOT / "data" / "index" / "corpus.sqlite"), match_all=False)
    return splits, corp, CorpusSearchTool(backend, top_k=3), MaskedSearch(backend)


def test_replaying_scripted_episodes_reproduces_them(env):
    splits, corp, tool, search = env
    checked = 0
    for q in splits["dev"][:25]:
        for b in builder.build_question(q, "dev", corp, tool, search):
            if b.episode is None:
                continue
            ep = b.episode
            spec = builder.make_specs(q, corp, search)[b.variant]
            replay = rollout(q, corp, tool, ReplayGenerator([s.text for s in ep.steps]), "sys", b.variant,
                             spec.masked, spec.first_search_masked, spec.prefill)
            assert [(s.text, s.observation) for s in replay.steps] == [(s.text, s.observation) for s in ep.steps]
            assert replay.complete == ep.complete and replay.user_text == ep.user_text
            checked += 1
    assert checked > 50


def _run(env, texts):
    splits, corp, tool, _ = env
    return rollout(splits["dev"][0], corp, tool, ReplayGenerator(texts), "sys", "open")


def test_garbage_is_invalid_not_abstain(env):
    ep = _run(env, ["I think the answer is 42."])
    assert ep.final.turn.kind == "invalid" and ep.stop_reason.startswith("invalid")


def test_repeated_query_ends_episode(env):
    s = protocol.format_search("some query")
    ep = _run(env, [s, s])
    assert ep.final.turn.kind == "invalid" and ep.stop_reason == "repeated_query" and ep.n_searches == 1


def test_search_past_budget_ends_episode(env):
    texts = [protocol.format_search(f"query number {i}") for i in range(4)]
    ep = _run(env, texts)
    assert ep.final.turn.kind == "invalid" and ep.stop_reason == "budget_exceeded" and ep.n_searches == 3
