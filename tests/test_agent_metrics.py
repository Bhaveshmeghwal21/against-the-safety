"""AuditKIT's tool-call reliability metrics (the Track 3 evaluation tool) must react to bad tool use."""

import json
from pathlib import Path

import pytest
from agenttune.rag.retrieval.sqlite_fts import SQLiteFTSBackend

from ragharness import corpus as corpus_mod, data, evaluation, protocol
from ragharness.retrieval import MaskedSearch
from ragharness.rollout import ReplayGenerator
from ragharness.tools import CorpusSearchTool

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "data" / "index" / "corpus.sqlite").exists(), reason="index not built")


@pytest.fixture(scope="module")
def env():
    man = json.loads((ROOT / "data" / "manifests" / "split_manifest.json").read_text(encoding="utf-8"))
    splits = {n: data.read_questions(ROOT / "data" / "processed" / f"questions_{n}.jsonl") for n in ("train", "dev", "test")}
    corp = corpus_mod.build_corpus([q for n in splits for q in splits[n]], seed=man["seed"])
    backend = SQLiteFTSBackend(str(ROOT / "data" / "index" / "corpus.sqlite"), match_all=False)
    tool, search = CorpusSearchTool(backend, top_k=3), MaskedSearch(backend)
    jobs = [j for j in evaluation.make_jobs(splits["dev"][:12], "dev", corp, tool, search, 0) if j.variant == "open"]
    return corp, tool, jobs


def _agent(env, make_texts):
    corp, tool, jobs = env
    res = evaluation.run_jobs(jobs, corp, tool, "sys", lambda job: ReplayGenerator(make_texts(job)), 3)
    return evaluation.score_results(res, corp, None, "t", "test", [tool.to_schema()])["summary"]["auditkit"]["open"]


def test_scripted_oracle_is_valid_and_loop_free(env):
    a = _agent(env, lambda job: job.scripted_texts)
    assert a["tool_call_validity"] == 1.0 and a["redundant_tool_calls"] == 0.0 and a["agent_loop_detection"] == 1.0


def test_repeated_search_is_redundant_and_loop(env):
    same = protocol.format_search("the same query again and again")
    a = _agent(env, lambda job: [same, same, same, protocol.format_abstain("no")])
    assert a["redundant_tool_calls"] > 0.0 and a["agent_loop_detection"] < 1.0


def test_wrong_tool_name_is_invalid(env):
    bad = json.dumps([{"tool_name": "web_search", "parameters": {"query": "x"}}])
    a = _agent(env, lambda job: [bad])
    assert a["tool_call_validity"] == 0.0
