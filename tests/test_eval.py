"""The evaluation pipeline must separate trivial policies, and AuditKIT's numbers must agree with ours."""

import json
from pathlib import Path

import pytest
from agenttune.rag.retrieval.sqlite_fts import SQLiteFTSBackend

from ragharness import corpus as corpus_mod, data, evaluation, protocol
from ragharness.metrics import paired_bootstrap
from ragharness.retrieval import MaskedSearch
from ragharness.tools import CorpusSearchTool

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "data" / "index" / "corpus.sqlite").exists(), reason="index not built")


@pytest.fixture(scope="module")
def jobs_env():
    man = json.loads((ROOT / "data" / "manifests" / "split_manifest.json").read_text(encoding="utf-8"))
    splits = {n: data.read_questions(ROOT / "data" / "processed" / f"questions_{n}.jsonl") for n in ("train", "dev", "test")}
    corp = corpus_mod.build_corpus([q for n in splits for q in splits[n]], seed=man["seed"])
    backend = SQLiteFTSBackend(str(ROOT / "data" / "index" / "corpus.sqlite"), match_all=False)
    tool, search = CorpusSearchTool(backend, top_k=3), MaskedSearch(backend)
    jobs = evaluation.make_jobs(splits["dev"][:30], "dev", corp, tool, search, extra_questions=0)
    return corp, tool, jobs


def _score(jobs_env, generator):
    corp, tool, jobs = jobs_env
    res = evaluation.run_jobs(jobs, corp, tool, "sys", lambda job: generator(job), 3)
    return evaluation.score_results(res, corp, None, "t", "test")["summary"]


def test_always_abstain(jobs_env):
    s = _score(jobs_env, lambda job: (lambda m: protocol.format_abstain("nothing found")))
    h = s["headline"]
    assert h["correct_abstain"] == 1.0 and h["false_abstain"] == 1.0 and h["supported_correct"] == 0.0 and h["invalid"] == 0.0
    assert h["end_to_end_good"] == round(h["n_unanswerable"] / h["n"], 3)
    assert s["auditkit"]["support_masked"]["abstention"] == 1.0 and s["auditkit"]["open"]["abstention"] == 0.0


def test_garbage_is_invalid_and_never_scores_as_abstention(jobs_env):
    s = _score(jobs_env, lambda job: (lambda m: "I am not sure."))
    h = s["headline"]
    assert h["invalid"] == 1.0 and h["correct_abstain"] == 0.0 and h["end_to_end_good"] == 0.0


def test_scripted_oracle_matches_auditkit_recall(jobs_env):
    from ragharness.rollout import ReplayGenerator

    s = _score(jobs_env, lambda job: ReplayGenerator(job.scripted_texts))
    assert s["headline"]["end_to_end_good"] == 1.0
    assert s["auditkit"]["open"]["evidence_set_recall"] == 1.0 and s["auditkit"]["support_masked"]["abstention"] == 1.0


def test_paired_bootstrap_detects_a_difference():
    a = [False] * 60 + [True] * 40
    b = [True] * 90 + [False] * 10
    p = paired_bootstrap(a, b, reps=2000)
    assert p.diff > 0.3 and p.ci_low > 0.15 and p.a_only >= 0 and p.b_only > 0
