import json
from pathlib import Path

import pytest

from ragharness import data, lineage


ROOT = Path(__file__).resolve().parents[1]
NEEDS_BENCH = pytest.mark.skipif(
    not (ROOT / "data/bench/r6t_squad/questions.jsonl").exists(),
    reason="round-6 benchmark data not built",
)


def ids(name):
    return {q.qid for q in data.read_questions(ROOT / "data/bench" / name / "questions.jsonl")}


@pytest.mark.skipif(not (ROOT / "reports/round6_allocation.json").exists(),
                    reason="round-6 allocation not built")
def test_r6_articles_are_unused():
    allocation = json.loads((ROOT / "reports/round6_allocation.json").read_text(encoding="utf-8"))
    prior = json.loads((ROOT / "reports/round5_allocation.json").read_text(encoding="utf-8"))["allocation"]
    assert len(allocation["r6test"]) == 25
    assert set(allocation["r6test"]) <= set(prior["reserve"])
    assert not set(allocation["r6test"]) & (set(prior["r5train"]) | set(prior["r5test"]))


@NEEDS_BENCH
def test_r6_questions_are_new():
    old = set().union(*(ids(n) for n in ("r3dev", "r4dev", "r4test", "r5train", "r5t_squad",
                                         "r5t_hotpot", "r3t_hotpot", "rgb", "r3t_rgb", "r5t_rgb")))
    assert not (ids("r6t_squad") | ids("r6t_hotpot") | ids("r6t_rgb")) & old


@NEEDS_BENCH
def test_r6_rgb_takes_what_is_left():
    manifest = json.loads((ROOT / "data/bench/r6t_rgb/manifest.json").read_text(encoding="utf-8"))
    assert 1 <= manifest["n_questions"] <= 40


@NEEDS_BENCH
def test_holdouts_include_r6():
    guard = lineage.PromptGuard(lineage.load_holdouts(ROOT))
    question = data.read_questions(ROOT / "data/bench/r6t_hotpot/questions.jsonl")[0]
    assert guard.matches([{"role": "user", "content": question.question}])
