import json
from pathlib import Path

import pytest
import torch

from scripts.build_verifier_train import as_prompt, filter_cases, swap_answer
from scripts.train_verifier import two_way_loss


ROOT = Path(__file__).resolve().parents[1]


def test_two_way_loss_matches_cross_entropy():
    logits = torch.zeros(2, 10)
    logits[0, 3] = 5.0
    logits[1, 7] = 5.0
    assert two_way_loss(logits, 3, 7, torch.tensor([1, 0])).item() < 0.01
    assert two_way_loss(logits, 3, 7, torch.tensor([0, 1])).item() > 4.9


def test_wording_is_shared():
    from scripts import train_verifier
    from ragharness import checker
    assert train_verifier.WORDING is checker.WORDING_A
    assert train_verifier.POSITIVE == checker.POSITIVE
    assert train_verifier.NEGATIVE == checker.NEGATIVE


def test_swap_needs_a_span_of_the_same_kind():
    assert swap_answer("1754", "It began in 1756 and ended in 1763.") in {"1756", "1763"}
    assert swap_answer("1754", "No digits here.") is None


def test_yes_no_cases_are_kept_and_flagged_cases_dropped():
    cases = [{"answer": "yes", "deterministic": {"cites_unshown": False, "answer_part_absent_from_cited": False}},
             {"answer": "Paris", "deterministic": {"cites_unshown": True, "answer_part_absent_from_cited": False}}]
    assert [c["answer"] for c in filter_cases(cases)] == ["yes"]


@pytest.mark.skipif(not (ROOT / "data/processed/verifier/train_v2.jsonl").exists(),
                    reason="round-6 verifier training data not built")
def test_built_set_is_balanced_and_guarded():
    from ragharness import lineage
    read = lambda path: [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows = read(ROOT / "data/processed/verifier/train_v2.jsonl")
    assert 0.45 <= sum(r["label"] for r in rows) / len(rows) <= 0.55
    assert len(rows) <= 2000
    guard = lineage.PromptGuard(lineage.load_holdouts(ROOT))
    assert not any(guard.matches(as_prompt(r)) for r in rows)
    validation = read(ROOT / "data/processed/verifier/val_v2.jsonl")
    assert not {r["qid"] for r in validation} & {r["qid"] for r in rows}
    labels = {}
    for row in rows + validation:
        key = (row["question"], row["answer"], row["passages"])
        assert key not in labels or labels[key] == row["label"]
        labels[key] = row["label"]
