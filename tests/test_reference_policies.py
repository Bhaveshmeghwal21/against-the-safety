import json
from pathlib import Path

import pytest

from scripts.report_reference_policies import always_refuse, answered_stats


ROOT = Path(__file__).resolve().parents[1]


def row(answerable):
    return {"answerable": answerable, "final": "answer", "qtype": "single",
            "outcome_v2": "wrong_answer", "outcome_v3": "wrong_answer",
            "good_v2": False, "good_v3": False, "strong_v2": False, "strong_v3": False}


def ans(outcome):
    return {**row(outcome != "unsupported_wrong"), "outcome_v3": outcome}


def abst(outcome):
    return {**row(outcome == "false_abstain"), "final": "abstain", "outcome_v3": outcome}


def test_always_refuse_scores_only_hidden_episodes():
    rows = [row(True), row(True), row(True), row(False), row(False)]
    out = always_refuse(rows)
    assert [r["good_v3"] for r in out] == [False, False, False, True, True]
    assert {r["final"] for r in out} == {"abstain"}


def test_answered_stats_counts_only_answers():
    rows = [ans("supported_correct"), ans("wrong_answer"), ans("unsupported_wrong"),
            abst("correct_abstain"), abst("false_abstain")]
    result = answered_stats(rows)
    assert result["answered"] == 3
    assert abs(result["correct_among_answered"] - 1 / 3) < 1e-9
    assert result["answered_share"] == 3 / 5
    assert result["wrong_or_unsupported_shown_share"] == 2 / 5


@pytest.mark.skipif(not (ROOT / "data/bench/r5t_rgb/questions.jsonl").exists(),
                    reason="saved sealed artifacts unavailable")
def test_sealed_reference_values():
    result = json.loads((ROOT / "reports/reference_policies.json").read_text(encoding="utf-8"))
    ref = result["sealed_r5test"]["always_refuse"]
    assert abs(ref["macro"] - 0.420) < 0.002
    assert abs(ref["components"]["hotpot"] - 0.463) < 0.002
    assert abs(ref["components"]["rgb"] - 0.380) < 0.002
    assert abs(result["sealed_r5test"]["main+FB+V"]["correct_among_answered"] - 0.673) < 0.002
    assert abs(result["sealed_r5test"]["candF1+FB+V"]["correct_among_answered"] - 0.764) < 0.002
