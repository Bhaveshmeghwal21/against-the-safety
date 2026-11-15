"""The second-look test (research/2026-11-15-round6-second-look-preregistration.md): the adoption rule and the replacement of refusals."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import report_round6_second_look as R  # noqa: E402


def comparison(macro_ci, hidden_ci):
    return {"macro": {"difference": sum(macro_ci) / 2, "ci95": list(macro_ci)},
            "unsupported_rate": {"difference": sum(hidden_ci) / 2, "ci95": list(hidden_ci)}}


def test_adoption_needs_macro_interval_above_zero_and_a_small_hidden_rise():
    assert R.decision(comparison((0.002, 0.03), (-0.01, 0.02)))["adopt"]
    assert not R.decision(comparison((-0.001, 0.03), (-0.01, 0.0)))["adopt"]  # the macro interval touches zero
    assert not R.decision(comparison((0.01, 0.03), (0.0, 0.021)))["adopt"]  # the hidden-evidence rate may rise by at most 2 points
    assert R.decision(comparison((0.01, 0.03), (0.0, 0.02)))["adopt"]


def test_registered_values():
    assert R.TAU_SECOND_LOOK == 0.90
    assert R.TAU_FIRST_CHECK == 0.50
    assert R.MAX_HIDDEN_RATE_RISE == 0.02
    assert (R.DRAWS, R.SEED) == (3000, 47)
    assert R.HEADLINE == "main+FB+V+second look" and R.REFERENCE == "main+FB+V"


def row(qid, final, part="r6t_squad"):
    return {"qid": qid, "variant": "open", "part": part, "comp": "single", "final": final, "answerable": True,
            "outcome_v3": "false_abstain", "good_v3": False, "strong_v3": False, "excluded_v3": False}


def entry(qid, p, outcome="supported_correct", part="r6t_squad"):
    score = {"outcome_v3": outcome, "good_v3": True, "strong_v3": True, "excluded_v3": False, "final": "answer", "answerable": True, "qid": qid, "variant": "open"}
    return {"key": (f"main_{R.TEST_SUFFIXES[0]}", qid, "open"), "p": p, "score": score, "valid": True,
            "truth": {"answerable": True, "excluded_v3": False}}


def test_second_look_replaces_only_refusals_at_or_above_the_threshold():
    rows = [row("a", "abstain"), row("b", "abstain"), row("c", "answer")]
    entries = [entry("a", 0.95), entry("b", 0.89), entry("c", 0.99)]
    out = R.apply_second_look(rows, entries, 0.90, "main", R.TEST_SUFFIXES)
    assert [r["final"] for r in out] == ["answer", "abstain", "answer"]
    assert out[0]["outcome_v3"] == "supported_correct" and out[0]["comp"] == "single" and out[0]["part"] == "r6t_squad"
    assert out[0]["strong_v2"] is True  # the v2 columns are filled from v3, as in the other reports
    assert out[2]["outcome_v3"] == "false_abstain"  # an episode already answered is left alone


def test_second_look_table_counts_only_eligible_refusals():
    hidden = entry("h", 0.97)
    hidden["truth"] = {"answerable": False, "excluded_v3": False}
    hidden["score"]["outcome_v3"] = "unsupported_uncited"
    skipped = entry("x", 0.99)
    skipped["truth"] = {"answerable": True, "excluded_v3": True}
    table = {r["threshold"]: r for r in R.second_look_table([entry("a", 0.92), entry("b", 0.92, "wrong"), hidden, skipped])}
    assert (table[0.9]["correct_added"], table[0.9]["wrong_shown"], table[0.9]["hidden_added"]) == (1, 1, 1)
    assert table[0.95]["correct_added"] == 0 and table[0.95]["hidden_added"] == 1
    assert table[0.9]["hidden_refusals"] == 1 and table[0.9]["answerable_refusals"] == 2


def test_checked_applies_the_first_check_at_a_threshold():
    case = {"passages": ["p"], "deterministic": {"cites_unshown": False, "answer_part_absent_from_cited": False}}
    answered = row("q", "answer")
    answered.update(outcome_v2="supported_correct", outcome_v3="supported_correct")
    assert R.checked(answered, case, 0.5, 0.5)["final"] == "answer"
    assert R.checked(answered, case, 0.49, 0.5)["final"] == "abstain"
    with pytest.raises(ValueError):
        R.checked(answered, None, 0.9, 0.5)


def test_the_report_refuses_without_a_committed_preregistration(monkeypatch):
    monkeypatch.setattr(R, "PREREG", "research/does-not-exist.md")
    with pytest.raises(ValueError):
        R.committed_preregistration("research/does-not-exist.md")
