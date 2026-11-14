"""The numbers in the README and the Article are the numbers in the saved reports."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

needs_reports = pytest.mark.skipif(not (ROOT / "reports" / "fair_rerun.json").exists(), reason="saved reports unavailable")


@needs_reports
@pytest.mark.parametrize("document", ["README.md", "article/ARTICLE.md"])
def test_results_tables_match_the_reports(document):
    import make_readme_table as tables

    text = (ROOT / document).read_text(encoding="utf-8")
    for table in (tables.results_table(), tables.hosted_table()):
        for line in table.splitlines():
            assert line in text, f"{document} is missing or has changed: {line}"


@needs_reports
def test_the_headline_claims_in_the_abstract_match_the_reports():
    import make_readme_table as tables

    abstract = (ROOT / "article" / "ABSTRACT.md").read_text(encoding="utf-8")
    main = tables.load("reference_policies.json")["sealed_r5test"]
    assert f"{main['base']['supported_correct'] * 100:.1f}% to {main['main+FB+V']['supported_correct'] * 100:.1f}%" in abstract
    assert f"{main['base']['unsupported']} of {main['base']['hidden_n']} to {main['main+FB+V']['unsupported']} of {main['main+FB+V']['hidden_n']}" in abstract
    assert f"{main['main+FB+V']['macro'] * 100:.1f}%" in abstract and f"{main['always_refuse']['macro'] * 100:.1f}%" in abstract
    assert len(abstract.split()) < 500


def test_local_links_in_the_readme_resolve():
    import re

    text = (ROOT / "README.md").read_text(encoding="utf-8")
    missing = []
    for target in re.findall(r"\]\(([^)\s#]+)", text):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        if not (ROOT / target).exists():
            missing.append(target)
    assert not missing, missing


def test_the_demo_run_in_the_readme_matches_the_saved_output():
    import json

    runs = json.loads((ROOT / "docs" / "examples" / "demo_smoke_t4.json").read_text(encoding="utf-8"))
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert len(runs) == 8
    right = wrong = over_refusal = 0
    for run in runs:
        refused_ok = run["expected"].startswith("refuse")
        if run["decision"] == "abstain":
            right, over_refusal = (right + 1, over_refusal) if refused_ok else (right, over_refusal + 1)
        elif refused_ok or run["answer"].strip().lower() != run["expected"].strip().lower():
            wrong += 1
        else:
            right += 1
    assert (right, wrong, over_refusal) == (5, 2, 1)
    assert "Five of eight match" in text and "five right, two wrong, one over-refusal" in text


@pytest.mark.parametrize("document", ["README.md", "article/ARTICLE.md"])
def test_the_auditkit_table_matches_the_first_round_scorecard(document):
    import re

    scorecard = (ROOT / "reports" / "scorecard_test_lenient.md").read_text(encoding="utf-8")
    text = (ROOT / document).read_text(encoding="utf-8")
    rows = {
        "auditkit_evidence_set_recall": ("| Evidence-set recall |", False),
        "auditkit_citation_support": ("| Citation support |", False),
        "auditkit_abstention": ("| Abstention |", False),
        "tool_call_validity": ("| Tool-call validity |", False),
        "redundant_tool_calls (lower is better)": ("| Redundant tool calls (lower is better) |", True),
        "agent_loop_detection (1.0 = loop-free)": ("| Loop-free episodes |", True),
    }
    for key, (label, percent) in rows.items():
        match = re.search(rf"^\| {re.escape(key)} \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", scorecard, re.M)
        assert match, key
        values = [f"{float(v) * 100:.1f}%" if percent else f"{float(v):.3f}" for v in match.groups()]
        line = next((l for l in text.splitlines() if l.startswith(label)), None)
        assert line is not None, f"{document} has no row {label}"
        assert line.endswith("| " + " | ".join(values) + " |"), f"{document}: {line} does not match {values}"


@needs_reports
@pytest.mark.parametrize("document", ["README.md", "article/ARTICLE.md"])
def test_the_external_suite_table_and_hosted_claims_match_the_reports(document):
    import make_readme_table as tables

    text = (ROOT / document).read_text(encoding="utf-8")
    for line in tables.external_table().splitlines():
        assert line in text, f"{document} is missing or has changed: {line}"
    fair = tables.load("fair_rerun.json")["systems"]
    for key in ("gpt_fair+FB", "deepseek_fair+FB", "main+FB+V", "candF1+FB+V"):
        assert tables.pct(fair[key]["supported_correct"]) in text, key
    for key in ("gpt_fair+FB", "deepseek_fair+FB", "main+FB+V"):
        assert f"{fair[key]['unsupported_rate'] * 100:.0f}%" in text, key
    for key in ("gpt_fair+FB", "deepseek_fair+FB", "main+FB+V", "candF1+FB+V"):
        parts = fair[key]["components"]
        assert tables.pct(parts["single"]) in text and tables.pct(parts["hotpot"]) in text, key
    assert tables.pct(fair["gpt_old+FB"]["macro"]) in text


def test_the_documents_state_that_only_english_was_tested():
    for document in ("README.md", "article/ARTICLE.md"):
        text = (ROOT / document).read_text(encoding="utf-8")
        assert "trained and tested only in English" in text, document
