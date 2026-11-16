"""The numbers in the README and the Article are the numbers in the saved reports."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

needs_reports = pytest.mark.skipif(not (ROOT / "reports" / "fair_rerun.json").exists(), reason="saved reports unavailable")

# The README's longer sections live on the linked pages under docs/. Checks of "the README" read those pages too.
WRITE_UP_PAGES = (
    "docs/GLOSSARY.md", "docs/TRAINING.md", "docs/TOOLS.md", "docs/RESULTS.md",
    "docs/AUDITKIT_METRICS.md", "docs/SIDE_STUDIES.md", "docs/FORECASTS.md",
)


def write_up(document):
    text = (ROOT / document).read_text(encoding="utf-8")
    if document == "README.md":
        text += "".join("\n" + (ROOT / page).read_text(encoding="utf-8") for page in WRITE_UP_PAGES)
    return text


@needs_reports
@pytest.mark.parametrize("document", ["README.md", "article/ARTICLE.md"])
def test_results_tables_match_the_reports(document):
    import make_readme_table as tables

    text = write_up(document)
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


def test_the_readme_links_every_write_up_page():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for page in WRITE_UP_PAGES:
        assert f"]({page})" in text or f"]({page}#" in text, f"README does not link {page}"


def _github_anchor(heading):
    import re

    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def test_links_and_anchors_in_the_readme_and_write_up_pages_resolve():
    import os
    import re

    pages = ["README.md", *WRITE_UP_PAGES]
    texts = {p: (ROOT / p).read_text(encoding="utf-8") for p in pages}
    anchors = {p: {_github_anchor(h) for h in re.findall(r"^#{1,6} (.+)$", t, re.M)} for p, t in texts.items()}
    problems = []
    for page, text in texts.items():
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            path, _, anchor = target.partition("#")
            dest = page if not path else os.path.normpath(os.path.join(os.path.dirname(page), path)).replace("\\", "/")
            if not (ROOT / dest).exists():
                problems.append(f"{page}: missing file {target}")
            elif anchor and dest in anchors and anchor not in anchors[dest]:
                problems.append(f"{page}: no heading #{anchor} in {dest}")
    assert not problems, problems


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
    text = write_up(document)
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

    text = write_up(document)
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


def test_the_documents_say_the_adapter_was_trained_in_english_only_and_show_the_language_check():
    import make_readme_table as tables

    for document in ("README.md", "article/ARTICLE.md"):
        text = write_up(document)
        assert "trained the adapter in English only" in text or "trained the adapter on English only" in text, document
        for line in tables.multilingual_table().splitlines():
            assert line in text, f"{document} is missing or has changed: {line}"
        assert "trained and tested only in English" not in text, document


def test_the_main_card_reports_the_language_check_with_the_saved_numbers():
    import make_readme_table as tables
    from ragharness import release_cards

    rep = tables.load("xquad_results.json")["systems"]
    note = release_cards.multilingual_note()
    for lang in ("hi", "en", "es"):
        assert f"{rep[lang]['main+FB+V']['correct_and_cited']:.1%}" in note
    assert "not a claim" in note


@needs_reports
@pytest.mark.parametrize("document", ["README.md", "article/ARTICLE.md"])
def test_the_fresh_test_table_and_the_second_look_claims_match_the_reports(document):
    import make_readme_table as tables

    text = write_up(document)
    for line in tables.fresh_test_table().splitlines():
        assert line in text, f"{document} is missing or has changed: {line}"
    sl = tables.load("round6_second_look.json")
    extra = tables.load("round6_second_look_extra.json")
    assert sl["decision"]["adopt"] is False  # the documents say the second look was not adopted
    macro = sl["comparisons"]["main+FB+V+second look/main+FB+V"]["macro"]
    interval = f"{macro['difference'] * 100:.1f} points (interval {macro['ci95'][0] * 100:+.1f} to {macro['ci95'][1] * 100:+.1f})".replace("-", "\u2212")
    assert f"rose {interval}" in text
    cited = sl["comparisons"]["main+FB+V+second look/main+FB+V"]["supported_correct"]
    assert f"(+{cited['difference'] * 100:.1f} points, interval +{cited['ci95'][0] * 100:.1f} to +{cited['ci95'][1] * 100:.1f})" in text
    margin = extra["comparisons"]["main+FB+V/always refuse"]["macro"]
    assert f"{margin['difference'] * 100:.1f} points above refusing everything (interval +{margin['ci95'][0] * 100:.1f} to +{margin['ci95'][1] * 100:.1f})" in text
    assert f"{sl['systems']['main+FB+V']['correct_among_answered'] * 100:.1f}% of the answers shown are correct" in text


@needs_reports
@pytest.mark.parametrize("document", ["README.md", "article/ARTICLE.md"])
def test_the_hindi_training_tables_and_verdict_match_the_report(document):
    import make_readme_table as tables

    text = write_up(document)
    for table in (tables.hindi_training_table(), tables.hindi_rules_table()):
        for line in table.splitlines():
            assert line in text, f"{document} is missing or has changed: {line}"
    decisions = tables.load("hindi_results.json")["decisions"]
    assert not any(d["published_as_hindi_variant"] for d in decisions.values())  # the documents say neither adapter qualified
    assert "the submitted English system is unchanged" in text


@needs_reports
def test_the_hindi_cards_state_the_failed_rules_with_the_saved_numbers():
    import make_readme_table as tables
    from ragharness import release_cards

    rep = tables.load("hindi_results.json")
    for name, (slug, parent, _) in release_cards.HINDI_ADAPTERS.items():
        card = release_cards.hindi_card(name, "BMEscanor")
        assert "experimental" in card and "not the submitted system" in card
        assert f"against-the-safety-tiny-aya-rag-{slug}" in card
        assert not rep["decisions"][name]["published_as_hindi_variant"]
        hi = rep["xq2_hi"]["systems"]
        assert f"{hi[f'{name}+FB+V']['correct_and_cited']:.1%}" in card and f"{hi[f'{parent}+FB+V']['correct_and_cited']:.1%}" in card
        for rule, ok in (("A2", rep["decisions"][name]["A2_hindi_safety"]), ("A1", rep["decisions"][name]["A1_hindi_gain"])):
            assert f"| {'met' if ok else 'not met'} |" in card or f": {'met' if ok else 'not met'} |" in card, rule


@needs_reports
def test_the_abstract_reports_the_second_test():
    import make_readme_table as tables

    sl = tables.load("round6_second_look.json")
    abstract = (ROOT / "article" / "ABSTRACT.md").read_text(encoding="utf-8")
    main, refuse = sl["systems"]["main+FB+V"], sl["systems"]["always refuse"]
    macro = sl["comparisons"]["main+FB+V+second look/main+FB+V"]["macro"]
    assert f"{main['macro'] * 100:.1f}% macro success against {refuse['macro'] * 100:.1f}% for refusing everything" in abstract
    assert f"{main['unsupported_rate'] * 100:.1f}% of hidden-evidence episodes answered" in abstract
    assert f"macro success +{macro['difference'] * 100:.1f} points, interval {macro['ci95'][0] * 100:.1f} to +{macro['ci95'][1] * 100:.1f}".replace("-", "\u2212") in abstract
    assert f"{sl['shared_eligible']}-episode" in abstract


@needs_reports
def test_the_main_card_reports_the_second_test_with_the_saved_numbers():
    import make_readme_table as tables
    from ragharness import release_cards

    sl = tables.load("round6_second_look.json")
    note = release_cards.fresh_test_note()
    assert f"{sl['systems']['main+FB+V']['macro']:.1%} macro success against {sl['systems']['always refuse']['macro']:.1%}" in note
    assert "not adopted" in note
    assert str(sl["shared_eligible"]) in note
    assert release_cards._second_look_headline() in release_cards.bench_card("BMEscanor")

@needs_reports
@pytest.mark.parametrize("document", ["README.md", "article/ARTICLE.md"])
def test_the_judge_context_and_numeric_tables_match_the_reports(document):
    import make_readme_table as tables

    text = write_up(document)
    for table in (tables.judge_table(), tables.judge_agreement_table(), tables.context_table(), tables.numeric_table(), tables.latency_table()):
        for line in table.splitlines():
            assert line in text, f"{document} is missing or has changed: {line}"


@needs_reports
def test_the_judge_report_follows_its_pre_registered_readings():
    import json

    rep = json.loads((ROOT / "reports" / "judge_results.json").read_text(encoding="utf-8"))
    assert rep["sample_episodes"] == 300 and rep["judge"]["run_summary"]["stopped_by"] is None
    readings = rep["pre_stated_readings"]
    assert readings["1_annotation_flag_corroborated"]["corroborated"] == (readings["1_annotation_flag_corroborated"]["mean_hallucination_of_main_answers_on_hidden_evidence"] > 0.5)
    low = readings["2_self_check_improves_faithfulness"]["ci95"][0]
    assert readings["2_self_check_improves_faithfulness"]["improves"] == (low > 0)
    manifest = json.loads((ROOT / "data" / "judge" / "sample_r5t.json").read_text(encoding="utf-8"))
    assert manifest["sample_sha256"] == "91aa3283ead72dbfe120ef947714cb0aa531bb59ff77cb8161ebad650216a984"  # the hash written in the pre-registration

@needs_reports
def test_the_abstract_quotes_the_judge_faithfulness_numbers():
    import json

    rep = json.loads((ROOT / "reports" / "judge_results.json").read_text(encoding="utf-8"))["systems"]
    abstract = (ROOT / "article" / "ABSTRACT.md").read_text(encoding="utf-8")
    assert f"{rep['main+FB+V']['metrics']['faithfulness']['all']['mean'] * 100:.1f}% of the claims" in abstract
    assert f"against {rep['base']['metrics']['faithfulness']['all']['mean'] * 100:.1f}% for the base model" in abstract
    assert len(abstract.split()) < 500

@needs_reports
def test_the_abstract_and_limits_quote_the_measured_latency():
    import json

    lat = json.loads((ROOT / "reports" / "latency.json").read_text(encoding="utf-8"))["arms"]["submitted"]["latency_ms"]
    abstract = (ROOT / "article" / "ABSTRACT.md").read_text(encoding="utf-8")
    assert f"a median of {lat['p50'] / 1000:.1f} s (95th percentile {lat['p95'] / 1000:.1f} s)" in abstract
    for document in ("README.md", "article/ARTICLE.md"):
        assert f"a median of {lat['p50'] / 1000:.1f} s per question" in write_up(document)

@needs_reports
@pytest.mark.parametrize("document", ["README.md", "article/ARTICLE.md"])
def test_the_tool_use_table_and_judge_agreement_match_the_reports(document):
    import make_readme_table as tables

    text = write_up(document)
    for line in tables.tool_use_table().splitlines():
        assert line in text, f"{document} is missing or has changed: {line}"
    assert tables.judge_agreement_sentence() in text
    assert tables.tool_use_validity_sentence() in text, f"{document} does not state what tool-call validity does not cover"
    assert "Every search call was well formed" not in text and "Well-formed tool calls" not in text


@needs_reports
@pytest.mark.parametrize("document", ["README.md", "article/ARTICLE.md"])
def test_the_write_up_states_hyperparameters_leak_checks_seed_and_forecasts(document):
    import json

    text = write_up(document)
    report = json.loads((ROOT / "runs/kaggle/ats-train-both/out/main/train_report.json").read_text(encoding="utf-8")) \
        if (ROOT / "runs/kaggle/ats-train-both/out/main/train_report.json").exists() else None
    if report:
        assert f"{report['optimizer_steps']} optimiser steps" in text and f"seed {report['args']['seed']}" in text
        assert f"{report['rows_trained']:,} (537 search" in text
    for phrase in ("**Splits and leak checks.**", "no embedding-based or MinHash", "one training run with one seed", "forecasts we have not tested",
                   "**Training settings.**", "**Library settings.**"):
        assert phrase.lower() in text.lower(), f"{document} lacks: {phrase}"


def test_the_article_abstract_is_the_abstract_file():
    abstract = (ROOT / "article" / "ABSTRACT.md").read_text(encoding="utf-8").strip()
    assert abstract in (ROOT / "article" / "ARTICLE.md").read_text(encoding="utf-8")

@needs_reports
@pytest.mark.parametrize("document", ["README.md", "article/ARTICLE.md"])
def test_the_capability_tables_match_the_report_and_the_abstract_quotes_it(document):
    import json
    import make_readme_table as tables

    text = write_up(document)
    for table in (tables.general_table(), tables.general_rules_table()):
        for line in table.splitlines():
            assert line in text, f"{document} is missing or has changed: {line}"
    rep = json.loads((ROOT / "reports" / "general_capability.json").read_text(encoding="utf-8"))
    abstract = (ROOT / "article" / "ABSTRACT.md").read_text(encoding="utf-8")
    drops = [abs(rep["benchmarks"][b]["difference_from_base"][a]["difference"]) for b in ("mmlu", "arc_c", "arc_e") for a in ("main", "candF1")]
    assert max(drops) < 0.01 and "within 1 point of the base model" in abstract
    tq = rep["benchmarks"]["truthfulqa"]["difference_from_base"]["main"]["difference"]
    assert f"{abs(tq) * 100:.1f} points lower" in abstract and tq < 0
    assert all(r["retained"] for a in rep["rules"].values() for r in a.values()) and "all three English sets" in text

@needs_reports
def test_the_adapter_cards_keep_the_correction_notice_and_carry_the_capability_note():
    from ragharness import release_cards as cards

    for text in (cards.model_card("main", "BMEscanor"), cards.cand_f1_card("BMEscanor")):
        assert "13 November 2026 correction" in text and "General capability (retention check)" in text
    assert "13 November 2026 correction" not in cards.general_capability_note("main")


@pytest.mark.parametrize("document", ["README.md", "article/ARTICLE.md"])
def test_the_dora_side_study_is_in_the_write_up_with_the_same_numbers(document):
    text = write_up(document)
    for number in ("44.71%", "50.09%", "34.00%", "47.34%", "27.82%", "45.41%", "51.74%"):
        assert number in text, f"{document} is missing the DoRA figure {number}"


def test_the_rounds_figure_includes_the_dora_side_study():
    rounds = (ROOT / "docs/figures/rounds.svg").read_text(encoding="utf-8")
    assert "Side study" in rounds and "DoRA instead of LoRA" in rounds

def test_tool_use_report_counts_the_turns_the_validity_metric_never_scores():
    import json

    import report_tool_use as r

    search = {"kind": "search", "assistant": '[{"tool_name": "search_corpus", "parameters": {"query": "x"}}]'}
    repeated = {"kind": "invalid", "assistant": '[{"tool_name": "search_corpus", "parameters": {"query": "x"}}]'}  # a JSON call, so AuditKIT scores it
    free_text = {"kind": "invalid", "assistant": "The answer is 42."}
    broken = {"kind": "invalid", "assistant": '[{"tool_name": "search_corpus", "parameters": {"query": '}
    assert [r.not_a_json_tool_call(s) for s in (search, repeated, free_text, broken)] == [False, False, True, True]
    if (ROOT / "reports" / "tool_use_r5t.json").exists():
        rep = json.loads((ROOT / "reports" / "tool_use_r5t.json").read_text(encoding="utf-8"))["trajectories"]
        for system, suites in rep.items():
            for suite, m in suites.items():
                # an invalid turn ends its episode, so the turns left out of the metric cannot outnumber the invalid endings
                assert m["turns_not_seen_by_validity"] <= round(m["invalid_endings"] * m["episodes"]), (system, suite)
