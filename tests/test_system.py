"""The packaged round-5 system: loop, fail-closed rule, self-check, and agreement with the scripts used for the experiments."""

import sys
from pathlib import Path

import pytest

from ragharness import checker, protocol, system
from ragharness.retrieval import Hit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

DOCS = [
    {"title": "Return deadline", "text": "The return deadline for Nimbus Home devices is 30 days after delivery."},
    {"title": "Warranty", "text": "Nimbus Home devices carry a warranty of two years."},
    {"title": "Shipping", "text": "Orders ship within three business days."},
    {"title": "Support hours", "text": "Support is open on weekdays from nine to five."},
]
QUESTION = "What is the return deadline for Nimbus Home devices?"


@pytest.fixture
def backend(tmp_path):
    b, chunks = system.index_documents(DOCS, tmp_path / "docs.sqlite")
    assert [c["chunk_id"] for c in chunks] == ["c00001", "c00002", "c00003", "c00004"]
    return b


class Script:
    """Replays assistant strings; records the messages of every call."""

    def __init__(self, *texts):
        self.texts, self.calls = list(texts), []

    def __call__(self, messages):
        self.calls.append(list(messages))
        return self.texts.pop(0)


def make(backend, script, p=0.9):
    seen = []

    def support(q, passages, answer):
        seen.append((q, passages, answer))
        return p

    return system.RagSystem(None, None, backend, generate=script, support_fn=support), seen


def test_supported_answer_passes_through(backend):
    script = Script(protocol.format_search("return deadline Nimbus Home"), protocol.format_answer("30 days", ["c00001"]))
    rag, seen = make(backend, script)
    d = rag.ask(QUESTION)
    assert (d.decision, d.answer, d.citations, d.repaired_by, d.model_final) == ("answer", "30 days", ["c00001"], None, "answer")
    assert d.queries == ["return deadline Nimbus Home"] and "c00001" in d.shown_chunk_ids and d.model_calls == 2
    assert len(seen) == 1 and "30 days after delivery" in seen[0][1] and seen[0][2] == "30 days"
    assert script.calls[1][-1]["role"] == "user" and "[chunk_id=c00001]" in script.calls[1][-1]["content"]


def test_unsupported_answer_is_withheld_by_the_self_check(backend):
    rag, _ = make(backend, Script(protocol.format_search("return deadline"), protocol.format_answer("30 days", ["c00001"])), p=0.2)
    d = rag.ask(QUESTION)
    assert d.decision == "abstain" and d.repaired_by == "self_check" and d.model_final == "answer" and d.p_supported == 0.2
    assert d.text.startswith("<abstain>") and d.answer is None


def test_threshold_is_inclusive_of_the_boundary(backend):
    rag, _ = make(backend, Script(protocol.format_search("return deadline"), protocol.format_answer("30 days", ["c00001"])), p=0.5)
    assert rag.ask(QUESTION).decision == "answer"  # blocked only strictly below 0.5, as in the offline evaluation


def test_citation_of_an_unshown_chunk_is_blocked_without_a_model_call(backend):
    rag, seen = make(backend, Script(protocol.format_search("return deadline"), protocol.format_answer("30 days", ["c00099"])))
    d = rag.ask(QUESTION)
    assert d.decision == "abstain" and d.repaired_by == "self_check" and "never shown" in d.reason and seen == []


def test_answer_part_missing_from_cited_passages_is_blocked(backend):
    rag, seen = make(backend, Script(protocol.format_search("return deadline"), protocol.format_answer("30 days; 45 hours", ["c00001"])))
    d = rag.ask(QUESTION)
    assert d.decision == "abstain" and "45 hours" in d.reason and seen == []


def test_repeated_query_ends_as_an_abstention_by_the_fail_closed_rule(backend):
    q = protocol.format_search("return deadline")
    rag, seen = make(backend, Script(q, q))
    d = rag.ask(QUESTION)
    assert d.decision == "abstain" and d.repaired_by == "fail_closed" and d.model_final == "invalid" and "repeated query" in d.reason and seen == []


def test_a_fourth_search_is_invalid(backend):
    rag, _ = make(backend, Script(*[protocol.format_search(f"query number {i}") for i in range(4)]))
    d = rag.ask(QUESTION)
    assert d.decision == "abstain" and d.repaired_by == "fail_closed" and "budget" in d.reason and len(d.queries) == 3


def test_malformed_output_ends_as_an_abstention(backend):
    rag, _ = make(backend, Script("I think the answer is thirty days."))
    d = rag.ask(QUESTION)
    assert d.decision == "abstain" and d.repaired_by == "fail_closed"


def test_the_models_own_abstention_is_kept(backend):
    rag, seen = make(backend, Script(protocol.format_search("return deadline"), protocol.format_abstain("the passages do not say")))
    d = rag.ask(QUESTION)
    assert (d.decision, d.repaired_by, d.model_final) == ("abstain", None, "abstain") and "do not say" in d.text and seen == []


def test_prefilled_evidence_is_in_the_prompt_and_counts_as_shown(backend):
    hit = Hit("c00001", "Return deadline", DOCS[0]["text"], 0.0)
    script = Script(protocol.format_answer("30 days", ["c00001"]))
    rag, seen = make(backend, script)
    d = rag.ask(QUESTION, prefill=[hit])
    assert d.decision == "answer" and "Evidence already provided" in script.calls[0][1]["content"] and "30 days after delivery" in script.calls[0][1]["content"]


def test_wording_and_threshold_match_the_experiment_scripts():
    import verify_round5

    assert verify_round5.WORDINGS["A"] == (checker.WORDING_A, checker.POSITIVE, checker.NEGATIVE)
    assert checker.THRESHOLD == 0.5


@pytest.mark.parametrize("part,text", [("30 days", "the deadline is 30 days"), ("45 days", "the deadline is 30 days"), ("yes", "anything"),
                                       ("the of", "anything"), ("Constitutional Reform Act 2005", "the Constitutional Reform Act of 2005 changed it")])
def test_deterministic_helpers_agree_with_the_case_builder(part, text):
    import build_verifier_cases as cases

    assert checker.contained(part, text) == cases.contained(part, text)
    assert checker.answer_parts("a; b ;; c") == cases.answer_parts("a; b ;; c") == ["a", "b", "c"]
