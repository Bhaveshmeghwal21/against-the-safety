"""The demo logic: parsing pasted documents, asking the packaged system with scripted replies, and describing the decision."""

import json
from pathlib import Path

import pytest

from ragharness import demo, protocol

ROOT = Path(__file__).resolve().parents[1]

DOCS_TEXT = "\n".join([
    "Return deadline: The return deadline for Nimbus Home devices is 30 days after delivery.",
    "Warranty: Nimbus Home devices carry a warranty of two years.",
    "Shipping: Orders ship within three business days.",
])
QUESTION = "What is the return deadline for Nimbus Home devices?"


class Script:
    def __init__(self, *texts):
        self.texts = list(texts)

    def __call__(self, messages):
        return self.texts.pop(0)


def session(tmp_path, *replies, p=0.9):
    return demo.DemoSession(None, None, generate=Script(*replies), support_fn=lambda q, passages, answer: p, workdir=tmp_path)


def test_parse_accepts_titled_lines_json_and_bare_text():
    docs, notes = demo.parse_documents('A title: some text\n{"title": "T", "text": "body"}\n\njust a bare passage\n')
    assert docs == [{"title": "A title", "text": "some text"}, {"title": "T", "text": "body"}, {"title": "Passage 3", "text": "just a bare passage"}]
    assert notes == []


def test_parse_splits_at_the_first_colon_only():
    docs, _ = demo.parse_documents("Specs (2026 edition): ports: 4")
    assert docs == [{"title": "Specs (2026 edition)", "text": "ports: 4"}]


def test_parse_warns_about_text_the_model_will_not_read():
    _, notes = demo.parse_documents("Long: " + "x" * 900)
    assert len(notes) == 1 and "800" in notes[0]


@pytest.mark.parametrize("text,fragment", [("", "at least one"), ("   \n\n", "at least one"), ("{broken json", "JSON"), ('{"title": "t"}', "text"), (": no title", "Title")])
def test_parse_rejects_unusable_input_with_one_sentence(text, fragment):
    with pytest.raises(demo.DocumentError) as error:
        demo.parse_documents(text)
    assert fragment in str(error.value)


def test_parse_rejects_too_many_passages():
    with pytest.raises(demo.DocumentError):
        demo.parse_documents("\n".join(f"T{i}: text {i}" for i in range(demo.MAX_DOCS + 1)))


def test_an_answered_question_lists_its_citations_and_searches(tmp_path):
    s = session(tmp_path, protocol.format_search("return deadline"), protocol.format_answer("30 days", ["c00001"]))
    r = s.ask(DOCS_TEXT, QUESTION)
    assert r["decision"] == "answer" and r["answer"] == "30 days" and r["searches"] == ["return deadline"]
    assert r["citations"][0]["title"] == "Return deadline" and "30 days after delivery" in r["citations"][0]["text"]
    verdict = demo.render_verdict(r)
    assert "**30 days**" in verdict and "c00001" in verdict and "P(SUPPORTED) = 0.90" in verdict
    assert "return deadline" in demo.render_trace(r)


def test_a_withheld_answer_shows_why_and_the_draft(tmp_path):
    s = session(tmp_path, protocol.format_search("return deadline"), protocol.format_answer("30 days", ["c00001"]), p=0.1)
    r = s.ask(DOCS_TEXT, QUESTION)
    assert r["decision"] == "abstain" and r["repaired_by"] == "self_check" and "withheld" in r["why"] and r["withheld_draft"]
    assert "Refusal" in demo.render_verdict(r)


def test_the_models_own_refusal_and_the_fail_closed_rule_are_told_apart(tmp_path):
    own = session(tmp_path, protocol.format_search("return deadline"), protocol.format_abstain("not in the passages")).ask(DOCS_TEXT, QUESTION)
    assert own["repaired_by"] is None and "said the documents do not hold" in own["why"]
    q = protocol.format_search("return deadline")
    closed = session(tmp_path, q, q).ask(DOCS_TEXT, QUESTION)
    assert closed["repaired_by"] == "fail_closed" and "repeated query" in closed["why"]


def test_the_index_is_reused_until_the_documents_change(tmp_path):
    turns = [protocol.format_search("return deadline"), protocol.format_answer("30 days", ["c00001"])] * 3
    s = session(tmp_path, *turns)
    s.ask(DOCS_TEXT, QUESTION)
    first = s._prepared
    s.ask(DOCS_TEXT, QUESTION)
    assert s._prepared is first
    s.ask(DOCS_TEXT + "\nSupport: Support is open on weekdays.", QUESTION)
    assert s._prepared is not first and len(s._prepared.chunks) == 4

def test_an_empty_question_is_a_user_error(tmp_path):
    with pytest.raises(demo.DocumentError):
        session(tmp_path).ask(DOCS_TEXT, "   ")


def test_example_files_are_consistent():
    docs = [json.loads(l) for l in (ROOT / "examples" / "nimbus_home.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    questions = json.loads((ROOT / "examples" / "nimbus_questions.json").read_text(encoding="utf-8"))
    assert len(docs) == 24 and all(len(d["text"]) <= demo.MAX_PASSAGE_CHARS for d in docs)
    text = "\n".join(f"{d['title']}: {d['text']}" for d in docs)
    parsed, notes = demo.parse_documents(text)
    assert len(parsed) == 24 and notes == []
    assert {q["kind"] for q in questions} == {"ordinary", "dated", "two_doc", "missing"} and all(q["question"] and q["expected"] for q in questions)
