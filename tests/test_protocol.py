"""Plan sections 4.1 and 6.1: the action format, AgentTune's parsers, and the Tiny Aya row assertions."""

import re

import pytest
from agenttune.agentic.events import Event, EventKind, EventLog
from agenttune.agentic.rollout_engines.rollout_factory import _extract_tool_calls

from ragharness import protocol, rows
from ragharness.retrieval import Hit, render_observation

SYSTEM = "You can call these tools: search_corpus."


def test_agenttune_rollout_parser_reads_our_search_text():
    import json

    query = 'Aster "Pro" battery'
    calls = _extract_tool_calls(None, protocol.format_search(query))
    assert len(calls) == 1 and calls[0]["function"]["name"] == "search_corpus"
    assert json.loads(calls[0]["function"]["arguments"]) == {"query": query}


def test_agenttune_chunk_id_marker_matches_our_rendering():
    out = render_observation([Hit("c00012", "T1", "a", 1.0), Hit("c00013", "T2", "b", 0.5)])
    assert re.findall(r"chunk_id=(.+?)(?:\s+doc_id=|\])", out) == ["c00012", "c00013"]


@pytest.mark.parametrize(
    "text",
    [
        "The answer is 42.",
        "<answer>42</answer>",  # no citations
        "<answer>42</answer><citations>doc1</citations>",  # not a chunk id
        "<answer></answer><citations>c00001</citations>",
        "<abstain></abstain>",
        '[{"tool_name": "search_corpus", "parameters": {"query": "a"}}, {"tool_name": "search_corpus", "parameters": {"query": "b"}}]',
        '[{"tool_name": "other", "parameters": {"query": "a"}}]',
        '[{"tool_name": "search_corpus", "parameters": {"query": "  "}}]',
        '[{"tool_name": "search_corpus", "parameters": {"query": "a"}, "extra": 1}]',
        "<answer>42</answer><citations>c00001</citations> trailing",
    ],
)
def test_invalid_turns_are_never_abstentions(text):
    assert protocol.parse_turn(text).kind == "invalid"


def test_valid_turns_round_trip():
    assert protocol.parse_turn(protocol.format_search("x y")).query == "x y"
    t = protocol.parse_turn(protocol.format_answer("Paris", ["c00001", "c00002"]))
    assert (t.kind, t.answer, t.citations) == ("answer", "Paris", ["c00001", "c00002"])
    assert protocol.parse_turn(protocol.format_abstain("no coverage stated")).kind == "abstain"


def _log() -> EventLog:
    log = EventLog(tier="full")
    log.append(Event(EventKind.OBSERVATION, {"text": "Who wrote X?"}))
    q = "who wrote X"
    log.append(Event(EventKind.TOOL_CALL, {"action": {"name": "search_corpus", "arguments": {"query": q}}, "raw": protocol.format_search(q)}))
    log.append(Event(EventKind.TOOL_RESULT, {"output": "[chunk_id=c00001] X\nX was written by Ann."}))
    log.append(Event(EventKind.TURN_COMPLETE, {"step": 0}))
    log.append(Event(EventKind.TEXT, {"text": protocol.format_answer("Ann", ["c00001"])}))
    return log


def test_rows_from_agenttune_sft_rows_pass_template_checks():
    [sft_row] = _log().as_dataset_rows("sft")
    got = rows.decisions(sft_row, SYSTEM)
    assert [protocol.parse_turn(d["completion"][0]["content"]).kind for d in got] == ["search", "answer"]
    for d in got:
        checked = rows.check_row(d)
        assert checked["prompt_text"].endswith(rows.PROMPT_END)
        assert checked["completion_text"].endswith(rows.COMPLETION_END)
    # the second row's prompt carries the observation as a user turn, not an empty tool turn
    assert "X was written by Ann." in got[1]["prompt"][-1]["content"] and got[1]["prompt"][-1]["role"] == "user"
    assert "X was written by Ann." in rows.render(got[1]["prompt"], True)


def test_template_rejects_broken_alternation():
    with pytest.raises(Exception):
        rows.render([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}], True)


def test_tool_role_would_render_empty_which_is_why_we_fold():
    text = rows.render([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}, {"role": "tool", "content": "OBS"}], True)
    assert "OBS" not in text
