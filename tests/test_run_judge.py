"""The judge runner: caching, metering, the budget stop, and what is recorded when a reply cannot be used. No network calls."""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_judge  # noqa: E402

USAGE = {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100}


def numbered(prompt: str, tag: str) -> int:
    block = re.search(rf"<{tag}>\n(.*?)\n</{tag}>", prompt, re.S).group(1)
    return len(re.findall(r"^\d+\. ", block, re.M))


def fake_post_factory(log, bad=None):
    def post(prompt):
        log.append(prompt)
        if bad and bad in prompt:
            reply = "I cannot say."
        elif "Break the ANSWER below" in prompt:
            reply = "1. Shirley Manson is younger than Jim Kerr.\n2. Shirley Manson is Scottish."
        elif "for each numbered CLAIM" in prompt:
            reply = "1: SUPPORTED\n2: UNSUPPORTED"
        elif "Break the RESPONSE below" in prompt:
            reply = "1. Shirley Manson is younger."
        elif "ADDRESSES if it is on-topic" in prompt:
            reply = "1: ADDRESSES"
        elif "SENTENCE of the response" in prompt:
            reply = "\n".join(f"{i}: SUPPORTED" for i in range(1, numbered(prompt, "response_sentences") + 1))
        elif "numbered CONTEXT chunk" in prompt:
            n = numbered(prompt, "contexts")
            reply = "\n".join(f"{i}: {'RELEVANT' if i == 1 else 'IRRELEVANT'}" for i in range(1, n + 1))
        elif "SENTENCE of the reference answer" in prompt:
            reply = "\n".join(f"{i}: ATTRIBUTED" for i in range(1, numbered(prompt, "reference_sentences") + 1))
        elif prompt in ('hello', 'a', 'b'):
            reply = 'ok ' + prompt
        else:
            raise AssertionError(prompt[:200])
        return {"model": run_judge.VERSION, "choices": [{"finish_reason": "stop", "message": {"content": reply}}], "usage": USAGE}
    return post


def case(**kw):
    base = {"system": "main", "part": "r5t_hotpot", "qid": "q1", "variant": "open", "comp": "hotpot", "answerable": True,
            "question": "Which singer is younger, Shirley Manson or Jim Kerr?", "gold": "Shirley Manson", "final_kind": "answer",
            "answer": "Shirley Manson", "citations": ["c1"], "context_ids": ["c1", "c2"],
            "contexts": ["Shirley Manson was born in 1966.", "Jim Kerr was born in 1959."]}
    return {**base, **kw}


def test_answered_answerable_case_gets_every_metric(tmp_path):
    log = []
    judge = run_judge.AzureJudge("k", tmp_path / "calls.jsonl", 5.0, post=fake_post_factory(log))
    rows = {r["metric"]: r for r in run_judge.score_case(case(), judge)}
    assert set(rows) == {"faithfulness", "hallucination", "response_groundedness", "answer_relevancy", "context_relevance", "context_precision", "context_recall"}
    assert rows["faithfulness"]["value"] == 0.5 and rows["hallucination"]["value"] == 0.5
    assert rows["context_relevance"]["value"] == 0.5 and rows["context_precision"]["value"] == 1.0 and rows["context_recall"]["value"] == 1.0
    assert all(r["status"] == "scored" for r in rows.values())
    # hallucination reuses the faithfulness judge calls: 2 (faithfulness) + 1 (groundedness) + 2 (relevancy) + 3 (context) = 8
    assert len(log) == 8 and judge.cache_hits == 2


def test_refusal_and_hidden_evidence_cases_run_fewer_metrics(tmp_path):
    judge = run_judge.AzureJudge("k", tmp_path / "calls.jsonl", 5.0, post=fake_post_factory([]))
    assert {r["metric"] for r in run_judge.score_case(case(final_kind="abstain", answer=None, answerable=False), judge)} == {"context_relevance"}
    bare = {r["metric"]: r for r in run_judge.score_case(case(contexts=[], context_ids=[]), judge)}
    assert set(bare) == {"faithfulness", "hallucination", "response_groundedness", "answer_relevancy"}
    assert bare["faithfulness"]["status"] == "no_context" and bare["answer_relevancy"]["status"] == "scored"
    hidden = {r["metric"] for r in run_judge.score_case(case(answerable=False), judge)}
    assert "context_precision" not in hidden and "context_recall" not in hidden and "faithfulness" in hidden


def test_unreadable_reply_is_recorded_not_scored_as_zero(tmp_path):
    judge = run_judge.AzureJudge("k", tmp_path / "calls.jsonl", 5.0, post=fake_post_factory([], bad="Break the ANSWER below"))
    rows = {r["metric"]: r for r in run_judge.score_case(case(), judge)}
    assert rows["faithfulness"]["status"] == "unreadable_reply" and rows["faithfulness"]["value"] is None
    assert rows["response_groundedness"]["status"] == "scored"


def test_cache_survives_a_restart_and_meters_cost(tmp_path):
    log = []
    path = tmp_path / "calls.jsonl"
    first = run_judge.AzureJudge("k", path, 5.0, post=fake_post_factory(log))
    first.complete("hello")
    expected = 1000 * 0.10 / 1e6 + 100 * 0.50 / 1e6
    assert first.spent == pytest.approx(expected) and first.new_calls == 1
    second = run_judge.AzureJudge("k", path, 5.0, post=fake_post_factory(log))
    assert second.spent == pytest.approx(expected)
    assert second.complete("hello") == first.complete("hello") and len(log) == 1 and second.cache_hits == 1


def test_budget_stops_new_calls_but_not_cached_ones(tmp_path):
    judge = run_judge.AzureJudge("k", tmp_path / "calls.jsonl", 0.00015, post=fake_post_factory([]))
    judge.complete("a")
    with pytest.raises(run_judge.BudgetExceeded):
        judge.complete("b")
    assert judge.complete("a")


def test_key_never_reaches_the_stored_files(tmp_path):
    judge = run_judge.AzureJudge("SECRET-KEY-VALUE", tmp_path / "calls.jsonl", 5.0, post=fake_post_factory([]))
    run_judge.score_case(case(), judge)
    assert "SECRET-KEY-VALUE" not in (tmp_path / "calls.jsonl").read_text(encoding="utf-8")

def test_identical_prompts_asked_together_are_sent_once(tmp_path):
    import threading
    import time

    log = []
    inner = fake_post_factory(log)

    def slow(prompt):
        time.sleep(0.05)
        return inner(prompt)

    judge = run_judge.AzureJudge("k", tmp_path / "calls.jsonl", 5.0, post=slow)
    replies = []
    threads = [threading.Thread(target=lambda: replies.append(judge.complete("hello"))) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(log) == 1 and len(set(replies)) == 1 and judge.new_calls == 1
