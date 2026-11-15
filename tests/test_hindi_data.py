"""The Hindi training, development and fresh-test data (research/2026-11-15-hindi-training-preregistration.md)."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_hindi as PH  # noqa: E402
from ragharness import benchmarks as B, qlora, training  # noqa: E402

needs_data = pytest.mark.skipif(not (ROOT / "data/processed/training/hiMain.jsonl").exists(), reason="Hindi training data not built")
needs_bench = pytest.mark.skipif(not (ROOT / "data/bench/xq2_hi/questions.jsonl").exists(), reason="fresh test not built")


def test_a_space_is_inserted_after_a_danda_and_the_answer_offset_follows():
    context = "\u092f\u0939 \u0939\u0948\u0964\u0935\u0939 \u0925\u093e\u0964\u0905\u0902\u0924"  # "yah hai।vah tha।ant"
    new, offset = PH.space_after_danda(context, context.index("\u0905\u0902\u0924"))
    assert new == "\u092f\u0939 \u0939\u0948\u0964 \u0935\u0939 \u0925\u093e\u0964 \u0905\u0902\u0924"
    assert new[offset:] == "\u0905\u0902\u0924"
    assert PH.space_after_danda("a b", 2) == ("a b", 2)  # nothing to insert


def test_sentences_split_at_the_inserted_space_and_answers_are_found():
    first, second = "\u092a\u0939\u0932\u093e \u0935\u093e\u0915\u094d\u092f \u0939\u0948\u0964", "\u0926\u0942\u0938\u0930\u093e \u0935\u093e\u0915\u094d\u092f \u092f\u0939\u093e\u0901 \u0939\u0948\u0964"
    context, start = PH.space_after_danda(first + second, len(first) + second.index("\u092f\u0939\u093e\u0901"))
    rows = [{"id": "q1", "context": context, "question": "\u092f\u0939 \u0915\u0939\u093e\u0901 \u0939\u0948 \u0907\u0938\u0915\u093e?", "answers": {"text": ["\u092f\u0939\u093e\u0901"], "answer_start": [start]}}]
    questions, holders, rejected = B.load_squad_rows(rows, "hi")
    assert len(questions) == 1 and not rejected
    assert questions[0].paragraphs[0].sentences == [first, second] and questions[0].support == {questions[0].paragraphs[0].title: [1]}


def test_broken_translations_are_dropped():
    from ragharness.data import Paragraph, Question
    from ragharness.policy import _STOP_ML

    def question(text, answer):
        return Question(qid="q", question=text, answer=answer, qtype="single", level="dev", paragraphs=[Paragraph("t", ["s"])], lang="hi")

    stop = _STOP_ML["hi"]
    ok = "\u092d\u093e\u0930\u0924 \u0915\u0940 \u0930\u093e\u091c\u0927\u093e\u0928\u0940 \u0915\u094c\u0928 \u0938\u0940 \u0939\u0948"
    assert PH.good_question(question(ok, "\u0928\u0908 \u0926\u093f\u0932\u094d\u0932\u0940"), stop) is None
    assert PH.good_question(question(ok, "\u0915\u093e"), stop) == "answer_length"
    assert PH.good_question(question(ok, "\u0915\u094d\u092f\u093e \u0939\u0948"), stop) == "answer_function_word"
    assert PH.good_question(question(ok + " \u0928\u0908 \u0926\u093f\u0932\u094d\u0932\u0940", "\u0928\u0908 \u0926\u093f\u0932\u094d\u0932\u0940"), stop) == "answer_inside_question"
    assert PH.good_question(question("\u0915\u094c\u0928 \u0939\u0948", "\u0928\u0908 \u0926\u093f\u0932\u094d\u0932\u0940"), stop) == "question_length"


def test_pick_spreads_questions_over_articles():
    from ragharness.data import Paragraph, Question

    qs = [Question(qid=f"{a}{i}", question="", answer="", qtype="single", level="dev", paragraphs=[Paragraph("t", ["s"])]) for a in "ab" for i in range(5)]
    art = {q.qid: q.qid[0] for q in qs}
    chosen = PH.pick(qs, art, ["a", "b"], 4, 1)
    assert sorted(art[q.qid] for q in chosen) == ["a", "a", "b", "b"]


@needs_data
def test_the_hindi_sets_keep_their_rows_in_the_saved_episodes_and_fit_the_length_limit():
    report = json.loads((ROOT / "reports/hindi_training_data.json").read_text(encoding="utf-8"))
    assert not set(report["train_articles"]) & set(report["dev_articles"])
    for name, splits in (("hiMain", ("train", "trainhi")), ("hiF", ("train", "train4", "trainhi"))):
        path = ROOT / f"data/processed/training/{name}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert max(r["meta"]["real_tokens"] for r in rows) <= 5120
        assert rows and sum(r["meta"]["split"] == "trainhi" for r in rows) == report["hindi_rows_chosen"]
        assert PH.sha256(path) == report["sets"][name]["sha256"]
        keys = training.read_keys(path)
        logs, metas = [], []
        for split in splits:
            ll, mm = training.load_episodes(ROOT / "data/processed/episodes", split)
            logs += ll
            metas += mm
        logs, metas = training.episodes_for_keys(logs, metas, keys)
        actual = qlora.to_hf_rows([r for log in logs for r in log.as_dataset_rows("sft")], training.read_system(ROOT / "data/processed/training/main.jsonl"), metas, keys)
        pair = lambda r: json.dumps([r["prompt"], r["completion"]], sort_keys=True)
        assert sorted(map(pair, actual)) == sorted(map(pair, rows))


@needs_data
def test_no_training_article_is_an_english_training_or_test_article():
    report = json.loads((ROOT / "reports/hindi_training_data.json").read_text(encoding="utf-8"))
    alloc = json.loads((ROOT / "reports/round5_allocation.json").read_text(encoding="utf-8"))["allocation"]
    round4 = json.loads((ROOT / "reports/round4_data.json").read_text(encoding="utf-8"))
    used = set(round4["excluded_old_articles"]) | set(alloc["r5train"]) | set(alloc["r5test"])  # rounds 3 (dev and train), 5 (train and test)
    for group in round4["groups"].values():  # round 4: train, dev, test
        used |= set(group["articles"])
    used |= set(json.loads((ROOT / "reports/round6_allocation.json").read_text(encoding="utf-8"))["r6test"])
    assert (set(report["train_articles"]) | set(report["dev_articles"])) <= set(alloc["reserve"])
    assert not (set(report["train_articles"]) | set(report["dev_articles"])) & used


@needs_bench
def test_the_fresh_xquad_questions_are_new_and_in_both_languages():
    first = set(json.loads((ROOT / "reports/xquad_selection.json").read_text(encoding="utf-8"))["ids"])
    fresh = json.loads((ROOT / "reports/hindi_test_selection.json").read_text(encoding="utf-8"))["ids"]
    assert len(fresh) == 300 and not set(fresh) & first
    ids = {lang: [json.loads(l)["qid"] for l in (ROOT / f"data/bench/xq2_{lang}/questions.jsonl").read_text(encoding="utf-8").splitlines()] for lang in ("hi", "en")}
    assert ids["hi"] == ids["en"] == fresh
