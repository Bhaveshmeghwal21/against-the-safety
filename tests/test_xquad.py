"""The multilingual check (research/2026-11-15-multilingual-plan.md): splitting, normalization, scoring, the tokenizer option, and an unchanged English path."""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ragharness import benchmarks as B, checker, corpus as corpus_mod, data, metrics, policy, rescore, textnorm  # noqa: E402
from ragharness.data import Paragraph, Question  # noqa: E402

import build_verifier_cases as BVC  # noqa: E402

HI = "\u0939\u093f\u0928\u094d\u0926\u0940"
DANDA = "\u0964"


def hindi_text() -> str:
    # "Panthers' defence gave 308 points." + "Cavan Short led the team." (Latin digits, as in XQuAD Hindi)
    return ("\u092a\u0948\u0902\u0925\u0930\u094d\u0938 \u0928\u0947 308 \u0905\u0902\u0915 \u0926\u093f\u090f" + DANDA +
            " \u0915\u093e\u0935\u0928 \u0936\u0949\u0930\u094d\u091f \u0928\u0947 \u091f\u0940\u092e \u0915\u093e \u0928\u0947\u0924\u0943\u0924\u094d\u0935 \u0915\u093f\u092f\u093e" + DANDA)


def q(lang="", answer="308", aliases=()):
    para = Paragraph("t", ["x"])
    return Question(qid="1", question="?", answer=answer, qtype="single", level="dev", paragraphs=[para], aliases=list(aliases), lang=lang)


def test_hindi_sentences_split_at_the_danda_and_english_default_is_unchanged():
    parts = B.sentences(hindi_text(), "hi")
    assert len(parts) == 2 and parts[0].endswith(DANDA)
    assert B.sentences(hindi_text()) == [hindi_text()]  # the English regex never sees a sentence end here
    assert B.sentences("Alpha beta. Gamma delta.") == ["Alpha beta.", "Gamma delta."] == B.sentences("Alpha beta. Gamma delta.", "en")
    assert B.sentences("Hola. \u00bfC\u00f3mo est\u00e1s? \u00a1Bien!", "es") == ["Hola.", "\u00bfC\u00f3mo est\u00e1s?", "\u00a1Bien!"]


def test_devanagari_answer_offsets_map_to_the_right_sentence_and_span():
    ctx = hindi_text()
    start = ctx.index("308")
    sid, (a, b) = B.annotated_span(ctx, "308", start, "hi")
    assert sid == 0 and ctx[a:b] == "308"
    second = ctx.index("\u0915\u093e\u0935\u0928")
    assert B.annotated_span(ctx, "\u0915\u093e\u0935\u0928", second, "hi")[0] == 1
    assert B.annotated_span(ctx, "308", start + 1, "hi") is None  # an offset that does not point at the answer


def test_passages_hold_whole_sentences_up_to_the_display_limit():
    parts = ["a" * 300, "b" * 300, "c" * 300, "d" * 900]
    groups = B.xquad_passages(parts, 800)
    assert groups == [[0, 1], [2], [3]]  # 300 + 1 + 300 fits; a third would not; a longer sentence stands alone
    assert all(len(" ".join(parts[i] for i in g)) <= 800 for g in groups[:2])


def test_normalization_follows_the_xquad_rules():
    assert textnorm.normalize("\u092d\u093e\u0930\u0924" + DANDA, "hi") == textnorm.normalize("\u092d\u093e\u0930\u0924", "hi")
    assert textnorm.normalize("El Perro, \u00bfno?", "es") == "perro no"
    assert textnorm.normalize("The \u201cBig\u201d Apple", "en") == "big apple"
    assert textnorm.normalize("\u096a\u0966", "hi") == "40"  # Devanagari digits read as ASCII digits
    assert textnorm.has_script_letters("308", "hi") is False and textnorm.has_script_letters("\u092d\u093e\u0930\u0924", "hi") is True
    assert textnorm.has_script_letters("x", "es") is None


def test_hindi_and_spanish_answers_are_scored_after_normalization():
    assert metrics.judge_answer("308" + DANDA, q("hi"))[2] is True
    assert metrics.judge_answer("309", q("hi"))[2] is False
    assert metrics.judge_answer("la Casa Blanca", q("es", "Casa Blanca"))[2] is True
    # a Hindi gold answer that the English metric would call different only because of the danda
    gold = "\u092d\u093e\u0930\u0924"
    assert metrics.judge_answer(gold + DANDA, q("hi", gold))[1] == 1.0
    assert metrics.judge_answer(gold + DANDA, q("", gold))[1] == 0.0  # legacy path keeps the danda, as before
    assert rescore.precision_ok(gold + DANDA, q("hi", gold)) == (True, 1.0)


def test_english_path_is_untouched_by_the_language_gate():
    assert rescore.tokens("The Eiffel Tower of Paris") == ["eiffel", "tower", "paris"]
    assert rescore.tokens("The Eiffel Tower of Paris") == rescore.tokens("The Eiffel Tower of Paris", "en")
    assert metrics.judge_answer("the Eiffel Tower", q("", "Eiffel Tower"))[2] is True
    assert Question(qid="a", question="q", answer="a", qtype="single", level="dev", paragraphs=[]).lang == ""
    old = {"qid": "a", "question": "q", "answer": "a", "qtype": "single", "level": "dev", "paragraphs": []}  # a file written before `lang` existed
    assert Question(**old).lang == ""


def test_self_check_containment_is_language_aware_and_both_copies_agree():
    cited = hindi_text()
    for lang in ("", "hi", "en", "es"):
        for part, text in (("308", cited), ("309", cited), ("Eiffel Tower", "the eiffel tower of paris"), ("yes", "x")):
            assert checker.contained(part, text, lang) == BVC.contained(part, text, lang)
    assert checker.contained("308" + DANDA, cited, "hi") is True  # the answer ends a sentence; the danda must not block it
    assert checker.contained("308" + DANDA, cited, "") is False  # the English rules keep the danda glued to the word, as before
    gold = "\u092d\u093e\u0930\u0924"
    assert checker.contained(gold + DANDA, "\u092d\u093e\u0930\u0924 \u0915\u093e", "hi") is True
    assert checker.contained(gold + DANDA, "\u092d\u093e\u0930\u0924 \u0915\u093e", "") is False  # English rules: the danda stays glued to the word


def test_scripted_keywords_keep_devanagari_words_whole():
    question = "\u092a\u0948\u0902\u0925\u0930\u094d\u0938 \u0921\u093f\u092b\u093c\u0947\u0902\u0938 \u0928\u0947 \u0915\u093f\u0924\u0928\u0947 \u0905\u0902\u0915 \u0926\u093f\u090f?"
    kw = policy.keyword_query(question, lang="hi").split()
    assert "\u0915\u093f\u0924\u0928\u0947" not in kw and "\u0905\u0902\u0915" in kw and all("?" not in w for w in kw)
    assert policy.keyword_query("Which city is the capital of France?") == policy.keyword_query("Which city is the capital of France?", lang="en")


def test_fts_tokenizer_option_keeps_hindi_words_whole(tmp_path):
    chunks = [{"chunk_id": "c1", "doc_id": "c1", "title": "a", "text": "\u0915\u093f\u0924\u093e\u092c \u0915\u0940 \u0915\u0940\u092e\u0924", "chunk_index": 0},
              {"chunk_id": "c2", "doc_id": "c2", "title": "b", "text": "\u0915\u0947 \u0932\u093f\u090f \u091c\u092e\u0940\u0928", "chunk_index": 0}]
    corp = corpus_mod.Corpus(chunks=chunks, by_question={}, index_hash="x")
    query = "\u0915\u0940"  # "of"; the default tokenizer reads it as the bare letter ka, which also starts "\u0915\u0947" in the other chunk
    default = corpus_mod.build_index(corp, tmp_path / "default.sqlite")
    assert {r["metadata"]["chunk_id"] for r in default.search(query, 5)} == {"c1", "c2"}
    whole = corpus_mod.build_index(corp, tmp_path / "whole.sqlite", tokenizer="unicode61 categories 'L* N* Co M*'")
    assert [r["metadata"]["chunk_id"] for r in whole.search(query, 5)] == ["c1"]
    assert "tokenize" in sqlite3.connect(str(tmp_path / "whole.sqlite")).execute("select sql from sqlite_master where name='chunks_fts'").fetchone()[0]


def test_prepared_sets_share_questions_and_carry_the_language_tag():
    root = ROOT / "data" / "bench"
    if not (root / "xq_hi" / "questions.jsonl").exists():
        pytest.skip("run scripts/prepare_xquad.py first")
    sets = {l: data.read_questions(root / f"xq_{l}" / "questions.jsonl") for l in ("hi", "en", "es")}
    assert {l: len(v) for l, v in sets.items()} == {"hi": 300, "en": 300, "es": 300}
    assert [x.qid for x in sets["hi"]] == [x.qid for x in sets["en"]] == [x.qid for x in sets["es"]]
    assert all(x.lang == l for l, v in sets.items() for x in v)
    man = json.loads((root / "xq_hi" / "manifest.json").read_text(encoding="utf-8"))
    assert man["tokenizer"] == "unicode61 categories 'L* N* Co M*'" and man["n_questions"] == 300
