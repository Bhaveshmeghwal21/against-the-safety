"""Regressions for the November 13 evidence/scoring and training-lineage audit."""
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ragharness import audit, benchmarks as B, corpus as C, evidence, lineage, rescore
from ragharness.data import Paragraph, Question
from ragharness.episodes import Episode, Step
from ragharness import protocol
from ragharness.retrieval import Hit

ROOT = Path(__file__).resolve().parents[1]


def transcript(q, citations, shown, answer=None, answerable=True, masked=()):
    return {"qid": q.qid, "variant": "open", "answerable": answerable, "masked": list(masked),
            "steps": [], "provided_evidence": [{"chunk_id": c} for c in shown],
            "final": {"kind": "answer", "answer": answer or q.answer, "citations": citations}}


@pytest.mark.parametrize("gold", ["yes", "no", "the"])
def test_precision_preserves_exact_gold(gold):
    q = Question("binary", "Does it work?", gold, "comparison", "test", [])
    assert rescore.precision_ok(gold.upper() + ".", q) == (True, 1.0)
    assert not rescore.precision_ok("nonsense", q)[0]


def rgb_world():
    q = Question("rgb-case", "Who won?", "Alice", "single", "test",
                 [Paragraph("one", ["Alice won the prize."]), Paragraph("two", ["The prize went to Alice."])],
                 {"one": [0]}, answer_parts=[["Alice"]])
    corp = C.build_corpus([q], 1)
    a, b = corp.by_question[q.qid].values()
    corp.extra_copies[q.qid] = [b]
    return q, corp, a, b


def test_rgb_alternative_is_sufficient_and_audit_snapshot_contains_all_alternatives():
    q, corp, a, b = rgb_world()
    row = rescore.score(transcript(q, [b], [b]), q, corp)
    assert not row["strong_v2"] and row["strong_v3"]
    assert evidence.sufficient_sets(q, corp) == [[c] for c in sorted([a, b])]
    text = protocol.format_answer(q.answer, [b])
    ep = Episode(q.qid, q.question, "open", frozenset(), prefill=[Hit(b, "two", corp.chunk(b)["text"], 1)],
                 steps=[Step(text, protocol.parse_turn(text))], max_chars=800)
    sample = audit.to_sample(ep, q, corp, audit.ChunkLookup(corp), True)
    assert {c for s in sample.metadata["rag_case"]["sufficient_evidence_sets"] for c in s} == {a, b}
    assert set(sample.metadata["corpus"]["documents"]) >= {a, b}


def test_new_v3_transcripts_roundtrip_without_being_compared_to_legacy_scores(tmp_path, monkeypatch):
    from scripts import rescore_round5 as RS
    from ragharness import episodes, metrics
    q, corp, _, b = rgb_world()
    text = protocol.format_answer(q.answer, [b])
    ep = Episode(q.qid, q.question, "open", frozenset(), prefill=[Hit(b, "two", corp.chunk(b)["text"], 1)],
                 steps=[Step(text, protocol.parse_turn(text))], max_chars=800)
    scored = metrics.score_episode(ep, q, corp, True)
    folder = tmp_path / "main_bench_regression" / "lenient"
    folder.mkdir(parents=True)
    (folder / "scores.jsonl").write_text(json.dumps(scored) + "\n", encoding="utf-8")
    (folder / "transcripts.jsonl").write_text(json.dumps(episodes.to_transcript(ep, q, scored)) + "\n", encoding="utf-8")
    monkeypatch.setattr(RS, "world", lambda kind: (corp, {q.qid: q}))
    _, _, rows, mismatches = RS.rescore_run(folder)
    assert not mismatches and rows[0]["strong_v3"] and not rows[0]["strong_v1"]


def test_compound_alternatives_require_every_component():
    q = Question("rgbint-case", "Who won both?", "Alice; Bob", "integration", "test",
                 [Paragraph("a", ["Alice won."]), Paragraph("b", ["Bob won."]), Paragraph("a2", ["Alice prevailed."])],
                 {"a": [0], "b": [0]}, answer_parts=[["Alice"], ["Bob"]])
    corp = C.build_corpus([q], 1)
    ids = corp.by_question[q.qid]
    corp.extra_copies[q.qid] = [ids["a2"]]
    assert not rescore.score(transcript(q, [ids["a2"]], list(ids.values())), q, corp)["strong_v3"]
    assert rescore.score(transcript(q, [ids["a2"], ids["b"]], list(ids.values())), q, corp)["strong_v3"]


def test_answer_coincidence_does_not_change_hidden_eligibility():
    q = Question("sydney", "Which two facts apply?", "Sydney", "bridge", "test",
                 [Paragraph("gold", ["Sydney supplied both requested facts."]),
                  Paragraph("decoy", ["The Anglican Diocese of Sydney was mentioned."])], {"gold": [0]})
    corp = C.build_corpus([q], 1)
    ids = corp.by_question[q.qid]
    rows = [rescore.score(transcript(q, [ids["decoy"]], shown, answerable=False, masked=[ids["gold"]]), q, corp)
            for shown in ([ids["decoy"]], [])]
    assert rows[0]["excluded_v2"]
    assert all(not r["excluded_v3"] for r in rows)
    assert rows[0]["eligibility_v3"] == rows[1]["eligibility_v3"]


def test_saved_excerpt_must_contain_support():
    q, corp, _, b = rgb_world()
    t = transcript(q, [b], [b])
    t["steps"] = [{"retrieved": [{"chunk_id": b}], "observation": f"[chunk_id={b}] two\nThe prize ..."}]
    assert not rescore.score(t, q, corp)["strong_v3"]
    t["final"]["citations"] = ["missing"]
    assert not rescore.score(t, q, corp)["citation_valid_v3"]


@pytest.mark.parametrize("form,text", [("48", "48-megapixel photos"), ("49", "$49/week"),
                                     ("eight", "her eighth title"), ("February 26", "February 26th"), ("GPT", "ChatGPT")])
def test_rgb_source_forms_survive_ordinal_currency_and_compound_rendering(form, text):
    assert evidence.rgb_contains(form, text)
    assert not evidence.rgb_contains("48", "448 megapixels")


def test_official_squad_offset_selects_second_occurrence_and_rejects_cutoff(tmp_path):
    context = "The name Paris appears here. The capital is Paris."
    offset = context.rindex("Paris")
    assert B.annotated_span(context, "Paris", offset) == (1, [offset, offset + 5])
    rows = [{"id": "offset", "title": "France", "context": context, "question": "What is the capital?",
             "answers": {"text": ["Paris"], "answer_start": [offset]}}]
    file = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist(rows), file)
    questions, _, rejected = B.load_squad2(file)
    assert questions[0].support == {"France (part 1)": [1]}
    assert questions[0].evidence_scope == "paragraph"
    rows[0]["context"] = "Long text. " * 90 + "The capital is Paris."
    rows[0]["answers"]["answer_start"] = [rows[0]["context"].index("Paris")]
    pq.write_table(pa.Table.from_pylist(rows), file)
    assert B.load_squad2(file)[2]["annotated_answer_beyond_display"] == 1


def test_squad_offset_maps_whitespace_without_searching_for_a_different_occurrence():
    context = "Paris is a name. The capital is \t Paris."
    raw = "\t Paris"
    sid, span = B.annotated_span(context, raw, context.index(raw))
    assert sid == 1
    assert " ".join(context.split())[span[0]:span[1]] == "Paris"


def test_squad_natural_question_is_bounded_to_its_paragraph():
    q = Question("unanswerable", "What year?", "", "unanswerable", "dev",
                 [Paragraph("original", ["There is no year here."])], natural_unanswerable=True)
    other = Question("other", "", "", "paragraph", "dev", [Paragraph("other", ["It started in 1754."])])
    corp = C.build_corpus([q, other], 1)
    assert evidence.allowed_ids(q, corp) == set(corp.by_question[q.qid].values())


def test_squad_gold_word_before_cutoff_cannot_replace_annotated_fact_after_cutoff():
    sentences = ["Paris is mentioned here.", "Padding text. " * 65, "The capital is Paris."]
    q = Question("offset-context", "What is the capital?", "Paris", "single", "dev",
                 [Paragraph("France", sentences)], {"France": [2]}, support_spans={"France": [[0, 5]]})
    corp = C.build_corpus([q], 1)
    cid = corp.support_chunks(q)["France"]
    r = rescore.score(transcript(q, [cid], [cid]), q, corp)
    assert r["excluded_v3"] and not r["strong_v3"]


def test_rgb_prefix_collisions_keep_distinct_passages_and_noise_zero_has_five(tmp_path):
    prefix = "These are the same first nine words in each"
    r = {"id": 1, "query": "Who won?", "answer": ["Alice"],
         "positive": [prefix + " Alice won.", prefix + " Bob lost, Alice won."],
         "negative": [f"Different negative passage number {i} is here." for i in range(5)]}
    file = tmp_path / "rgb.jsonl"
    file.write_text(json.dumps(r), encoding="utf-8")
    q = B.load_rgb(file, False)[0][0]
    assert len(q._pos_titles[0]) == 2
    ids = {p.title: str(i) for i, p in enumerate(q.paragraphs)}
    variants = B.rgb_prefill_variants(q, ids, 1)
    assert "rgb_noise_0" not in variants
    assert len(variants["rgb_noise_0.8"]["prefill"]) == 5


def test_prompt_guard_detects_support_under_another_title_and_conflicting_targets():
    q = Question("heldout", "Held-out question?", "answer", "bridge", "test",
                 [Paragraph("gold", ["This is a distinctive sentence containing the held out answer."])], {"gold": [0]})
    guard = lineage.PromptGuard({"test": [q]})
    base = {"prompt": [{"role": "user", "content": "Different question?"}],
            "chosen": [{"role": "assistant", "content": "search"}]}
    conflict = {**base, "chosen": [{"role": "assistant", "content": "abstain"}]}
    leak = {**base, "prompt": [{"role": "user", "content": "New title\n" + q.paragraphs[0].text}]}
    clean, report = lineage.clean_rows([base, conflict, leak], guard)
    assert not clean and report["conflicting_prompts"] == 1
    with pytest.raises(ValueError, match="training validation failed"):
        guard.assert_clean([leak])


def test_training_corpus_removes_heldout_support_preserving_ids():
    held = Question("held", "Held?", "A", "bridge", "test", [Paragraph("held", ["The held out evidence is a distinctive long sentence."])], {"held": [0]})
    train = Question("train", "Train?", "B", "bridge", "train",
                     [Paragraph("train", ["Training support is here."]), held.paragraphs[0]], {"train": [0]})
    original = C.build_corpus([train, held], 1)
    safe, qs = lineage.training_corpus(original, [train], lineage.PromptGuard({"test": [held]}))
    assert safe.support_chunks(qs[0]) == original.support_chunks(train)
    assert len(safe.chunks) == len(qs[0].paragraphs) == 1


def test_current_policy_requires_both_documents_and_rejects_superseded_citation():
    q = Question("policy", "Current price?", "2", "two_doc", "product",
                 [Paragraph("current", ["Damage is covered with Protect."]), Paragraph("plan", ["Protect costs 2 dollars."]),
                  Paragraph("old", ["Damage was excluded."])], {"current": [0], "plan": [0]})
    corp = C.build_corpus([q], 1)
    ids = corp.by_question[q.qid]
    t = transcript(q, [ids["plan"], ids["old"]], list(ids.values()))
    assert not rescore.score(t, q, corp)["strong_v3"]
    t["final"]["citations"] = [ids["current"], ids["plan"], ids["old"]]
    assert not rescore.score(t, q, corp, chunk_metadata={ids["old"]: {"superseded_date": "2026-01-01"}}, decision_date="2026-11-13")["strong_v3"]


@pytest.mark.skipif(not (ROOT / "data/bench/r5t_rgb/questions.jsonl").exists(), reason="saved sealed artifacts unavailable")
def test_saved_rgb_177_oracle_regression():
    from ragharness import data
    corp = C.load_corpus(ROOT / "data/bench/r5t_rgb/corpus.json")
    q = next(q for q in data.read_questions(ROOT / "data/bench/r5t_rgb/questions.jsonl") if q.qid == "rgb-177")
    rows = [json.loads(l) for l in (ROOT / "runs/oracle_r5/r5t_rgb/strict/transcripts.jsonl").read_text(encoding="utf-8").splitlines()]
    t = next(t for t in rows if t["qid"] == q.qid and t["variant"] == "rgb_noise_0.8")
    r = rescore.score(t, q, corp)
    assert r["good_v2"] and not r["strong_v2"] and r["strong_v3"]
