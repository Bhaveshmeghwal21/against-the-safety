"""Adapter from our episodes to AuditKIT's RAG-stress contract (plan section 8).

Each episode becomes an `ak.Sample` whose `actual_trace` is an AuditKIT `RAGTrace`, with the gold in a `RAGCase`
and a per-sample `CorpusSnapshot`. The snapshot holds only the chunks the case needs (gold support, everything the
episode retrieved, and any cited ID that exists in the corpus), so a cited ID that is not in the corpus is
correctly reported as nonexistent. `trace.abstained` is always set explicitly; a parse failure is `abstained=False`.
"""

from __future__ import annotations

import auditkit as ak
from agenttune.rag.rewards.qa_metrics import exact_match_score, f1_score
from auditkit.metrics.rag_stress import CorpusSnapshot, RAGCase, RAGTrace, make_doc

from .corpus import Corpus
from .data import Question
from .episodes import Episode

SCORERS = ["evidence_set_recall", "citation_support", "abstention"]
FRESHNESS = ["freshness"]
AGENT_SCORERS = ["tool_call_validity", "redundant_tool_calls", "agent_loop_detection"]
ALL_SCORERS = SCORERS + AGENT_SCORERS
TOOL_SCHEMAS: list[dict] = []  # set by the caller from the AgentTune tool: [tool.to_schema()]
DECISION_DATE: str | None = None  # ISO "as of" date; with CHUNK_META it switches on AuditKIT freshness (dated documents only)
CHUNK_META: dict[str, dict] = {}  # chunk_id -> {"effective_date", "superseded_date"}


def active_scorers() -> list[str]:
    names = ALL_SCORERS if TOOL_SCHEMAS else SCORERS
    return names + (FRESHNESS if DECISION_DATE and CHUNK_META else [])


class ChunkLookup:
    def __init__(self, corpus: Corpus):
        self._by_id = {c["chunk_id"]: c for c in corpus.chunks}

    def text(self, chunk_id: str) -> str | None:
        c = self._by_id.get(chunk_id)
        return c["text"] if c else None


def _doc(text: str, chunk_id: str | None = None) -> dict:
    meta = CHUNK_META.get(chunk_id or "", {})
    return make_doc(content=text, spans=[{"start": 0, "end": len(text), "page": 1}],
                    effective_date=meta.get("effective_date"), superseded_date=meta.get("superseded_date"))


def rag_case(q: Question, corpus: Corpus, chunks: ChunkLookup, answerable: bool) -> RAGCase:
    support_ids = list(corpus.support_chunks(q).values())
    if not answerable:  # support was masked: nothing can be retrieved or cited, so only abstention is tested
        return RAGCase(case_id=q.qid, question=q.question, answerability="unanswerable", expected_action="abstain", decision_date=DECISION_DATE)
    extra = sorted(set(corpus.extra_copies.get(q.qid, ())) - set(support_ids))
    if extra:  # benchmarks with several passages that each hold the answer (RGB): any one of them is a sufficient set
        support_ids = support_ids + extra
    docs = {c: _doc(chunks.text(c), c) for c in support_ids}
    return RAGCase(
        case_id=q.qid,
        question=q.question,
        answerability="answerable",
        expected_action="answer",
        gold_answer=q.answer,
        sufficient_evidence_sets=[sorted(support_ids)],
        evidence_digests={c: d["digest"] for c, d in docs.items()},
        gold_citations=[{"doc_id": c, "start": 0, "end": len(chunks.text(c))} for c in support_ids],
        decision_date=DECISION_DATE,
    )


def rag_trace(ep: Episode, chunks: ChunkLookup) -> RAGTrace:
    queries = [s.turn.query for s in ep.steps if s.turn.kind == "search"]
    retrieved = ep.retrieved_chunks()
    final = ep.final.turn
    cited = []
    if final.kind == "answer":
        for cid in final.citations:
            t = chunks.text(cid)
            cited.append({"doc_id": cid, "start": 0, "end": len(t) if t is not None else 1})
    return RAGTrace(
        query=ep.question,
        rewrites=queries[1:],
        retrieved=[{"id": h.chunk_id, "score": h.score} for h in retrieved],
        final_context=[{"doc_id": h.chunk_id, "start": 0, "end": len(h.text)} for h in retrieved],
        cited_spans=cited,
        answer_claims=[final.answer] if final.kind == "answer" else [],
        abstained=final.kind == "abstain",
    )


def agent_trace(ep: Episode) -> dict:
    """The tool-call side of the episode for AuditKIT's agent metrics: one turn per search, arguments as the model wrote them.

    An unparseable turn is not a tool call, so `tool_call_validity` only sees well-formed JSON lists; parse failures are
    counted by our own `invalid` rate instead. A repeated or over-budget search is kept as a call so that
    `redundant_tool_calls` and `agent_loop_detection` can see it.
    """
    turns = []
    for s in ep.steps:
        t = s.turn
        if t.kind == "search":
            turns.append([{"name": "search_corpus", "arguments": {"query": t.query}}])
        elif t.kind == "invalid" and s.text.lstrip().startswith("[{"):
            try:
                import json

                calls = json.loads(s.text)
                turns.append([{"name": c.get("tool_name"), "arguments": c.get("parameters", {})} for c in calls if isinstance(c, dict)])
            except ValueError:
                pass
    return {"tool_calls": turns, "messages": [{"role": "assistant", "content": s.text} for s in ep.steps]}


def to_sample(ep: Episode, q: Question, corpus: Corpus, chunks: ChunkLookup, answerable: bool) -> ak.Sample:
    case = rag_case(q, corpus, chunks, answerable)
    trace = {**rag_trace(ep, chunks).to_dict(), **agent_trace(ep)}
    needed = set(corpus.support_chunks(q).values()) | {h.chunk_id for h in ep.retrieved_chunks()}
    needed |= {c for c in ep.final.turn.citations if chunks.text(c) is not None}
    snapshot = CorpusSnapshot(documents={c: _doc(chunks.text(c), c) for c in needed}, chunker_version="paragraph-1")
    return ak.Sample(
        input=q.question,
        target=q.answer,
        id=f"{q.qid}:{ep.condition}",
        actual_output=ep.final.text,
        actual_trace=trace,
        tags=[ep.condition, q.qtype],
        tools=TOOL_SCHEMAS or None,
        metadata={"rag_case": case.to_dict(), "corpus": snapshot.to_dict(), "condition": ep.condition},
    )


def answer_scores(ep: Episode, q: Question) -> dict:
    """Deterministic answer quality via AgentTune's own QA helpers, plus citation-vs-observation checks."""
    t = ep.final.turn
    seen = {h.chunk_id for h in ep.retrieved_chunks()}
    return {
        "answered": t.kind == "answer",
        "abstained": t.kind == "abstain",
        "invalid": t.kind == "invalid",
        "answer_f1": f1_score(t.answer, q.answer) if t.kind == "answer" else None,
        "answer_em": exact_match_score(t.answer, q.answer) if t.kind == "answer" else None,
        "cited_not_observed": sum(c not in seen for c in t.citations) if t.kind == "answer" else None,
        "n_searches": ep.n_searches,
    }


def evaluate(samples: list[ak.Sample], name: str):
    return ak.evaluate(dataset=samples, model="precomputed", scorers=active_scorers(), experiment_name=name)
