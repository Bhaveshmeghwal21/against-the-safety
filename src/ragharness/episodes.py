"""Episodes run through AgentTune's tool and recorded as AgentTune trajectories.

`run_scripted` drives the scripted oracle policy (policy.next_query) through `CorpusSearchTool`. It is a reference
policy used to build verified training episodes and to report the retrieval ceiling; it is not a trained model.
`to_eventlog` records an episode as an AgentTune full-tier `EventLog` (no logprobs are invented), and
`to_record` stores the same episode in AgentTune's `TrajectoryStore` schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from agenttune.agentic.events import Event, EventKind, EventLog
from agenttune.rag.storage.trajectory_store import TrajectoryRecord

from . import protocol
from .corpus import Corpus
from .data import Question
from .policy import next_query
from .retrieval import Hit, render_observation
from .tools import CorpusSearchTool

ABSTAIN_TEXT = 'The accessible passages do not contain the evidence needed to answer: {question}'


@dataclass
class Step:
    text: str  # exact assistant output
    turn: protocol.Turn
    observation: str | None = None
    hits: list[Hit] = field(default_factory=list)


@dataclass
class Episode:
    qid: str
    question: str
    condition: str
    masked: frozenset[str]
    first_search_masked: frozenset[str] = frozenset()
    prefill: list[Hit] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    complete: bool = False  # every gold support chunk was observed or pre-supplied
    stop_reason: str | None = None
    max_chars: int | None = None  # per-passage display limit used for this episode

    @property
    def user_text(self) -> str:
        if not self.prefill:
            return self.question
        return f"{self.question}\n\nEvidence already provided:\n{render_observation(self.prefill, max_chars=self.max_chars)}"

    @property
    def final(self) -> Step:
        return self.steps[-1]

    @property
    def n_searches(self) -> int:
        return sum(s.turn.kind == "search" for s in self.steps)

    def retrieved_chunks(self) -> list[Hit]:
        """Distinct hits (pre-supplied first, then searches) in the order first seen."""
        seen: dict[str, Hit] = {h.chunk_id: h for h in self.prefill}
        for s in self.steps:
            for h in s.hits:
                seen.setdefault(h.chunk_id, h)
        return list(seen.values())


def run_scripted(
    q: Question,
    corpus: Corpus,
    tool: CorpusSearchTool,
    condition: str,
    masked: frozenset[str] = frozenset(),
    budget: int = 3,
    first_search_masked: frozenset[str] = frozenset(),
    prefill: list[Hit] | None = None,
    finish: str = "auto",
) -> Episode:
    """Run the oracle searches, then add the final turn.

    finish="auto": answer if the full support set was observed, otherwise abstain (used by the slice check).
    finish="answer_if_complete": answer if complete, otherwise leave the episode without a final turn
        (the builder marks it unreachable and never trains on it).
    finish="abstain": always abstain (designed missing-support episodes).
    """
    support = corpus.support_chunks(q)
    tool.start_episode(masked, first_search_masked, frozenset(h.chunk_id for h in (prefill or [])))
    ep = Episode(q.qid, q.question, condition, masked, first_search_masked, list(prefill or []), max_chars=tool.max_chars)
    observed: dict[str, Hit] = {h.chunk_id: h for h in ep.prefill}
    used: set[str] = set()
    for n in range(budget):
        query, stop = next_query(q, support, observed, used, n, prefilled=bool(ep.prefill))
        if query is None:
            ep.stop_reason = stop
            break
        used.add(" ".join(query.lower().split()))
        text = protocol.format_search(query)
        turn = protocol.parse_turn(text)
        assert turn.kind == "search" and turn.query == query.strip()
        result = tool.execute(query=turn.query)
        for h in tool.last_hits:
            observed.setdefault(h.chunk_id, h)
        ep.steps.append(Step(text, turn, result.output, list(tool.last_hits)))
    else:
        ep.stop_reason = "budget"
    ep.complete = all(c in observed for c in support.values())

    if finish == "abstain" or (finish == "auto" and not ep.complete):
        text = protocol.format_abstain(ABSTAIN_TEXT.format(question=q.question))
    elif ep.complete:
        text = protocol.format_answer(q.answer, list(support.values()))
    else:
        return ep  # answer_if_complete and not complete: no final turn
    ep.steps.append(Step(text, protocol.parse_turn(text)))
    return ep


def to_eventlog(ep: Episode) -> EventLog:
    """Full-tier EventLog. The exact assistant string is kept in `payload['raw']` so training rows reproduce it."""
    log = EventLog(tier="full")
    log.append(Event(EventKind.OBSERVATION, {"text": ep.user_text}))
    for i, s in enumerate(ep.steps):
        if s.turn.kind == "search":
            log.append(
                Event(EventKind.TOOL_CALL, {"action": {"name": protocol.TOOL_NAME, "arguments": {"query": s.turn.query}}, "raw": s.text})
            )
            log.append(Event(EventKind.TOOL_RESULT, {"output": s.observation, "chunk_ids": [h.chunk_id for h in s.hits]}))
            log.append(Event(EventKind.TURN_COMPLETE, {"step": i}))
        else:
            log.append(Event(EventKind.TEXT, {"text": s.text}))
    return log


def to_transcript(ep: Episode, q: Question, score: dict) -> dict:
    """A readable, self-contained record of one evaluated episode (for the trace viewer and the published trace dataset)."""
    steps = []
    for s in ep.steps:
        steps.append({
            "assistant": s.text, "kind": s.turn.kind, "query": s.turn.query, "observation": s.observation,
            "retrieved": [{"chunk_id": h.chunk_id, "title": h.title} for h in s.hits], "error": s.turn.error,
        })
    final = ep.final.turn
    return {
        "qid": ep.qid, "variant": ep.condition, "question": ep.question, "gold_answer": q.answer, "answerable": score.get("answerable"),
        "provided_evidence": [{"chunk_id": h.chunk_id, "title": h.title} for h in ep.prefill],
        "steps": steps,
        "final": {"kind": final.kind, "answer": final.answer, "citations": final.citations, "missing": final.missing, "error": final.error},
        "outcome": score.get("outcome"), "f1": score.get("f1"), "n_searches": ep.n_searches, "stop": ep.stop_reason,
    }


def eventlog_to_dict(log: EventLog) -> dict:
    return {
        "id": log.id,
        "tier": log.tier,
        "events": [
            {"kind": e.kind.value, "payload": e.payload, "scope": e.scope, "token_span": None, "logprobs": None}
            for e in log.events
        ],
    }


def write_eventlogs(path: Path, logs: list[EventLog]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for log in logs:
            f.write(json.dumps(eventlog_to_dict(log), ensure_ascii=False) + "\n")


def to_record(ep: Episode, q: Question, run_id: str, model: str, extra: dict | None = None) -> TrajectoryRecord:
    steps = []
    for i, s in enumerate(ep.steps):
        is_search = s.turn.kind == "search"
        steps.append(
            {
                "step_number": i,
                "action_name": protocol.TOOL_NAME if is_search else s.turn.kind,
                "action_args": {"query": s.turn.query} if is_search else {},
                "observation": s.observation or "",
                "is_tool_step": is_search,
                "is_terminal": not is_search,
                "tool_calls": [{"tool_name": protocol.TOOL_NAME, "query": s.turn.query, "result": s.observation or ""}] if is_search else [],
            }
        )
    return TrajectoryRecord(
        run_id=run_id,
        question=ep.question,
        gold_answer=q.answer,
        final_answer=ep.final.text,
        n_tool_calls=ep.n_searches,
        has_answer_tag="<answer>" in ep.final.text.lower(),
        condition=ep.condition,
        model=model,
        metadata={"qid": ep.qid, "masked": sorted(ep.masked), "final_kind": ep.final.turn.kind, **(extra or {})},
        steps=steps,
    )
