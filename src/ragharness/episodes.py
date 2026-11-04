"""Episodes run through AgentTune's tool and recorded as AgentTune trajectories.

`run_scripted` drives the scripted oracle policy (ceiling.choose_query) through `CorpusSearchTool`. It is a
reference policy used to check the plumbing and report the retrieval ceiling; it is not a trained model.
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
from .ceiling import choose_query
from .corpus import Corpus
from .data import Question
from .retrieval import Hit
from .tools import CorpusSearchTool


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
    steps: list[Step] = field(default_factory=list)

    @property
    def final(self) -> Step:
        return self.steps[-1]

    @property
    def n_searches(self) -> int:
        return sum(s.turn.kind == "search" for s in self.steps)

    def retrieved_chunks(self) -> list[Hit]:
        """Distinct hits in the order first seen across searches."""
        seen: dict[str, Hit] = {}
        for s in self.steps:
            for h in s.hits:
                seen.setdefault(h.chunk_id, h)
        return list(seen.values())


def run_scripted(
    q: Question, corpus: Corpus, tool: CorpusSearchTool, condition: str, masked: frozenset[str] = frozenset(), budget: int = 3
) -> Episode:
    support = corpus.support_chunks(q)
    tool.start_episode(masked)
    ep = Episode(q.qid, q.question, condition, masked)
    observed: dict[str, Hit] = {}
    for step in range(budget):
        query, _ = choose_query(q, support, observed, step)
        if query is None:
            break
        text = protocol.format_search(query)
        turn = protocol.parse_turn(text)
        assert turn.kind == "search" and turn.query == query.strip()
        result = tool.execute(query=turn.query)
        for h in tool.last_hits:
            observed.setdefault(h.chunk_id, h)
        ep.steps.append(Step(text, turn, result.output, list(tool.last_hits)))
    if all(c in observed for c in support.values()):
        text = protocol.format_answer(q.answer, list(support.values()))
    else:
        text = protocol.format_abstain("The accessible passages do not contain the needed evidence.")
    ep.steps.append(Step(text, protocol.parse_turn(text)))
    return ep


def to_eventlog(ep: Episode) -> EventLog:
    """Full-tier EventLog. The exact assistant string is kept in `payload['raw']` so training rows reproduce it."""
    log = EventLog(tier="full")
    log.append(Event(EventKind.OBSERVATION, {"text": ep.question}))
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


def to_record(ep: Episode, q: Question, run_id: str, model: str) -> TrajectoryRecord:
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
        metadata={"qid": ep.qid, "masked": sorted(ep.masked), "final_kind": ep.final.turn.kind},
        steps=steps,
    )
