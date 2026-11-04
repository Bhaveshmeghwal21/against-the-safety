"""Run a model (or any text generator) through the same loop, tool, budget and protocol as the training episodes.

`generate(messages) -> str` returns one assistant string per call. The loop appends search results as user turns
with AgentTune's tool-result wording, exactly as `rows.decisions` builds training prompts. Explicit terminal labels
(plan section 4.1): an unparseable turn, a repeated query and a search past the budget each end the episode with an
`invalid` final turn; none of them counts as an abstention.
"""

from __future__ import annotations

from typing import Callable

from . import protocol
from .corpus import Corpus
from .data import Question
from .episodes import Episode, Step
from .retrieval import Hit
from .tools import CorpusSearchTool

Generator = Callable[[list[dict]], str]


def _norm(query: str) -> str:
    return " ".join(query.lower().split())


def rollout(
    q: Question,
    corpus: Corpus,
    tool: CorpusSearchTool,
    generate: Generator,
    system: str,
    condition: str,
    masked: frozenset[str] = frozenset(),
    first_search_masked: frozenset[str] = frozenset(),
    prefill: list[Hit] | None = None,
    budget: int = 3,
) -> Episode:
    support = corpus.support_chunks(q)
    tool.start_episode(masked, first_search_masked, frozenset(h.chunk_id for h in (prefill or [])))
    ep = Episode(q.qid, q.question, condition, masked, first_search_masked, list(prefill or []), max_chars=tool.max_chars)
    observed = {h.chunk_id: h for h in ep.prefill}
    used: set[str] = set()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": ep.user_text}]

    for _ in range(budget + 1):
        text = generate(messages)
        turn = protocol.parse_turn(text)
        if turn.kind == "search":
            key = _norm(turn.query)
            if key in used:
                ep.steps.append(Step(text, protocol.Turn("invalid", error="repeated query")))
                ep.stop_reason = "repeated_query"
                break
            if ep.n_searches >= budget:
                ep.steps.append(Step(text, protocol.Turn("invalid", error="search budget exceeded")))
                ep.stop_reason = "budget_exceeded"
                break
            used.add(key)
            result = tool.execute(query=turn.query)
            for h in tool.last_hits:
                observed.setdefault(h.chunk_id, h)
            ep.steps.append(Step(text, turn, result.output, list(tool.last_hits)))
            messages += [{"role": "assistant", "content": text}, {"role": "user", "content": protocol.format_tool_result(result.output)}]
            continue
        ep.steps.append(Step(text, turn))
        ep.stop_reason = turn.kind if turn.kind != "invalid" else f"invalid:{turn.error}"
        break

    ep.complete = all(c in observed for c in support.values())
    return ep


class ReplayGenerator:
    """Replays a fixed list of assistant strings, one per call. Used to check the loop against scripted episodes."""

    def __init__(self, texts: list[str]):
        self.texts = list(texts)

    def __call__(self, messages: list[dict]) -> str:
        return self.texts.pop(0)
