"""The submitted system as one callable: the main adapter, a three-search loop over a document set, then the two round-5 repairs.

  fail-closed   a rollout that does not end in a valid answer or abstention (a repeated query, a fourth search, malformed output) ends as an abstention;
  self-check    an answer is withheld, and the episode ends as an abstention, when the cited passages do not support it (src/ragharness/checker.py).

Both repairs act on a finished episode, so `ask` returns exactly what the offline evaluation of round 5 scored as "main + fail-closed + V".
The loop mirrors `ragharness.rollout.rollout_steps` (same system prompt, tool, budget, lenient parsing, and the same message layout) without needing
gold answers. Decoding is greedy, one episode at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from . import checker, episodes, protocol
from .retrieval import Hit, render_observation
from .tools import CorpusSearchTool


@dataclass
class Decision:
    decision: str                      # "answer" or "abstain": what the user sees
    text: str                          # the final assistant text shown to the user
    answer: str | None = None
    citations: list[str] = field(default_factory=list)
    repaired_by: str | None = None     # None, "fail_closed" or "self_check"
    model_final: str = ""              # what the model itself ended with: "answer", "abstain" or "invalid"
    model_text: str = ""               # the model's own last text before any repair
    reason: str = ""
    p_supported: float | None = None
    queries: list[str] = field(default_factory=list)
    shown_chunk_ids: list[str] = field(default_factory=list)
    model_calls: int = 0


def _norm(query: str) -> str:
    return " ".join(query.lower().split())


def index_documents(docs: list[dict], db_path) -> tuple[object, list[dict]]:
    """Build a SQLite FTS5 index with AgentTune's backend from [{"title", "text"}, ...]. Returns (backend, chunks)."""
    from pathlib import Path

    from agenttune.rag.retrieval.sqlite_fts import SQLiteFTSBackend

    chunks = [{"chunk_id": f"c{i + 1:05d}", "doc_id": f"c{i + 1:05d}", "title": d["title"], "text": d["text"], "chunk_index": 0} for i, d in enumerate(docs)]
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(str(p) + suffix).unlink(missing_ok=True)
    backend = SQLiteFTSBackend(str(p), match_all=False)
    backend.index(chunks)
    return backend, chunks


class RagSystem:
    def __init__(self, model, tokenizer, backend, *, budget: int = 3, top_k: int = 3, max_chars: int | None = 800, threshold: float = checker.THRESHOLD,
                 generate=None, support_fn=None, force_generate=None, recover_threshold: float | None = None):
        """`generate(messages) -> str` and `support_fn(question, passages, answer) -> float` default to the model; tests pass stubs."""
        self.model, self.tokenizer, self.backend = model, tokenizer, backend
        self.budget, self.top_k, self.max_chars, self.threshold = budget, top_k, max_chars, threshold
        if generate is None:
            from .generate import HFGenerator

            generate = HFGenerator(model, tokenizer, max_new_tokens=160)
        self.generate = generate
        self.support_fn = support_fn or (lambda q, p, a: checker.support_probability(self.model, self.tokenizer, q, p, a))
        if recover_threshold is not None and not 0 <= recover_threshold <= 1:
            raise ValueError('recover_threshold must be between 0 and 1')
        self.recover_threshold = recover_threshold
        if force_generate is None and recover_threshold is not None:
            from .generate import forced_answer
            force_generate = lambda messages: forced_answer(self.model, self.tokenizer, messages)
        self.force_generate = force_generate
        self.system_prompt = protocol.system_prompt(self._tool())

    def _tool(self) -> CorpusSearchTool:
        return CorpusSearchTool(self.backend, top_k=self.top_k, max_chars=self.max_chars)

    def ask(self, question: str, *, masked: frozenset[str] = frozenset(), first_search_masked: frozenset[str] = frozenset(), prefill: list[Hit] | None = None) -> Decision:
        prefill = list(prefill or [])
        tool = self._tool()
        tool.start_episode(masked, first_search_masked, frozenset(h.chunk_id for h in prefill))
        user = question if not prefill else f"{question}\n\nEvidence already provided:\n{render_observation(prefill, max_chars=self.max_chars)}"
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user}]
        shown: dict[str, Hit] = {h.chunk_id: h for h in prefill}
        used: set[str] = set()
        queries: list[str] = []
        calls, n_searches, last_text, final = 0, 0, "", None
        for _ in range(self.budget + 1):
            raw = self.generate(messages)
            calls += 1
            turn, text = protocol.parse_turn_lenient(raw)
            last_text = text
            if turn.kind == "search":
                key = _norm(turn.query)
                if key in used:
                    final = protocol.Turn("invalid", error="repeated query")
                    break
                if n_searches >= self.budget:
                    final = protocol.Turn("invalid", error="search budget exceeded")
                    break
                used.add(key)
                result = tool.execute(query=turn.query)
                n_searches += 1
                queries.append(turn.query)
                for h in tool.last_hits:
                    shown.setdefault(h.chunk_id, h)
                messages += [{"role": "assistant", "content": text}, {"role": "user", "content": protocol.format_tool_result(result.output)}]
                continue
            final = turn
            break
        if final is None:
            final = protocol.Turn("invalid", error="no final turn")
        base = dict(model_final=final.kind, model_text=last_text, queries=queries, shown_chunk_ids=list(shown), model_calls=calls)
        abstain_text = protocol.format_abstain(episodes.ABSTAIN_TEXT.format(question=question))
        if final.kind != 'answer' and self.recover_threshold is not None and shown:
            raw_forced = self.force_generate(messages)
            forced, forced_text = protocol.parse_turn_lenient(raw_forced)
            failure, probability = '', None
            if forced.kind != 'answer' or not forced.citations:
                failure = 'second look did not produce an answer with citations'
            else:
                flags = checker.deterministic_flags(forced.answer or '', forced.citations, list(shown.values()), self.max_chars)
                if flags['cites_unshown']:
                    failure = 'second look cites a passage that was never shown'
                elif flags['answer_part_absent_from_cited']:
                    failure = 'second look answer parts absent from cited passages'
                else:
                    probability = self.support_fn(question, flags['passages'], forced.answer or '')
                    if not math.isfinite(probability) or not 0 <= probability <= 1 or probability < self.recover_threshold:
                        failure = 'second look support probability below threshold or invalid'
            if not failure:
                return Decision('answer', forced_text, answer=forced.answer, citations=list(forced.citations),
                                repaired_by='second_look', p_supported=probability, **base)
            text = last_text if final.kind == 'abstain' else abstain_text
            repair = None if final.kind == 'abstain' else 'fail_closed'
            return Decision('abstain', text, repaired_by=repair, reason=failure, p_supported=probability, **base)
        if final.kind == "abstain":
            return Decision("abstain", last_text, **base)
        if final.kind != "answer":
            return Decision("abstain", abstain_text, repaired_by="fail_closed", reason=f"invalid turn: {final.error}", **base)
        flags = checker.deterministic_flags(final.answer or "", final.citations or [], list(shown.values()), self.max_chars)
        reason = ""
        p = None
        if flags["cites_unshown"]:
            reason = "cites a passage that was never shown"
        elif flags["answer_part_absent_from_cited"]:
            reason = f"answer parts not in the cited passages: {flags['absent_parts']}"
        else:
            p = self.support_fn(question, flags["passages"], final.answer or "")
            if not math.isfinite(p) or not 0 <= p <= 1:
                reason = "self-check: invalid support probability"
            elif p < self.threshold:
                reason = f"self-check: probability of SUPPORTED {p:.2f} below {self.threshold}"
        if reason:
            return Decision("abstain", abstain_text, repaired_by="self_check", reason=reason, p_supported=p, **base)
        return Decision("answer", last_text, answer=final.answer, citations=list(final.citations or []), p_supported=p, **base)
