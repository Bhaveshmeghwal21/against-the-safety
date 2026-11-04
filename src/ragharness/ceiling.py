"""CPU retrieval ceiling (plan section 4.3).

A scripted oracle searches with the question first. For each later search it picks a missing
support title whose name (parenthetical dropped) appears in the question or in an already
observed gold passage, and queries with that name. This is an upper bound on a policy that
builds queries from visible text; it is not a learned result and not a model baseline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .corpus import Corpus
from .data import Question
from .retrieval import Hit, MaskedSearch

_PAREN = re.compile(r"\s*\([^)]*\)\s*$")


def mention(title: str) -> str:
    return _PAREN.sub("", title).strip()


def contains(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, flags=re.IGNORECASE) is not None


def choose_query(
    q: Question, support: dict[str, str], observed: dict[str, Hit], step: int, bridge_query: str = "mention"
) -> tuple[str | None, str | None]:
    """Oracle query for search number `step` (0-based): returns (query, stop_reason).

    stop_reason is "complete" when every support paragraph was already observed, and
    "no_observable_bridge" when no missing title is mentioned in the question or an observed gold passage.
    """
    if step == 0:
        return q.question, None
    missing = [t for t, c in support.items() if c not in observed]
    if not missing:
        return None, "complete"
    visible = [q.question] + [observed[c].text for c in support.values() if c in observed]
    usable = [t for t in missing if any(contains(v, mention(t)) for v in visible)]
    if not usable:
        return None, "no_observable_bridge"
    query = mention(usable[0])
    if bridge_query == "mention+question":
        query = f"{query} {q.question}"
    return query, None


@dataclass
class OracleRun:
    qid: str
    qtype: str
    n_support: int
    found_after: list[int] = field(default_factory=list)  # support paragraphs observed after search 1..budget
    stuck: str | None = None  # why the oracle stopped early
    hit_chars: list[int] = field(default_factory=list)


def run_oracle(
    q: Question,
    corpus: Corpus,
    search: MaskedSearch,
    top_k: int,
    budget: int,
    bridge_query: str = "mention",
    masked: frozenset[str] = frozenset(),
) -> OracleRun:
    support = corpus.support_chunks(q)
    observed: dict[str, Hit] = {}
    run = OracleRun(q.qid, q.qtype, len(support))
    for step in range(budget):
        query, stop = choose_query(q, support, observed, step, bridge_query)
        if query is None:
            if stop == "no_observable_bridge":
                run.stuck = run.stuck or stop
            run.found_after.append(sum(c in observed for c in support.values()))
            continue
        hits = search.search(query, top_k, masked)
        run.hit_chars.extend(len(h.text) for h in hits)
        for h in hits:
            observed.setdefault(h.chunk_id, h)
        run.found_after.append(sum(c in observed for c in support.values()))
    return run


def summarize(runs: list[OracleRun], budget: int) -> dict:
    """Complete-support-set recall and mean per-paragraph recall after each search count."""

    def agg(rs: list[OracleRun], b: int) -> dict:
        if not rs:
            return {"n": 0}
        complete = sum(r.found_after[b - 1] == r.n_support for r in rs)
        para = sum(r.found_after[b - 1] / r.n_support for r in rs) / len(rs)
        return {"n": len(rs), "complete_recall": round(complete / len(rs), 3), "paragraph_recall": round(para, 3)}

    out: dict = {}
    for b in range(1, budget + 1):
        out[f"searches<={b}"] = {
            "all": agg(runs, b),
            "bridge": agg([r for r in runs if r.qtype == "bridge"], b),
            "comparison": agg([r for r in runs if r.qtype == "comparison"], b),
        }
    out["stuck_no_observable_bridge"] = sum(r.stuck == "no_observable_bridge" for r in runs)
    return out
