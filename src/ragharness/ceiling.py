"""CPU retrieval ceiling (plan section 4.3).

The scripted oracle (policy.next_query) searches with the question first. Later searches use a title mention
found in the question or an already observed gold passage, then a keyword reformulation of the question. Queries
are never repeated. This is an upper bound for a policy that builds queries from visible text; it is not a
learned result and not a model baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .corpus import Corpus
from .data import Question
from .policy import next_query
from .retrieval import Hit, MaskedSearch


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
    keywords: bool = True,
    masked: frozenset[str] = frozenset(),
) -> OracleRun:
    support = corpus.support_chunks(q)
    observed: dict[str, Hit] = {}
    used: set[str] = set()
    run = OracleRun(q.qid, q.qtype, len(support))
    for n in range(budget):
        query, stop = next_query(q, support, observed, used, n, keywords=keywords)
        if query is None:
            if stop == "no_new_query":
                run.stuck = run.stuck or stop
            run.found_after.append(sum(c in observed for c in support.values()))
            continue
        used.add(" ".join(query.lower().split()))
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
    out["stuck_no_new_query"] = sum(r.stuck == "no_new_query" for r in runs)
    return out
