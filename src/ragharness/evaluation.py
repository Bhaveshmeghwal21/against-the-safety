"""Run one system over the evaluation episodes and score it (plan section 8).

Jobs come from the episode builder, so every system sees the same questions, masks and pre-supplied evidence.
Headline set: `open` and `support_masked` on every reachable question. Extra evidence conditions
(`first_search_masked`, `prefill_*`) run on the first `extra_questions` reachable questions only, to bound compute.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from agenttune.rag.storage.trajectory_store import TrajectoryStore

from . import audit, builder, episodes, metrics, rollout as rollout_mod
from .corpus import Corpus
from .data import Question
from .episodes import Episode
from .retrieval import MaskedSearch
from .tools import CorpusSearchTool

HEADLINE = ("open", "support_masked")


@dataclass
class Job:
    q: Question
    variant: str
    spec: builder.Spec
    scripted_texts: list[str]  # the scripted oracle's assistant strings for this episode (reference system only)


def make_jobs(qs: list[Question], split: str, corp: Corpus, tool: CorpusSearchTool, search: MaskedSearch, extra_questions: int) -> list[Job]:
    jobs: list[Job] = []
    reachable = 0
    for q in qs:
        built = builder.build_question(q, split, corp, tool, search)
        accepted = {b.variant: b for b in built if b.episode is not None}
        if "open" not in accepted:
            continue
        specs = builder.make_specs(q, corp, search)
        reachable += 1
        for variant, b in accepted.items():
            if variant in HEADLINE or reachable <= extra_questions:
                jobs.append(Job(q, variant, specs[variant], [s.text for s in b.episode.steps]))
        if reachable <= extra_questions and "prefill_clean" in specs and "prefill_sufficient" in accepted:
            # same scripted answer as the distractor variant: nothing to search, answer from the supplied evidence
            texts = [s.text for s in accepted["prefill_sufficient"].episode.steps]
            jobs.append(Job(q, "prefill_clean", specs["prefill_clean"], texts))
    return jobs


Factory = Callable[[Job], Callable[[list[dict]], str]]


def run_jobs(jobs: list[Job], corp: Corpus, tool: CorpusSearchTool, system: str, generator_for: Factory, budget: int = 3, progress: int = 0):
    results = []
    t_all = time.time()
    for i, job in enumerate(jobs):
        gen = generator_for(job)
        n0 = getattr(gen, "n_calls", 0), getattr(gen, "input_tokens", 0), getattr(gen, "output_tokens", 0)
        t0 = time.time()
        ep = rollout_mod.rollout(job.q, corp, tool, gen, system, job.variant, job.spec.masked, job.spec.first_search_masked, job.spec.prefill, budget)
        cost = {
            "seconds": round(time.time() - t0, 3),
            "model_calls": getattr(gen, "n_calls", 0) - n0[0],
            "input_tokens": getattr(gen, "input_tokens", 0) - n0[1],
            "output_tokens": getattr(gen, "output_tokens", 0) - n0[2],
        }
        results.append((job, ep, cost))
        if progress and (i + 1) % progress == 0:
            print(f"  {i + 1}/{len(jobs)} episodes, {time.time() - t_all:.0f}s", flush=True)
    return results


def run_jobs_batched(
    jobs: list[Job], corp: Corpus, tool_factory: Callable[[], CorpusSearchTool], system: str, gen, budget: int = 3,
    concurrency: int = 24, progress: int = 0, lenient: bool = False,
):
    """Same episodes as `run_jobs`, with up to `concurrency` of them in flight so their model calls share one batch.

    `gen.generate_batch(list of message lists) -> list[GenOut]`. Each in-flight episode has its own tool instance
    (the tool keeps per-episode mask and shown-passage state). Per-episode `seconds` is amortized: each model call
    is charged its batch's wall time divided by the batch size. Results come back in job order.
    """
    results: list = [None] * len(jobs)
    queue = list(range(len(jobs)))[::-1]
    active: dict[int, dict] = {}
    t_all, done = time.time(), 0

    def start(i: int) -> None:
        job = jobs[i]
        co = rollout_mod.rollout_steps(job.q, corp, tool_factory(), system, job.variant, job.spec.masked, job.spec.first_search_masked, job.spec.prefill, budget, lenient)
        active[i] = {"co": co, "messages": next(co), "seconds": 0.0, "calls": 0, "in": 0, "out": 0}

    while queue or active:
        while queue and len(active) < concurrency:
            start(queue.pop())
        ids = list(active)
        t0 = time.time()
        outs = gen.generate_batch([active[i]["messages"] for i in ids])
        share = (time.time() - t0) / len(ids)
        for i, out in zip(ids, outs):
            st = active[i]
            st["seconds"] += share
            st["calls"] += 1
            st["in"] += out.input_tokens
            st["out"] += out.output_tokens
            try:
                st["messages"] = st["co"].send(out.text)
            except StopIteration as stop:
                cost = {"seconds": round(st["seconds"], 3), "model_calls": st["calls"], "input_tokens": st["in"], "output_tokens": st["out"]}
                results[i] = (jobs[i], stop.value, cost)
                del active[i]
                done += 1
                if progress and done % progress == 0:
                    print(f"  {done}/{len(jobs)} episodes, {time.time() - t_all:.0f}s", flush=True)
    return results


def score_results(
    results, corp: Corpus, store: TrajectoryStore | None = None, run_id: str = "eval", model: str = "", tool_schemas: list | None = None
) -> dict:
    """`tool_schemas` ([tool.to_schema()]) switches on AuditKIT's tool-call metrics (validity, redundancy, loops)."""
    audit.TOOL_SCHEMAS = list(tool_schemas or [])
    chunks = audit.ChunkLookup(corp)
    scores, samples = [], {"all": []}
    for job, ep, cost in results:
        s = metrics.score_episode(ep, job.q, corp, job.spec.answerable)
        s["qtype"] = job.q.qtype
        if audit.CHUNK_META and audit.DECISION_DATE:  # dated documents: did the answer cite an edition that was already superseded?
            cited = ep.final.turn.citations if ep.final.turn.kind == "answer" else []
            s["cites_superseded"] = any((audit.CHUNK_META.get(c, {}).get("superseded_date") or "9999") <= audit.DECISION_DATE for c in cited)
        s.update(cost)
        scores.append(s)
        sample = audit.to_sample(ep, job.q, corp, chunks, job.spec.answerable)
        samples["all"].append(sample)
        samples.setdefault(job.variant, []).append(sample)
        if store is not None:
            store.store(episodes.to_record(ep, job.q, run_id, model, {"variant": job.variant, "outcome": s["outcome"], **cost}))
    summary = {"overall": metrics.summarize(scores), "by_variant": {}}
    for v in sorted({s["variant"] for s in scores}):
        summary["by_variant"][v] = metrics.summarize([s for s in scores if s["variant"] == v])
    headline = [s for s in scores if s["variant"] in HEADLINE]
    summary["headline"] = metrics.summarize(headline)
    summary["auditkit"] = {}
    for v, ss in samples.items():
        if not ss:
            continue
        res = audit.evaluate(ss, f"{model}-{v}")
        summary["auditkit"][v] = {
            **{n: (None if res.stats.get(n) is None else round(res.stats[n].mean, 3)) for n in dict.fromkeys(audit.active_scorers() + (["stale_hit_rate"] if audit.DECISION_DATE and audit.CHUNK_META else []))},
            "not_scored": res.unscored,
        }
    summary["cost"] = {
        "episodes": len(scores),
        "mean_seconds": round(sum(s["seconds"] for s in scores) / len(scores), 3) if scores else None,
        "mean_model_calls": round(sum(s["model_calls"] for s in scores) / len(scores), 2) if scores else None,
        "mean_input_tokens": round(sum(s["input_tokens"] for s in scores) / len(scores), 1) if scores else None,
        "mean_output_tokens": round(sum(s["output_tokens"] for s in scores) / len(scores), 1) if scores else None,
    }
    return {"scores": scores, "summary": summary}
