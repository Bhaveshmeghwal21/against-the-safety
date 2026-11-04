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


def score_results(results, corp: Corpus, store: TrajectoryStore | None = None, run_id: str = "eval", model: str = "") -> dict:
    chunks = audit.ChunkLookup(corp)
    scores, samples = [], {"all": []}
    for job, ep, cost in results:
        s = metrics.score_episode(ep, job.q, corp, job.spec.answerable)
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
            **{n: (None if res.stats.get(n) is None else round(res.stats[n].mean, 3)) for n in audit.SCORERS},
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
