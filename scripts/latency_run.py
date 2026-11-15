"""Single-request latency and throughput of the submitted system on one GPU (research/2026-11-16-judge-latency-preregistration.md).

  python scripts/latency_run.py --adapter <main adapter dir> --out runs/latency --per-part 50 --seed 61

One question at a time, batch size 1, greedy decoding, the retrieval backend on local disk. Two arms share one process and one loaded model:

  base       the base model with the adapter switched off, same prompt and search loop, no self-check (the "before" system)
  submitted  the main adapter with the fail-closed rule and the self-check (`ragharness.system.RagSystem`, exactly as `scripts/ask.py` runs it)

Each request is timed from the moment the question is handed to `RagSystem.ask` until the decision comes back. The time is split into the model's
generation turns, the corpus searches and the self-check; "other" is the remainder. The episodes are the jobs `scripts/run_eval.py` builds for the
r5t suites (questions, masks and pre-supplied evidence unchanged), `--per-part` of each, drawn with `--seed`. Statistics use AuditKIT's own
`LatencyStats` and `Throughput` so the definitions (p50, p95, p99, requests per second) are the ones AuditKIT reports.
"""

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

PARTS = ("r5t_squad", "r5t_hotpot", "r5t_rgb")


def pick(jobs: list, per_part: int, seed: int) -> list:
    """`per_part` jobs drawn without replacement, returned in a fixed order that does not depend on the arm."""
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(jobs)), min(per_part, len(jobs))))
    return [jobs[i] for i in idx]


class Meter:
    """Accumulates the seconds spent in one kind of work during the current request."""

    def __init__(self):
        self.total = 0.0
        self.calls = 0

    def wrap(self, fn):
        def timed(*a, **k):
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                self.total += time.perf_counter() - t
                self.calls += 1
        return timed

    def reset(self):
        self.total, self.calls = 0.0, 0


def sync(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize(values_ms: list[float], wall_s: float) -> dict:
    from auditkit.metrics.perf import LatencyStats, Throughput

    lat, thr = LatencyStats(), Throughput()
    for v in values_ms:
        lat.record(v)
    thr.record(sum(values_ms), len(values_ms))
    return {"latency_ms": lat.stats(), "throughput": thr.stats(), "wall_seconds": round(wall_s, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-part", type=int, default=50)
    ap.add_argument("--seed", type=int, default=61)
    ap.add_argument("--warmup", type=int, default=3, help="untimed requests before each arm")
    ap.add_argument("--arms", default="base,submitted")
    args = ap.parse_args()

    import torch
    from ask import load_model_for_system
    from ragharness import corpus as corpus_mod, data, evaluation, system
    from ragharness.retrieval import open_verified_backend
    from ragharness.tools import CorpusSearchTool

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tok = load_model_for_system(args.adapter)

    worlds = []
    for part in PARTS:
        bench = ROOT / "data" / "bench" / part
        corp = corpus_mod.load_corpus(bench / "corpus.json")
        backend = open_verified_backend(bench / "corpus.sqlite", corp)
        from ragharness.retrieval import MaskedSearch
        tool, search = CorpusSearchTool(backend, top_k=3), MaskedSearch(backend)
        qs = data.read_questions(bench / "questions.jsonl")
        jobs = evaluation.make_jobs(qs, f"bench_{part}", corp, tool, search, 0)
        if (bench / "prompt_variants.json").exists():
            jobs += evaluation.prompt_jobs(qs, corp, tool, json.loads((bench / "prompt_variants.json").read_text(encoding="utf-8")))
        worlds.append((part, backend, pick(jobs, args.per_part, args.seed)))

    gen_meter, search_meter, check_meter = Meter(), Meter(), Meter()
    original_execute = CorpusSearchTool.execute
    CorpusSearchTool.execute = lambda self, *a, **k: search_meter.wrap(original_execute)(self, *a, **k)

    info = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", "torch": torch.__version__,
            "python": platform.python_version(), "batch_size": 1, "decoding": "greedy", "max_new_tokens": 160,
            "episodes_per_part": args.per_part, "seed": args.seed, "warmup": args.warmup}
    try:
        import transformers, peft
        info.update(transformers=transformers.__version__, peft=peft.__version__)
    except Exception:
        pass
    summary = {"environment": info, "arms": {}}

    for arm in args.arms.split(","):
        records = []
        t_arm = time.time()
        for part, backend, jobs in worlds:
            if arm == "base":
                rag = system.RagSystem(model, tok, backend, support_fn=lambda q, p, a: 1.0)
            else:
                rag = system.RagSystem(model, tok, backend)
                rag.support_fn = check_meter.wrap(rag.support_fn)
            hf = rag.generate  # the HFGenerator; it counts the tokens it produces
            rag.generate = gen_meter.wrap(hf)
            warm = jobs[: args.warmup] if part == worlds[0][0] else []
            for job in warm:  # untimed: CUDA kernels, allocator and caches settle on the first requests of the run
                ask(rag, job, arm, model)
            for job in jobs:
                for m in (gen_meter, search_meter, check_meter):
                    m.reset()
                tokens_before = hf.output_tokens
                sync(torch)
                t0 = time.perf_counter()
                d = ask(rag, job, arm, model)
                sync(torch)
                total = time.perf_counter() - t0
                records.append({"arm": arm, "part": part, "qid": job.q.qid, "variant": job.variant, "total_ms": total * 1000,
                                "generation_ms": gen_meter.total * 1000, "search_ms": search_meter.total * 1000, "check_ms": check_meter.total * 1000,
                                "model_calls": d.model_calls, "searches": search_meter.calls, "checks": check_meter.calls,
                                "output_tokens": hf.output_tokens - tokens_before, "decision": d.decision, "repaired_by": d.repaired_by})
                if len(records) % 10 == 0:
                    print(f"  {arm}: {len(records)} requests, {time.time() - t_arm:.0f}s", flush=True)
        values = [r["total_ms"] for r in records]
        stats = summarize(values, time.time() - t_arm)
        stats["parts"] = {p: summarize([r["total_ms"] for r in records if r["part"] == p], 0.0)["latency_ms"] for p, _, _ in worlds}
        stats["mean_ms"] = {k: sum(r[k] for r in records) / len(records) for k in ("generation_ms", "search_ms", "check_ms")}
        stats["mean_output_tokens"] = sum(r["output_tokens"] for r in records) / len(records)
        stats["answered"] = sum(r["decision"] == "answer" for r in records)
        stats["requests"] = len(records)
        stats["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None
        summary["arms"][arm] = stats
        with (out / f"requests_{arm}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(json.dumps({arm: stats["latency_ms"], "throughput": stats["throughput"]}), flush=True)
    (out / "latency_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def ask(rag, job, arm, model):
    if arm == "base":
        with model.disable_adapter():
            return rag.ask(job.q.question, masked=job.spec.masked, first_search_masked=job.spec.first_search_masked, prefill=job.spec.prefill)
    return rag.ask(job.q.question, masked=job.spec.masked, first_search_masked=job.spec.first_search_masked, prefill=job.spec.prefill)


if __name__ == "__main__":
    main()
