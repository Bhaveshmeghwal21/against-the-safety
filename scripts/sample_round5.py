"""Draw sampled rollouts on TRAINING episodes only, to build preference pairs (plan section 27, priority 2).

  python scripts/sample_round5.py --adapter <dir> --tag main --out runs/samples/main --k 2 --temperature 0.8

Sets (all training-split material; no development or test episode is read):
  r5train        data/bench/r5train  fresh SQuAD training-split articles that no earlier round used (open, support_masked and the
                                     round-4 prompt variants)
  hotpot_train   HotpotQA training questions (first-round training split), variants open / support_masked / first_search_masked /
                 prefill_sufficient / prefill_partial / prefill_partial_masked
Each set is run k times at the sampling temperature. Output: <out>/samples_<set>.jsonl (one rollout per line) and sample_summary.json.
The pair builder (scripts/build_pairs_round5.py) reads these files together with the local corpora; nothing is decided here.
"""

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

from ragharness import corpus as corpus_mod, data, evaluation, metrics, protocol, round4
from ragharness.retrieval import MaskedSearch, open_verified_backend
from ragharness.tools import CorpusSearchTool

ROOT = Path(__file__).resolve().parents[1]
MODEL = "CohereLabs/tiny-aya-global"
HOTPOT_VARIANTS = {"open", "support_masked", "first_search_masked", "prefill_sufficient", "prefill_partial", "prefill_partial_masked"}
CHUNK = 120  # rollouts per scheduler run; a finished chunk is written to disk before the next starts


def build_set(name: str, hotpot_questions: int, seed: int):
    """Returns (corpus, backend, jobs). Jobs are the same objects the evaluation harness builds."""
    if name == "r5train":
        bench = ROOT / "data" / "bench" / "r5train"
        corp = corpus_mod.load_corpus(bench / "corpus.json")
        backend = open_verified_backend(bench / "corpus.sqlite", corp)
        tool, search = CorpusSearchTool(backend, top_k=3), MaskedSearch(backend)
        qs = data.read_questions(bench / "questions.jsonl")
        jobs = evaluation.make_jobs(qs, "bench_r5train", corp, tool, search, 0)
        jobs += evaluation.prompt_jobs(qs, corp, tool, json.loads((bench / "prompt_variants.json").read_text(encoding="utf-8")))
        return corp, backend, jobs
    if name == "hotpot_train":
        man = json.loads((ROOT / "data" / "manifests" / "split_manifest.json").read_text(encoding="utf-8"))
        splits = {n: data.read_questions(ROOT / "data" / "processed" / f"questions_{n}.jsonl") for n in ("train", "dev", "test")}
        corp = corpus_mod.build_corpus([q for n in splits for q in splits[n]], seed=man["seed"])
        backend = open_verified_backend(ROOT / "data" / "index" / "corpus.sqlite", corp)
        tool, search = CorpusSearchTool(backend, top_k=3), MaskedSearch(backend)
        order = data.stratified_order(sorted(splits["train"], key=lambda q: q.qid), seed)[: hotpot_questions]
        jobs = evaluation.make_jobs(order, "train", corp, tool, search, len(order))
        return corp, backend, [j for j in jobs if j.variant in HOTPOT_VARIANTS]
    raise SystemExit(f"unknown set {name!r}")


def record(job, ep, score, tag: str, set_name: str, run: int) -> dict:
    final = ep.final.turn
    return {
        "set": set_name, "system": tag, "run": run, "qid": job.q.qid, "variant": job.variant, "answerable": job.spec.answerable,
        "qtype": job.q.qtype, "masked": sorted(ep.masked), "first_search_masked": sorted(ep.first_search_masked),
        "prefill": [h.chunk_id for h in ep.prefill], "user_text": ep.user_text, "max_chars": ep.max_chars,
        "steps": [{"text": s.text, "kind": s.turn.kind, "query": s.turn.query, "observation": s.observation,
                   "hits": [h.chunk_id for h in s.hits], "error": s.turn.error} for s in ep.steps],
        "final": {"kind": final.kind, "answer": final.answer, "citations": final.citations, "error": final.error},
        "stop": ep.stop_reason, "outcome": score["outcome"], "good": score["good"], "f1": score["f1"],
        "cited_all_shown": score["cited_all_shown"], "retrieved_aspect_recall": score.get("retrieved_aspect_recall"),
        "all_aspects_cited": score.get("all_aspects_cited"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--tag", required=True, help="system name written into every record, e.g. main or candA")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="r5train,hotpot_train")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--hotpot-questions", type=int, default=40)
    ap.add_argument("--seed", type=int, default=5301)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--limit-rollouts", type=int, default=0, help="debug: stop each set after this many rollouts")
    args = ap.parse_args()

    import torch

    from ragharness import generate, modelref

    torch.manual_seed(args.seed)
    model_path, revision, source = modelref.resolve()
    model, tok = generate.load_model(model_path, revision, adapter=args.adapter)
    gen = generate.HFGenerator(model, tok, batch_size=args.batch_size, temperature=args.temperature, top_p=args.top_p)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"system": args.tag, "adapter": args.adapter, "model_source": source, "k": args.k, "temperature": args.temperature,
               "top_p": args.top_p, "seed": args.seed, "sets": {}}
    for name in args.sets.split(","):
        t0 = time.time()
        corp, backend, jobs = build_set(name, args.hotpot_questions, 53)
        system = protocol.system_prompt(CorpusSearchTool(backend, top_k=3))
        rollouts = [(job, run) for run in range(args.k) for job in jobs]
        if args.limit_rollouts:
            rollouts = rollouts[: args.limit_rollouts]
        print(f"{args.tag}/{name}: {len(jobs)} episodes x k={args.k} = {len(rollouts)} rollouts", flush=True)
        outcomes, path = Counter(), out / f"samples_{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for start in range(0, len(rollouts), CHUNK):
                chunk = rollouts[start:start + CHUNK]
                results = evaluation.run_jobs_batched([j for j, _ in chunk], corp, lambda: CorpusSearchTool(backend, top_k=3), system, gen, 3,
                                                      args.concurrency, progress=0, lenient=True)
                for (job, run), (_, ep, _) in zip(chunk, results):
                    score = metrics.score_episode(ep, job.q, corp, job.spec.answerable)
                    score.update(round4.aspect_coverage(job.q, corp, ep))
                    outcomes[score["outcome"]] += 1
                    f.write(json.dumps(record(job, ep, score, args.tag, name, run), ensure_ascii=False) + "\n")
                f.flush()
                print(f"  {args.tag}/{name}: {min(start + CHUNK, len(rollouts))}/{len(rollouts)} rollouts, {time.time() - t0:.0f}s", flush=True)
        summary["sets"][name] = {"episodes": len(jobs), "rollouts": len(rollouts), "outcomes": dict(outcomes), "wall_seconds": round(time.time() - t0, 1),
                                 "truncated_inputs": gen.truncated_inputs}
        (out / "sample_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
