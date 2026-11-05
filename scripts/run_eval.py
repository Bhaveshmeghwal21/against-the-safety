"""Evaluate one system on the DEV or TEST episodes.

  python scripts/run_eval.py --system oracle   --split dev  --out runs/eval/oracle_dev          (CPU, reference row)
  python scripts/run_eval.py --system base     --split test --out runs/eval/base_test           (GPU, Tiny Aya as shipped)
  python scripts/run_eval.py --system adapter  --adapter runs/main/adapter --split test --out runs/eval/main_test
  python scripts/run_eval.py --system closed_book --split test --out runs/eval/closed_test      (GPU, question only)

TEST is for the final report; use DEV for prompt, retrieval and checkpoint choices.
Writes <out>/scores.jsonl, summary.json, eventlogs.jsonl, trajectories.sqlite/.jsonl (the AgentTune trace dataset).
"""

import argparse
import json
import time
from pathlib import Path

from agenttune.rag.storage.trajectory_store import TrajectoryStore

from ragharness import audit, corpus as corpus_mod, data, evaluation, product, protocol, episodes, rollout
from ragharness.metrics import F1_THRESHOLD
from ragharness.retrieval import MaskedSearch, open_verified_backend
from ragharness.tools import CorpusSearchTool

ROOT = Path(__file__).resolve().parents[1]
MODEL = "CohereLabs/tiny-aya-global"
REVISION = "af89d219b53ed9b13b8a4645f9c8028973510324"
CLOSED_BOOK_SYSTEM = "Answer the question with a short phrase. Reply with only <answer>the answer</answer>."


def closed_book(jobs, gen, out: Path) -> dict:
    """Question only, no tool. Measures how much the model already knows (plan section 5.1).
    Questions with no answer in the documents (SQuAD 2.0's natural unanswerables) are skipped; answers are judged with
    every accepted form (`metrics.judge_answer`), which is plain F1 >= 0.5 for HotpotQA."""
    from agenttune.rag.rewards.qa_metrics import extract_answer_tag

    from ragharness.metrics import judge_answer

    seen, rows, batch, qids = set(), [], [], []
    for job in jobs:
        if job.q.qid in seen or job.q.natural_unanswerable:
            continue
        seen.add(job.q.qid)
        qids.append(job.q)
        batch.append([{"role": "system", "content": CLOSED_BOOK_SYSTEM}, {"role": "user", "content": job.q.question}])
    for q, res in zip(qids, gen.generate_batch(batch)):
        f1, em, ok = judge_answer(extract_answer_tag(res.text), q)
        rows.append({"qid": q.qid, "answer": res.text, "f1": f1, "correct": ok})
    (out / "closed_book.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return {"questions": len(rows), "closed_book_correct": round(sum(r["correct"] for r in rows) / len(rows), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["oracle", "base", "adapter", "closed_book"], required=True)
    ap.add_argument("--split", required=True,
                    help="dev | test | test2 (the fresh second-round split) | product (fictional Nimbus Home suite) | "
                         "bench_<name> (external benchmark built by scripts/prepare_benchmarks.py, e.g. bench_squad2)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--extra-questions", type=int, default=30)
    ap.add_argument("--limit-questions", type=int, default=0)
    ap.add_argument("--no-quantize", action="store_true")
    ap.add_argument("--concurrency", type=int, default=24, help="episodes in flight; their model calls are batched")
    ap.add_argument("--batch-size", type=int, default=12, help="prompts per generate() call")
    ap.add_argument("--abstain-bias", type=float, default=0.0, help="logit bias on the token that starts <abstain>; 0 = the model as trained")
    ap.add_argument("--parse", default="lenient,strict", help="comma list: lenient (first valid action, the fair baseline reading) and/or strict (protocol compliance)")
    args = ap.parse_args()
    if args.split not in ("dev", "test", "test2", "product") and not args.split.startswith("bench_"):
        raise SystemExit(f"unknown split {args.split!r}")
    t_start = time.time()

    man = json.loads((ROOT / "data" / "manifests" / "split_manifest.json").read_text(encoding="utf-8"))
    splits = {n: data.read_questions(ROOT / "data" / "processed" / f"questions_{n}.jsonl") for n in ("train", "dev", "test")}
    bench_dir = ROOT / "data" / "bench" / args.split[len("bench_"):] if args.split.startswith("bench_") else None
    if bench_dir is not None:
        corp = corpus_mod.load_corpus(bench_dir / "corpus.json")
        backend = open_verified_backend(bench_dir / "corpus.sqlite", corp)
        splits[args.split] = data.read_questions(bench_dir / "questions.jsonl")
    elif args.split == "product":
        suite = product.build()
        corp = suite.corpus
        backend = open_verified_backend(ROOT / "data" / "product" / "corpus.sqlite", corp)
        audit.DECISION_DATE, audit.CHUNK_META = product.AS_OF, suite.meta
    else:
        corp = corpus_mod.build_corpus([q for n in splits for q in splits[n]], seed=man["seed"])
        db = ROOT / "data" / "index" / "corpus.sqlite"
        if args.split == "test2":
            splits["test2"] = data.read_questions(ROOT / "data" / "processed" / "questions_test2.jsonl")
            corp = corpus_mod.extend_corpus(corp, splits["test2"])
            db = ROOT / "data" / "index" / "corpus_test2.sqlite"
        backend = open_verified_backend(db, corp)
    tool, search = CorpusSearchTool(backend, top_k=3), MaskedSearch(backend)
    system = protocol.system_prompt(tool)

    if args.split == "product":
        jobs = product.make_jobs(suite)
        qs = [j.q for j in jobs]
    else:
        qs = splits[args.split][: args.limit_questions or None]
        extra_q = 0 if bench_dir is not None else args.extra_questions  # benchmarks: headline episodes only (plan section 22)
        jobs = evaluation.make_jobs(qs, args.split, corp, tool, search, extra_q)
        if bench_dir is not None and (bench_dir / "prompt_variants.json").exists():
            jobs += evaluation.prompt_jobs(qs, corp, tool, json.loads((bench_dir / "prompt_variants.json").read_text(encoding="utf-8")))
    if not jobs:
        raise SystemExit(f"no evaluation episodes for {len(qs)} {args.split} questions: retrieval returned nothing reachable")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.system} on {args.split}: {len(jobs)} episodes over {len({j.q.qid for j in jobs})} questions")

    gen = None
    if args.system == "oracle":
        generator_for = lambda job: rollout.ReplayGenerator(job.scripted_texts)
        model_name = "scripted-oracle"
    else:
        from ragharness import generate, modelref

        model_path, revision, source = modelref.resolve()
        model, tok = generate.load_model(model_path, revision, adapter=args.adapter if args.system == "adapter" else None,
                                         quantize=not args.no_quantize)
        gen = generate.HFGenerator(model, tok, batch_size=args.batch_size, abstain_bias=args.abstain_bias)
        model_name = f"{MODEL}@{REVISION[:8]} ({source})" + (f"+{Path(args.adapter).parent.name}/{Path(args.adapter).name}" if args.system == "adapter" else "")
        if args.system == "closed_book":
            extra = closed_book(jobs, gen, out)
            (out / "summary.json").write_text(json.dumps({"closed_book": extra, "model": model_name}, indent=2), encoding="utf-8")
            print(extra)
            return

    modes = ["strict"] if gen is None else [m.strip() for m in args.parse.split(",") if m.strip()]
    for mode in modes:
        mdir = out / mode
        mdir.mkdir(parents=True, exist_ok=True)
        t_mode = time.time()
        store_path = mdir / "trajectories.sqlite"
        store_path.unlink(missing_ok=True)
        store = TrajectoryStore(str(store_path))
        run_id = f"{args.system}-{args.split}-{mode}"
        store.register_run(run_id, "eval", model_name, config={"top_k": tool.top_k, "budget": 3, "max_chars": tool.max_chars, "index_hash": corp.index_hash, "parse": mode})

        if gen is None:
            results = evaluation.run_jobs(jobs, corp, tool, system, generator_for, 3, progress=25)
        else:
            results = evaluation.run_jobs_batched(jobs, corp, lambda: CorpusSearchTool(backend, top_k=3), system, gen, 3, args.concurrency,
                                                  progress=25, lenient=(mode == "lenient"))
        scored = evaluation.score_results(results, corp, store, run_id, model_name, [tool.to_schema()])
        scored["summary"]["model"] = model_name
        scored["summary"]["split"] = args.split
        scored["summary"]["parse"] = mode
        scored["summary"]["index_hash"] = corp.index_hash
        if gen is not None:
            scored["summary"]["generation"] = {"batch_size": args.batch_size, "concurrency": args.concurrency, "max_new_tokens": gen.max_new_tokens,
                                               "truncated_inputs": gen.truncated_inputs, "abstain_bias": args.abstain_bias, "wall_seconds": round(time.time() - t_mode, 1)}

        with (mdir / "scores.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for s in scored["scores"]:
                f.write(json.dumps(s) + "\n")
        with (mdir / "transcripts.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for (job, ep, _), s in zip(results, scored["scores"]):
                f.write(json.dumps(episodes.to_transcript(ep, job.q, s), ensure_ascii=False) + "\n")
        episodes.write_eventlogs(mdir / "eventlogs.jsonl", [episodes.to_eventlog(ep) for _, ep, _ in results])
        store.export_jsonl(str(mdir / "trajectories.jsonl"), run_id=run_id)
        (mdir / "summary.json").write_text(json.dumps(scored["summary"], indent=2), encoding="utf-8", newline="\n")
        print(f"[{mode}]", json.dumps({k: scored["summary"][k] for k in ("headline", "cost")}, indent=1))

if __name__ == "__main__":
    main()
