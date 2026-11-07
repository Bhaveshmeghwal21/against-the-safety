"""Run the packaged system (src/ragharness/system.py) on the episodes of one benchmark folder, to compare with the offline round-5 evaluation.

  python scripts/check_system_round5.py --adapter <main adapter dir> --bench r5t_hotpot --out runs/system_check [--limit 0]

Builds the same jobs as scripts/run_eval.py (so the same questions, masks and pre-supplied evidence), runs `RagSystem.ask` on each, and writes one line
per episode: the decision, the model's own ending, the repair and the self-check probability. scripts/compare_system_round5.py compares those lines with
the saved offline traces of main (fail-closed and the checker applied offline).
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--bench", required=True, help="a folder name under data/bench, e.g. r5t_hotpot")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from ask import load_model_for_system
    from ragharness import corpus as corpus_mod, data, evaluation, system
    from ragharness.retrieval import MaskedSearch, open_verified_backend
    from ragharness.tools import CorpusSearchTool

    bench = ROOT / "data" / "bench" / args.bench
    corp = corpus_mod.load_corpus(bench / "corpus.json")
    backend = open_verified_backend(bench / "corpus.sqlite", corp)
    tool, search = CorpusSearchTool(backend, top_k=3), MaskedSearch(backend)
    qs = data.read_questions(bench / "questions.jsonl")
    jobs = evaluation.make_jobs(qs, f"bench_{args.bench}", corp, tool, search, 0)
    if (bench / "prompt_variants.json").exists():
        jobs += evaluation.prompt_jobs(qs, corp, tool, json.loads((bench / "prompt_variants.json").read_text(encoding="utf-8")))
    if args.limit:
        jobs = jobs[: args.limit]
    model, tok = load_model_for_system(args.adapter)
    rag = system.RagSystem(model, tok, backend)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with (out / f"system_{args.bench}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for i, job in enumerate(jobs):
            d = rag.ask(job.q.question, masked=job.spec.masked, first_search_masked=job.spec.first_search_masked, prefill=job.spec.prefill)
            f.write(json.dumps({"qid": job.q.qid, "variant": job.variant, "decision": d.decision, "model_final": d.model_final, "repaired_by": d.repaired_by,
                                "reason": d.reason, "p_supported": d.p_supported, "answer": d.answer, "citations": d.citations, "queries": d.queries,
                                "model_calls": d.model_calls}, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(jobs)} episodes, {time.time() - t0:.0f}s", flush=True)
    print(json.dumps({"bench": args.bench, "episodes": len(jobs), "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
