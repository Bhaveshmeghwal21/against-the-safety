"""Round 5b, step 2: make a system answer at decision points where abstaining is right (research/2026-11-07-round5b-preregistration.md).

  python scripts/force_answers_round5.py --adapter <dir> --tag main --contexts data/processed/training/force_contexts_r5.jsonl --out runs/forced

For every context that this system produced, the model sees exactly the messages it saw when it abstained, and its reply is started with the tokens of
`<answer>`. It then continues by itself (sampling, temperature 0.7, top-p 0.95, at most 160 new tokens). The text `<answer>` + continuation is the
rejected side of the pair; the builder (scripts/build_pairs_round5b.py) checks that it parses as a complete answer with citations.
Training-split contexts only. Output: <out>/forced_<tag>.jsonl and forced_summary_<tag>.json.
"""

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="omit for the base model")
    ap.add_argument("--tag", required=True, help="system whose contexts are answered, e.g. main or candA")
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=5303)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N contexts")
    ap.add_argument("--greedy", action="store_true", help="greedy continuation instead of sampling")
    args = ap.parse_args()

    import torch

    from ragharness import generate, modelref

    torch.manual_seed(args.seed)
    contexts = [json.loads(l) for l in Path(args.contexts).read_text(encoding="utf-8").splitlines() if l.strip()]
    contexts = [c for c in contexts if c["system"] == args.tag]
    if args.limit:
        contexts = contexts[: args.limit]
    model_path, revision, source = modelref.resolve()
    model, tok = generate.load_model(model_path, revision, adapter=args.adapter)
    gen = generate.HFGenerator(model, tok, batch_size=args.batch_size, temperature=0.0 if args.greedy else args.temperature, top_p=args.top_p)
    encoded = [generate.forced_input(gen, c["prompt"]) for c in contexts]
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0, done = time.time(), 0
    path = out / f"forced_{args.tag}.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for s in range(0, len(order), args.batch_size):
            idx = order[s:s + args.batch_size]
            for i, r in zip(idx, gen._run([encoded[i] for i in idx])):
                f.write(json.dumps({"id": contexts[i]["id"], "system": args.tag, "text": "<answer>" + r.text, "output_tokens": r.output_tokens},
                                   ensure_ascii=False) + "\n")
            f.flush()
            done += len(idx)
            print(f"  {args.tag}: {done}/{len(order)} forced answers, {time.time() - t0:.0f}s", flush=True)
    summary = {"system": args.tag, "adapter": args.adapter, "model_source": source, "contexts": len(contexts), "temperature": 0.0 if args.greedy else args.temperature,
               "top_p": args.top_p, "seed": args.seed, "wall_seconds": round(time.time() - t0, 1), "truncated_inputs": gen.truncated_inputs}
    (out / f"forced_summary_{args.tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
