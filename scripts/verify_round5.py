"""Priority 1 (plan section 27): Tiny Aya as its own answer checker. One forward pass per case; no text is generated.

  python scripts/verify_round5.py --cases data/verifier/cases_dev.jsonl --tag main --adapter <dir> --out runs/verify/main --wordings A,B
  python scripts/verify_round5.py --cases data/verifier/cases_dev.jsonl --tag base --out runs/verify/base --wordings A,B

The model sees the question, the cited passages exactly as the tool rendered them, and the answer. The score is the probability of the
positive first token against the negative one (wording A: SUPPORTED / NOT; wording B: Yes / No), taken from the next-token logits.
The decision threshold (0.5) is fixed in the pre-registration; nothing here is tuned on the labels.
"""

import argparse
import json
import time
from pathlib import Path

WORDINGS = {
    "A": ("You are checking whether an answer is backed by the quoted passages.\n\nQuestion: {question}\n\nPassages:\n{passages}\n\n"
          "Proposed answer: {answer}\n\nIs every part of the proposed answer stated in the passages above? Reply with exactly one word: SUPPORTED or NOT.",
          "SUPPORTED", "NOT"),
    "B": ("Read the passages, then decide whether they contain what is needed to give the answer below to the question.\n\n"
          "Passages:\n{passages}\n\nQuestion: {question}\nAnswer to check: {answer}\n\nDo the passages contain this answer? Reply Yes or No.",
          "Yes", "No"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wordings", default="A,B")
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument('--last-logits-only', action='store_true', help='compute only final-position logits to reduce peak GPU memory')
    args = ap.parse_args()

    import torch

    from ragharness import generate, modelref

    cases = [json.loads(l) for l in Path(args.cases).read_text(encoding="utf-8").splitlines() if l.strip()]
    model_path, revision, source = modelref.resolve()
    model, tok = generate.load_model(model_path, revision, adapter=args.adapter)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"tag": args.tag, "adapter": args.adapter, "model_source": source, "cases": len(cases),
               "batch_size": args.batch_size, "last_logits_only": args.last_logits_only, "wordings": {}}
    for w in args.wordings.split(","):
        template, pos, neg = WORDINGS[w]
        ip, ineg = tok.encode(pos, add_special_tokens=False)[0], tok.encode(neg, add_special_tokens=False)[0]
        assert ip != ineg, "the two answer words must start with different tokens"
        todo = [c for c in cases if c["passages"]]
        prompts = [tok.apply_chat_template([{"role": "user", "content": template.format(question=c["question"], passages=c["passages"], answer=c["answer"])}],
                                           add_generation_prompt=True, return_dict=True)["input_ids"] for c in todo]
        order = sorted(range(len(todo)), key=lambda i: len(prompts[i]))
        rows, t0, top1_pos = {}, time.time(), 0
        for s in range(0, len(order), args.batch_size):
            idx = order[s:s + args.batch_size]
            width = max(len(prompts[i]) for i in idx)
            ids = torch.full((len(idx), width), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(idx), width), dtype=torch.long)
            for r, i in enumerate(idx):
                ids[r, width - len(prompts[i]):] = torch.tensor(prompts[i])
                att[r, width - len(prompts[i]):] = 1
            with torch.no_grad():
                options = {'logits_to_keep': 1} if args.last_logits_only else {}
                logits = model(input_ids=ids.to(model.device), attention_mask=att.to(model.device), use_cache=False, **options).logits[:, -1, :].float()
            two = torch.stack([logits[:, ip], logits[:, ineg]], dim=1).softmax(dim=1)[:, 0]
            for r, i in enumerate(idx):
                top = int(logits[r].argmax())
                top1_pos += top in (ip, ineg)
                rows[i] = {"run": todo[i]["run"], "qid": todo[i]["qid"], "variant": todo[i]["variant"], "wording": w, "p_pos": float(two[r]),
                           "top1_is_candidate": top in (ip, ineg)}
        with (out / f"verdicts_{w}.jsonl").open("w", encoding="utf-8") as f:
            for i in sorted(rows):
                f.write(json.dumps(rows[i]) + "\n")
        summary["wordings"][w] = {"scored": len(rows), "top1_is_a_candidate_token": top1_pos, "seconds": round(time.time() - t0, 1),
                                  "tokens": [tok.decode([ip]), tok.decode([ineg])]}
        print(w, summary["wordings"][w], flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
