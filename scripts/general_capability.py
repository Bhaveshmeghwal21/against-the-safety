"""General-capability check: does the fine-tuning change what the model can do outside our task?
(research/2026-11-16-general-capability-preregistration.md)

  python scripts/general_capability.py --adapter main=<dir> --adapter candF1=<dir> --benches mmlu,arc_c --out runs/general

One loaded base model (4-bit, as in every other run). Systems: `base` (adapters switched off) and one per `--adapter NAME=DIR`. Each multiple-choice question is
shown as plain text, zero-shot, in the usual letter format, with no system prompt and no chat template, and the answer is the letter whose token has the
highest next-token probability after "Answer:". The same prompt goes to every system, so differences come from the weights only. One JSONL line per
(system, question): the letter probabilities, the chosen letter and whether it is right. Greedy by construction, nothing is sampled.
"""

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
HEADERS = {"arc_c": "The following is a multiple choice science question (grade school).", "arc_e": "The following is a multiple choice science question (grade school).",
           "truthfulqa": "The following is a multiple choice question.\n"}


def header(row: dict) -> str:
    bench = row["bench"]
    if bench in HEADERS:
        return HEADERS[bench].strip() + "\n\n"
    return f"The following are multiple choice questions (with answers) about {row['subject'].replace('_', ' ')}.\n\n"


def prompt(row: dict) -> str:
    body = row["question"].strip() + "\n" + "\n".join(f"{LETTERS[i]}. {c.strip()}" for i, c in enumerate(row["choices"]))
    return header(row) + body + "\nAnswer:"


def letter_ids(tok, n: int) -> list[int]:
    """Token id of " A", " B", ... as it appears after "Answer:". Taken from the tokenizer's own split of the full string, so no guess about spacing."""
    base = tok.encode("Answer:", add_special_tokens=False)
    ids = []
    for letter in LETTERS[:n]:
        full = tok.encode(f"Answer: {letter}", add_special_tokens=False)
        assert full[: len(base)] == base and len(full) == len(base) + 1, f"letter {letter} is not one token after 'Answer:'"
        ids.append(full[-1])
    assert len(set(ids)) == n
    return ids


def evaluate(model, tok, rows: list[dict], system: str, batch_tokens: int, max_input: int, out_file) -> dict:
    import torch

    ids_for = {n: letter_ids(tok, n) for n in sorted({len(r["choices"]) for r in rows})}
    enc = [tok.encode(prompt(r), add_special_tokens=True)[-max_input:] for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(enc[i]))
    t0, correct, done = time.time(), 0, 0
    batch: list[int] = []

    def run(batch):
        nonlocal correct, done
        width = max(len(enc[i]) for i in batch)
        ids = torch.full((len(batch), width), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(batch), width), dtype=torch.long)
        for r, i in enumerate(batch):
            ids[r, width - len(enc[i]):] = torch.tensor(enc[i])
            att[r, width - len(enc[i]):] = 1
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            logits = model(input_ids=ids.to(model.device), attention_mask=att.to(model.device), position_ids=pos.to(model.device), use_cache=False,
                           logits_to_keep=1).logits[:, -1, :].float()
        for r, i in enumerate(batch):
            letters = ids_for[len(rows[i]["choices"])]
            scores = torch.log_softmax(logits[r], dim=-1)[letters].tolist()
            pred = max(range(len(scores)), key=scores.__getitem__)
            correct += pred == rows[i]["gold"]
            done += 1
            out_file.write(json.dumps({"system": system, "bench": rows[i]["bench"], "id": rows[i]["id"], "subject": rows[i]["subject"], "gold": rows[i]["gold"], "pred": pred,
                                       "correct": pred == rows[i]["gold"], "logprobs": [round(s, 4) for s in scores]}) + "\n")

    for i in order:
        if batch and (len(batch) + 1) * len(enc[i]) > batch_tokens:
            run(batch)
            batch = []
        batch.append(i)
    if batch:
        run(batch)
    out_file.flush()
    return {"system": system, "bench": rows[0]["bench"], "questions": done, "accuracy": correct / done, "seconds": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", action="append", default=[], help="NAME=DIR, repeatable")
    ap.add_argument("--benches", required=True, help="comma list of files under data/general, e.g. mmlu,arc_c")
    ap.add_argument("--data", default="data/general")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-tokens", type=int, default=6144)
    ap.add_argument("--max-input-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N questions per benchmark")
    args = ap.parse_args()

    import torch
    from peft import PeftModel

    from ragharness import generate, modelref

    model_path, revision, source = modelref.resolve()
    model, tok = generate.load_model(model_path, revision, adapter=None)
    tok.padding_side = "left"
    names = []
    for spec in args.adapter:
        name, path = spec.split("=", 1)
        if names:
            model.load_adapter(path, adapter_name=name)
        else:
            model = PeftModel.from_pretrained(model, path, adapter_name=name)
        names.append(name)
    model.eval()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    summary = {"model_source": source, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", "torch": torch.__version__, "results": []}
    for bench in args.benches.split(","):
        rows = [json.loads(l) for l in (ROOT / args.data / f"{bench}.jsonl").read_text(encoding="utf-8").split("\n") if l.strip()]  # not splitlines(): some MMLU questions contain U+2028 and similar
        if args.limit:
            rows = rows[: args.limit]
        with (out / f"{bench}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for system in ["base"] + names:
                if system == "base":
                    with model.disable_adapter():
                        r = evaluate(model, tok, rows, system, args.batch_tokens, args.max_input_tokens, f)
                else:
                    model.set_adapter(system)
                    r = evaluate(model, tok, rows, system, args.batch_tokens, args.max_input_tokens, f)
                summary["results"].append(r)
                print(json.dumps(r), flush=True)
    (out / f"summary_{args.benches.replace(',', '_')}.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
