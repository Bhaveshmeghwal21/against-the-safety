"""Train the two-token answer checker on prepared, question-disjoint cases."""

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from ragharness.checker import NEGATIVE, POSITIVE, WORDING_A as WORDING


STAND_IN = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def two_way_loss(logits, pos_id, neg_id, labels):
    pair = torch.stack((logits[:, pos_id], logits[:, neg_id]), dim=1)
    return F.cross_entropy(pair.float(), 1 - labels.long())


def read(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def encode(cases, tokenizer, max_length):
    encoded, dropped = [], 0
    for case in cases:
        content = WORDING.format(question=case["question"], passages=case["passages"], answer=case["answer"])
        ids = list(tokenizer.apply_chat_template([{"role": "user", "content": content}],
                                                 add_generation_prompt=True, return_dict=True)["input_ids"])
        if len(ids) > max_length:
            dropped += 1
        else:
            encoded.append((ids, int(case["label"])))
    return encoded, dropped


def validation(model, rows, device, positive, negative, amp):
    values = []
    model.eval()
    with torch.no_grad(), amp():
        for tokens, label in rows:
            ids = torch.tensor([tokens], device=device)
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
            prob = torch.softmax(torch.stack([logits[:, positive], logits[:, negative]], dim=1), dim=1)[0, 0].item()
            values.append((prob, label))
    yes = [p for p, label in values if label == 1]
    no = [p for p, label in values if label == 0]
    auc = (sum((a > b) + 0.5 * (a == b) for a in yes for b in no) / (len(yes) * len(no))) if yes and no else None
    return {"rows": len(values), "auroc": auc,
            "supported_kept_at_0_5": sum(p >= 0.5 for p in yes) / len(yes) if yes else None,
            "unsupported_passed_at_0_5": sum(p >= 0.5 for p in no) / len(no) if no else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--init-adapter")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2560)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--stand-in", action="store_true")
    args = parser.parse_args()
    train_cases, val_cases = read(args.cases), read(args.val)
    if {c["qid"] for c in train_cases} & {c["qid"] for c in val_cases}:
        raise ValueError("Training and validation questions overlap")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from ragharness import modelref, qlora, rows

    if args.stand_in:
        path, revision, device = STAND_IN, None, "cpu"
        # A small balanced sample exercises the training loop without a GPU.
        train_cases = [c for label in (0, 1) for c in [x for x in train_cases if x["label"] == label][:2]]
        val_cases = [c for label in (0, 1) for c in [x for x in val_cases if x["label"] == label][:2]]
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        tokenizer.chat_template = rows.FIXTURE.read_text(encoding="utf-8")
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, local_files_only=True)
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
                                               target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    else:
        if not torch.cuda.is_available():
            raise SystemExit("A CUDA GPU is required for verifier training; use --stand-in for the CPU check.")
        path, revision, _ = modelref.resolve()
        device = "cuda"
        tokenizer = AutoTokenizer.from_pretrained(path, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(path, revision=revision, dtype=torch.float16, device_map={"": 0},
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        if args.init_adapter:
            model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
                                                     target_modules="all-linear", task_type="CAUSAL_LM"))
        qlora.cast_trainable_to_fp32(model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model.config.use_cache = False
    positive = tokenizer.encode(POSITIVE, add_special_tokens=False)[0]
    negative = tokenizer.encode(NEGATIVE, add_special_tokens=False)[0]
    if positive == negative:
        raise ValueError("Checker labels have the same first token")
    training, train_dropped = encode(train_cases, tokenizer, args.max_length)
    val, val_dropped = encode(val_cases, tokenizer, args.max_length)
    if not training or {label for _, label in val} != {0, 1}:
        raise ValueError("Training and both validation labels must survive the length filter")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    total_steps = math.ceil(len(training) / args.grad_accum) * args.epochs
    warmup = max(1, math.ceil(total_steps * .05))
    def learning_rate(step):
        if step < warmup:
            return (step + 1) / warmup
        return .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    amp = lambda: torch.autocast(device_type="cuda", dtype=torch.float16) if device == "cuda" else nullcontext()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    results = []
    for epoch in range(args.epochs):
        start = time.time()
        random.shuffle(training)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for index, (tokens, label) in enumerate(training):
            ids = torch.tensor([tokens], device=device)
            with amp():
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1).logits[:, -1, :]
                loss = two_way_loss(logits, positive, negative, torch.tensor([label], device=device))
            loss_sum += loss.item()
            group_start = (index // args.grad_accum) * args.grad_accum
            group_size = min(args.grad_accum, len(training) - group_start)
            scaler.scale(loss / group_size).backward()
            if (index + 1) % args.grad_accum == 0 or index + 1 == len(training):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        checkpoint = output / ("adapter" if epoch == args.epochs - 1 else f"adapter_ep{epoch + 1}")
        model.save_pretrained(str(checkpoint))
        tokenizer.save_pretrained(str(checkpoint))
        stats = validation(model, val, device, positive, negative, amp)
        stats.update(epoch=epoch + 1, train_loss=loss_sum / len(training), seconds=round(time.time() - start, 1))
        results.append(stats)
        print(json.dumps(stats), flush=True)
    report = {"name": args.name, "stand_in": args.stand_in, "train_rows": len(training), "val_rows": len(val),
              "train_dropped_for_length": train_dropped, "val_dropped_for_length": val_dropped,
              "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0,
              "epochs": results, "args": vars(args)}
    (output / "train_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(f"Required saved artifact is missing: {exc.filename or exc}") from exc
