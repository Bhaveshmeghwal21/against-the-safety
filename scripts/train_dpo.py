"""Round 5, priority 2: preference training of Tiny Aya on the pair file (research/2026-11-07-round5-preregistration.md).

  python scripts/train_dpo.py --init-adapter <adapter dir> --pairs data/processed/training/dpo_r5.jsonl --out runs/candG --name candG

LoRA continues from `--init-adapter` (4-bit NF4, fp16 compute, adapter weights in float32). The reference policy is that starting adapter: its
log-probabilities are computed once, before the first update. Loss per pair, as in TRL's DPOTrainer ("sigmoid" plus "sft" terms):
    -logsigmoid(beta * ((pi_c - pi_r) - (ref_c - ref_r))) + sft_weight * mean token NLL of the chosen text.

Why this is a small loop and not AgentTune's TrlAgenticDPO: that class wraps TRL 1.7.1's DPOTrainer, which materialises the full-vocabulary
logits (262k) of both sequences of a pair. On a 15 GB T4 that peaked at 12.8 GB for 1,280 tokens and ran out of memory for longer pairs, so only 38
of the 335 pairs fit (kernel `ats-train-dpo-r5b`). Here only the logits of the completion positions are computed (`logits_to_keep`), which makes
the memory independent of the prompt length. Everything else (tokenisation with the pinned chat template, the loss, beta, the schedule) follows the
registration. `--stand-in` runs the same code on CPU with a tiny public model (plumbing check only).
"""

import argparse
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
MODEL = "CohereLabs/tiny-aya-global"
REVISION = "af89d219b53ed9b13b8a4645f9c8028973510324"
STAND_IN = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def encode(pairs, tok, max_length, strict=True):
    """(prompt ids, chosen completion ids, rejected completion ids) with the pinned chat template; the completion ends with the end-of-turn markers."""
    out, dropped = [], 0
    for p in pairs:
        prompt = list(tok.apply_chat_template(p["prompt"], add_generation_prompt=True, return_dict=True)["input_ids"])
        comps = []
        for key in ("chosen", "rejected"):
            if strict:
                full = list(tok.apply_chat_template(p["prompt"] + p[key], return_dict=True)["input_ids"])
                assert full[: len(prompt)] == prompt, "prompt tokens are not a prefix of prompt + completion tokens"
                comps.append(full[len(prompt):])
            else:  # stand-in tokenizer: the markers are plain text there, so tokenise the completion on its own
                comps.append(list(tok.encode(p[key][0]["content"] + "<|END_RESPONSE|><|END_OF_TURN_TOKEN|>", add_special_tokens=False)))
        if len(prompt) + max(len(c) for c in comps) > max_length:
            dropped += 1
            continue
        out.append((prompt, comps[0], comps[1]))
    return out, dropped


def completion_logps(model, prompt, comp, device):
    """Per-token log-probabilities of `comp` after `prompt`. Only the last len(comp) + 1 positions get logits."""
    import torch

    ids = torch.tensor([prompt + comp], device=device)
    k = len(comp) + 1
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=k).logits[0, :-1].float()
    targets = ids[0, -len(comp):]
    return torch.log_softmax(logits, dim=-1).gather(-1, targets[:, None])[:, 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-adapter", default=None)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="candG")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--sft-weight", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2560)
    ap.add_argument("--seed", type=int, default=53)
    ap.add_argument("--limit-pairs", type=int, default=0)
    ap.add_argument("--stand-in", action="store_true", help="CPU plumbing check with a tiny public model")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from ragharness import modelref, rows

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    stand_in = args.stand_in
    if not stand_in:
        assert torch.cuda.is_available(), "this script trains on a CUDA GPU (use --stand-in for the CPU check)"
    pairs = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]
    pairs = pairs[: args.limit_pairs] if args.limit_pairs else pairs

    # ---- model, tokenizer, starting adapter ----------------------------------------------------------------------------
    if stand_in:
        tok = AutoTokenizer.from_pretrained(STAND_IN)
        tok.chat_template = rows.FIXTURE.read_text(encoding="utf-8")
        init_dir = out / "stand_in_init"
        m0 = get_peft_model(AutoModelForCausalLM.from_pretrained(STAND_IN, dtype=torch.float32),
                            LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        m0.save_pretrained(str(init_dir))
        model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(STAND_IN, dtype=torch.float32), str(init_dir), is_trainable=True)
        init_adapter, model_source, fp16, device = str(init_dir), "stand-in", False, "cpu"
    else:
        model_path, revision, model_source = modelref.resolve()
        init_adapter, fp16, device = args.init_adapter, True, "cuda"
        tok = AutoTokenizer.from_pretrained(init_adapter)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, revision=revision, dtype=torch.float16, device_map={"": 0},
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.data = p.data.to(torch.float32)  # trainable adapter weights in float32
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0  # DPO uses no dropout, so the policy and the reference see identical networks at the start
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n, p in model.named_parameters() if p.requires_grad), "only adapter weights may train"

    encoded, dropped_long = encode(pairs, tok, args.max_length, strict=not stand_in)
    n = len(encoded)
    assert n >= 8, "too few pairs"
    steps_per_epoch = math.ceil(n / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    half_step = total_steps // 2
    warmup = max(1, total_steps // 20)
    amp = torch.autocast(device_type="cuda", dtype=torch.float16) if fp16 else torch.autocast(device_type="cpu", enabled=False)
    print(f"[setup] pairs {n} (dropped for length {dropped_long}), trainable tensors {len(trainable)}, steps {total_steps} (snapshot at {half_step})", flush=True)

    # ---- reference log-probabilities from the starting adapter (before any update) ------------------------------------------
    model.eval()
    t0 = time.time()
    ref = []
    with torch.no_grad(), amp:
        for prompt, c, r in encoded:
            ref.append((float(completion_logps(model, prompt, c, device).sum()), float(completion_logps(model, prompt, r, device).sum())))
    print(f"[reference] {n} pairs in {time.time() - t0:.0f}s", flush=True)
    init_state = {k: v.detach().clone() for k, v in model.named_parameters() if v.requires_grad}

    # ---- training ----------------------------------------------------------------------------------------------------------
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    model.train()
    log, step, t_train = [], 0, time.time()

    def save(name):
        model.save_pretrained(str(out / name))
        tok.save_pretrained(str(out / name))
        print(f"[saved] {name} at step {step}", flush=True)

    for epoch in range(args.epochs):
        order = list(range(n))
        random.Random(args.seed + epoch).shuffle(order)
        for g in range(0, n, args.grad_accum):
            group = order[g:g + args.grad_accum]
            agg = {"loss": 0.0, "dpo": 0.0, "sft": 0.0, "margin": 0.0, "acc": 0.0, "chosen_logratio": 0.0, "rejected_logratio": 0.0}
            for i in group:
                prompt, c, r = encoded[i]
                with amp:
                    lc, lr_ = completion_logps(model, prompt, c, device), completion_logps(model, prompt, r, device)
                logits_diff = (lc.sum() - lr_.sum()) - (ref[i][0] - ref[i][1])
                dpo = -F.logsigmoid(args.beta * logits_diff)
                sft = -lc.mean()
                loss = (dpo + args.sft_weight * sft) / len(group)
                scaler.scale(loss).backward()
                with torch.no_grad():
                    agg["loss"] += float(dpo + args.sft_weight * sft) / len(group)
                    agg["dpo"] += float(dpo) / len(group)
                    agg["sft"] += float(sft) / len(group)
                    agg["margin"] += float(args.beta * logits_diff) / len(group)
                    agg["acc"] += float(logits_diff > 0) / len(group)
                    agg["chosen_logratio"] += float(lc.sum() - ref[i][0]) / len(group)
                    agg["rejected_logratio"] += float(lr_.sum() - ref[i][1]) / len(group)
            scaler.unscale_(opt)
            gn = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            agg.update(step=step, epoch=epoch + 1, grad_norm=round(gn, 4), lr=sched.get_last_lr()[0], seconds=round(time.time() - t_train, 1))
            log.append({k: (round(v, 5) if isinstance(v, float) else v) for k, v in agg.items()})
            print("[step]", json.dumps(log[-1]), flush=True)
            if step == half_step and total_steps > 2:
                save("adapter_half")
    save("adapter")
    wall = time.time() - t_train

    moved = sum(float((p.detach() - init_state[k]).abs().max()) for k, p in model.named_parameters() if k in init_state)
    window = lambda a, b: {k: round(sum(r[k] for r in log[a:b]) / max(1, len(log[a:b])), 4) for k in ("loss", "dpo", "sft", "margin", "acc", "chosen_logratio", "rejected_logratio")}
    import peft, transformers

    report = {
        "name": args.name, "trainer": "custom DPO loop (TRL sigmoid + sft loss terms, completion-position logits only)", "model": MODEL, "revision": REVISION,
        "model_source": model_source, "init_adapter": str(init_adapter), "git_commit": git_commit(), "pairs_in_file": len(pairs), "pairs_trained": n,
        "dropped_for_length": dropped_long, "steps": step, "wall_seconds": round(wall, 1), "seconds_per_pair_pass": round(wall / (n * args.epochs), 2),
        "max_abs_adapter_change": moved, "first_steps": window(0, 5), "last_steps": window(-5, None),
        "peak_gpu_memory_gb": None if stand_in else round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "args": vars(args), "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
    }
    (out / "train_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "log_history.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("pairs_trained", "steps", "wall_seconds", "seconds_per_pair_pass", "max_abs_adapter_change", "first_steps", "last_steps", "peak_gpu_memory_gb")}, indent=2))


if __name__ == "__main__":
    main()
