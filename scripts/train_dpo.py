"""Round 5, priority 2: preference training of Tiny Aya through AgentTune's TrlAgenticDPO (plan section 27).

  python scripts/train_dpo.py --init-adapter <adapter dir> --pairs data/processed/training/dpo_r5.jsonl --out runs/candG --name candG

LoRA continues from `--init-adapter` (4-bit NF4, fp16 compute). TRL keeps a frozen copy of the starting adapter named `ref` as the reference policy,
so the reference is the starting adapter, not the bare base. Loss: sigmoid DPO + `--sft-weight` x NLL of the chosen text.
Before training, on a GPU, a memory probe picks the longest sequence length that fits and a timing probe sets how many pairs fit the time budget;
both decisions are written to train_report.json. `--stand-in` runs the same code on CPU with a tiny public model (plumbing check only).
"""

import argparse
import json
import os
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


def load_pairs(path: Path, limit: int) -> list[dict]:
    pairs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pairs[:limit] if limit else pairs


def _probe_step(model, length: int) -> None:
    """One worst-case DPO step in miniature: a pair (2 sequences) forward with a graph, a reference forward without one, and the backward."""
    import torch
    from trl.trainer.utils import selective_log_softmax

    ids = torch.randint(1000, 9000, (2, length), device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        logp = selective_log_softmax(logits[:, :-1], ids[:, 1:])
        with torch.no_grad():
            ref = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
            ref_logp = selective_log_softmax(ref[:, :-1], ids[:, 1:])
    loss = -logp[:, -60:].sum() + 0.0 * ref_logp.sum()
    loss.backward()


def probe(model, length: int):
    """(ok, peak GiB, seconds) of one probe step at `length`; an out-of-memory error is reported as not ok, and memory is released either way."""
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0, ok = time.time(), True
    try:
        _probe_step(model, length)
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        ok = False
    seconds = time.time() - t0
    model.zero_grad(set_to_none=True)
    gc.collect()  # the exception's traceback has been cleared by now, so the failed attempt's tensors can go
    torch.cuda.empty_cache()
    return ok, torch.cuda.max_memory_allocated() / 2**30, seconds


def probe_memory(model, lengths, budget_gb: float):
    """Largest length whose probe step succeeds with a peak under budget_gb."""
    model.train()
    for length in lengths:
        ok, peak, _ = probe(model, length)
        print(f"[probe] length {length}: ok={ok} peak={peak:.2f} GiB", flush=True)
        if ok and peak <= budget_gb:
            return length, round(peak, 2)
    raise SystemExit("no tested sequence length fits in GPU memory")


def probe_seconds_per_pair(model, length: int, repeats: int = 3) -> float:
    """Wall seconds for one pair at `length` (policy forward and backward plus the reference forward): mean of the last `repeats` of repeats + 1 steps."""
    model.train()
    times = [probe(model, length)[2] for _ in range(repeats + 1)]
    return sum(times[1:]) / repeats

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-adapter", default=None)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="candG")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--sft-weight", type=float, default=0.5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2560)
    ap.add_argument("--seed", type=int, default=53)
    ap.add_argument("--limit-pairs", type=int, default=0)
    ap.add_argument("--time-budget-h", type=float, default=1.7, help="training seconds the probe may spend on pairs (GPU only)")
    ap.add_argument("--memory-gb", type=float, default=13.5)
    ap.add_argument("--stand-in", action="store_true", help="CPU plumbing check with a tiny public model")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback

    from agenttune.backends.trl.agentic.dpo.agentic_dpo import TrlAgenticDPO
    from ragharness import modelref, qlora, rows

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    stand_in = args.stand_in
    if not stand_in:
        assert torch.cuda.is_available(), "this script trains on a CUDA GPU (use --stand-in for the CPU check)"
    pairs = load_pairs(Path(args.pairs), args.limit_pairs)
    decisions: dict = {"pairs_in_file": len(pairs)}

    # ---- model, tokenizer, starting adapter -------------------------------------------------------------------------
    if stand_in:
        tok = AutoTokenizer.from_pretrained(STAND_IN)
        tok.chat_template = rows.FIXTURE.read_text(encoding="utf-8")
        model = AutoModelForCausalLM.from_pretrained(STAND_IN, dtype=torch.float32)
        init_dir = out / "stand_in_init"
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        model.save_pretrained(str(init_dir))
        base = AutoModelForCausalLM.from_pretrained(STAND_IN, dtype=torch.float32)
        model = PeftModel.from_pretrained(base, str(init_dir), is_trainable=True)
        init_adapter = str(init_dir)
        model_source, fp16 = "stand-in", False
    else:
        model_path, revision, model_source = modelref.resolve()
        init_adapter = args.init_adapter
        tok = AutoTokenizer.from_pretrained(init_adapter)
        dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_path, revision=revision, dtype=dtype, device_map={"": 0},
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True),
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
        fp16 = True
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for n, p in model.named_parameters():  # trainable adapter weights in float32 (fp16 AMP's GradScaler cannot unscale bf16 gradients)
        if "lora_" in n:
            p.data = p.data.to(torch.float32)

    # ---- probes (GPU only): sequence length that fits, pairs that fit the time budget ---------------------------------------
    max_length = args.max_length
    if not stand_in:
        lengths = [l for l in (2560, 2304, 2048, 1792, 1536, 1280) if l <= args.max_length]
        max_length, peak = probe_memory(model, lengths, args.memory_gb)
        eligible = [p for p in pairs if p["meta"]["tokens"] <= max_length]
        mean_tokens = int(sum(p["meta"]["tokens"] for p in eligible) / max(1, len(eligible)))
        seconds = probe_seconds_per_pair(model, mean_tokens)
        fit = int(args.time_budget_h * 3600 / (seconds * max(1.0, args.epochs)))
        decisions.update(max_length=max_length, probe_peak_gb=peak, pairs_within_length=len(eligible), mean_pair_tokens=mean_tokens,
                         seconds_per_pair=round(seconds, 2), pairs_fitting_budget=fit)
        pairs = eligible[:fit]
        print("[probe]", json.dumps(decisions), flush=True)
    decisions["pairs_trained"] = len(pairs)
    assert len(pairs) >= 8, "too few pairs"
    ds = Dataset.from_list([{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]} for p in pairs])
    steps_per_epoch = max(1, len(pairs) // args.grad_accum)
    total_steps = int(steps_per_epoch * args.epochs)
    half_step = max(1, total_steps // 2)

    class Snapshots(TrainerCallback):
        def _save(self, model, tokenizer, name):
            path = out / name
            model.save_pretrained(str(path), selected_adapters=["default"])
            tokenizer.save_pretrained(str(path))
            print(f"saved {name} at step", flush=True)

        def on_step_end(self, a, state, control, model=None, processing_class=None, **kw):
            if state.global_step == half_step and total_steps > 2:
                self._save(model, processing_class, "adapter_half")

    sink: dict = {}
    agent = TrlAgenticDPO(
        model=model, processing_class=tok, train_dataset=ds, output_dir=str(out),
        num_train_epochs=args.epochs, per_device_train_batch_size=1, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=max(1, total_steps // 20), seed=args.seed,
        beta=args.beta, max_length=max_length, loss_type=["sigmoid", "sft"], loss_weights=[1.0, args.sft_weight],
        fp16=fp16, bf16=False, gradient_checkpointing=not stand_in, optim="adamw_torch" if stand_in else "paged_adamw_8bit",
        weight_decay=0.0, max_grad_norm=1.0, report_to="none", save_strategy="no", logging_steps=1, use_cpu=stand_in,
        callbacks=[Snapshots(), qlora.make_dtype_guard(sink, cast=fp16)],
    )
    agent.setup_data()
    agent.setup_trainer()
    trainer = agent.trainer
    cfg = trainer.args
    assert cfg.loss_type == ["sigmoid", "sft"] and list(cfg.loss_weights) == [1.0, args.sft_weight], (cfg.loss_type, cfg.loss_weights)
    assert cfg.beta == args.beta and cfg.max_length == max_length and abs(cfg.learning_rate - args.lr) < 1e-12, "a requested setting was dropped"
    pm = trainer.model

    # TRL cast the trainable weights to bf16 and froze nothing explicitly: restore exact float32 copies of the starting adapter
    # in both the trainable adapter and the frozen reference, then freeze the reference.
    state = {k: v.to(torch.float32) for k, v in load_file(str(Path(init_adapter) / "adapter_model.safetensors")).items()}
    for n, p in pm.named_parameters():
        if "lora_" in n:
            p.data = p.data.to(torch.float32)
    for adapter in ("default", "ref"):
        if adapter in pm.peft_config:
            set_peft_model_state_dict(pm, state, adapter_name=adapter)
    for n, p in pm.named_parameters():
        if ".ref." in n:
            p.requires_grad_(False)
    trainable = [n for n, p in pm.named_parameters() if p.requires_grad]
    assert trainable and all(".default." in n for n in trainable), "only the default adapter may be trainable"
    assert "ref" in pm.peft_config, "the reference adapter was not created"
    ref_before = {n: p.detach().clone() for n, p in pm.named_parameters() if ".ref." in n}
    default_before = {n: p.detach().clone() for n, p in pm.named_parameters() if ".default." in n and p.requires_grad}
    print(f"[setup] trainable tensors {len(trainable)}, reference tensors {len(ref_before)}, pairs {len(pairs)}, steps {total_steps} (half at {half_step})", flush=True)

    t0 = time.time()
    if not stand_in:
        torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    wall = time.time() - t0

    pm.save_pretrained(str(out / "adapter"), selected_adapters=["default"])
    tok.save_pretrained(str(out / "adapter"))
    unchanged_ref = all(torch.equal(p.detach(), ref_before[n]) for n, p in pm.named_parameters() if ".ref." in n)
    moved = sum(float((p.detach() - default_before[n]).abs().max()) for n, p in pm.named_parameters() if n in default_before)
    log = [h for h in trainer.state.log_history if "loss" in h]
    keys = ("loss", "rewards/accuracies", "rewards/margins", "rewards/chosen", "rewards/rejected", "logps/chosen", "logps/rejected")

    def window(rows_, a, b):
        sel = rows_[a:b]
        return {k: round(sum(r[k] for r in sel if k in r) / max(1, sum(k in r for r in sel)), 4) for k in keys}

    import peft, transformers, trl

    report = {
        "name": args.name, "trainer": "agenttune.backends.trl.agentic.dpo.TrlAgenticDPO", "model": MODEL, "revision": REVISION, "model_source": model_source,
        "init_adapter": str(init_adapter), "git_commit": git_commit(), "decisions": decisions, "steps": getattr(result, "global_step", None),
        "wall_seconds": round(wall, 1), "reference_unchanged": unchanged_ref, "max_abs_adapter_change": moved,
        "peak_gpu_memory_gb": None if stand_in else round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "first_steps": window(log, 0, 5), "last_steps": window(log, -5, None), "train_loss": getattr(result, "training_loss", None),
        "dtype_guard": sink, "args": vars(args),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "trl": trl.__version__, "peft": peft.__version__},
    }
    (out / "train_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("steps", "wall_seconds", "reference_unchanged", "max_abs_adapter_change", "first_steps", "last_steps")}, indent=2))
    assert unchanged_ref, "the frozen reference adapter changed during training"


if __name__ == "__main__":
    main()
