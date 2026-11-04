"""QLoRA SFT of Tiny Aya through AgentTune's create_distill_trainer, on one GPU.

  python scripts/train.py --set main    --out runs/main
  python scripts/train.py --set control --out runs/control

Needs: data/processed/{episodes,training} (python scripts/prepare_data.py, build_index.py, build_episodes.py,
make_training_set.py), an HF token that has accepted the Tiny Aya gate (HF_TOKEN), and a CUDA GPU.
The main and control runs use the same model revision, LoRA config, optimizer settings and length policy.
Writes <out>/adapter, <out>/train_report.json (real token counts, steps, wall time, peak memory, versions).
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "CohereLabs/tiny-aya-global"
REVISION = "af89d219b53ed9b13b8a4645f9c8028973510324"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


class EpochSnapshots:
    """Saves the adapter at the end of every epoch but the last (adapter_ep1, ...), so the epoch count can be chosen on dev."""

    def __new__(cls, out, total):
        from transformers import TrainerCallback

        class _Cb(TrainerCallback):
            def on_epoch_end(self, args, state, control, model=None, **kwargs):
                ep = int(round(state.epoch))
                if ep < total:
                    model.save_pretrained(str(out / f"adapter_ep{ep}"))
                    kwargs["processing_class"].save_pretrained(str(out / f"adapter_ep{ep}"))
                    print(f"saved adapter_ep{ep} at step {state.global_step}", flush=True)

        return _Cb()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["main", "control"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=2560)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--limit-rows", type=int, default=0, help="debug: keep only the first N selected rows")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    import torch
    from agenttune.agentic.distill_trainer import create_distill_trainer

    from ragharness import modelref, qlora, training

    assert torch.cuda.is_available(), "this script trains on a CUDA GPU"
    model_path, model_revision, model_source = modelref.resolve()
    print("model source:", model_source)
    torch.manual_seed(args.seed)
    ep_dir = ROOT / "data" / "processed" / "episodes"
    set_path = ROOT / "data" / "processed" / "training" / f"{args.set}.jsonl"
    keys = training.read_keys(set_path)
    if args.limit_rows:
        keys = set(sorted(keys)[: args.limit_rows])
    logs, metas = training.load_episodes(ep_dir, "train")
    logs, metas = training.episodes_for_keys(logs, metas, keys)
    system = training.read_system(ROOT / "data" / "processed" / "training" / "main.jsonl")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    factory = qlora.QLoraFactory(
        system, quantize=True, dtype="float16", lora_r=args.lora_r, lora_alpha=2 * args.lora_r, revision=model_revision,
        max_length=args.max_length, metas=metas, keep=keys,
        callbacks=[EpochSnapshots(out, int(args.epochs))] if float(args.epochs).is_integer() and args.epochs > 1 else [],
    )
    trainer = create_distill_trainer(
        model_path, trajectories=logs, trainer_factory=factory, output_dir=str(out),
        num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=0.03,
        per_device_train_batch_size=1, gradient_accumulation_steps=args.grad_accum, seed=args.seed,
        optim="paged_adamw_8bit", weight_decay=0.0, max_grad_norm=1.0,
    )
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    wall = time.time() - t0

    sft = factory.sft_trainer
    ds = sft.train_dataset
    sft.model.save_pretrained(str(out / "adapter"))  # save first: nothing below may lose the trained adapter
    factory.tokenizer.save_pretrained(str(out / "adapter"))
    processed = int(sum(len(x) for x in ds["input_ids"]))
    supervised = int(sum(sum(m) for m in ds["completion_mask"]))

    import peft, transformers, trl

    report = {
        "set": args.set, "model": MODEL, "revision": REVISION, "model_source": model_source, "git_commit": git_commit(),
        "rows_selected": len(keys), "rows_trained": factory.n_train_rows, "episodes": len(logs),
        "processed_tokens": processed, "supervised_tokens": supervised,
        "wall_seconds": round(wall, 1), "processed_tokens_per_second": round(processed * args.epochs / wall, 1),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "optimizer_steps": getattr(result, "global_step", None),
        "dtype_info": factory.dtype_info,
        "train_loss": getattr(result, "training_loss", None) or getattr(result, "metrics", {}).get("train_loss"),
        "args": vars(args),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "trl": trl.__version__, "peft": peft.__version__,
                     "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
    }
    (out / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
