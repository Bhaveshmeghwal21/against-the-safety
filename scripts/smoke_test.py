"""Two-hour GPU smoke test (plan section 7.1). Every stage is isolated: a failure is recorded and the rest still run.

  Kaggle T4:   python scripts/smoke_test.py --out runs/smoke
  CPU check:   python scripts/smoke_test.py --stand-in --out runs/smoke_cpu      (tiny public model, glue code only)

Stages
  env         nvidia-smi, driver, torch/CUDA, library versions, free memory
  tokens      exact Tiny Aya token counts for every training and dev row; strict prompt-prefix check; share over the cap
  fp16        FP16 forward pass is finite and agrees with an FP32 CPU run on a short prompt
  train       QLoRA through AgentTune's create_distill_trainer: longest rows first, then a random slice; memory and speed
  generate    base model and base+adapter through the rollout loop on dev questions; parse rate and latency
  reload      adapter saved, reloaded in a fresh model, outputs match
  vllm        vLLM can load and generate with the model in FP16 (separate process; failure is informative, not fatal)
Writes <out>/smoke_report.json.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "CohereLabs/tiny-aya-global"
REVISION = "af89d219b53ed9b13b8a4645f9c8028973510324"
STAND_IN = "hf-internal-testing/tiny-random-LlamaForCausalLM"
REPORT: dict = {"stages": {}}


def stage(name):
    def deco(fn):
        def run(*a, **k):
            t0 = time.time()
            try:
                out = fn(*a, **k)
                REPORT["stages"][name] = {"ok": True, "seconds": round(time.time() - t0, 1), **(out or {})}
            except Exception as e:  # noqa: BLE001 - the report must survive any failure
                REPORT["stages"][name] = {"ok": False, "seconds": round(time.time() - t0, 1), "error": f"{type(e).__name__}: {e}",
                                          "trace": traceback.format_exc()[-1500:]}
            print(f"[{name}]", json.dumps({k: v for k, v in REPORT["stages"][name].items() if k != "trace"}, default=str)[:600], flush=True)
            return REPORT["stages"][name]
        return run
    return deco


def read_rows(name: str) -> list[dict]:
    p = ROOT / "data" / "processed" / "training" / f"{name}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]


@stage("env")
def env_stage(args):
    import torch, transformers

    info = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
            "cuda_available": torch.cuda.is_available(), "cuda": torch.version.cuda}
    for mod in ("trl", "peft", "bitsandbytes", "accelerate", "agenttune", "auditkit", "vllm"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception as e:  # noqa: BLE001
            info[mod] = f"unavailable ({type(e).__name__})"
    try:
        info["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"], text=True).strip().splitlines()
    except Exception as e:  # noqa: BLE001
        info["nvidia_smi"] = f"unavailable ({type(e).__name__})"
    if torch.cuda.is_available():
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
    return info


@stage("tokens")
def tokens_stage(args, tok):
    from ragharness import rows as rowmod

    out = {"counted": {}}
    cap = args.cap_tokens
    for name in ("main", "control", "dev"):
        rows = read_rows(name)
        totals, sups, bad = [], [], 0
        for r in rows:
            try:
                t, s = rowmod.real_tokens(r, tok, strict=not args.stand_in)
            except ValueError:
                bad += 1
                continue
            totals.append(t), sups.append(s)
        totals.sort()
        out["counted"][name] = {
            "rows": len(rows), "prefix_violations": bad, "processed_total": sum(totals), "supervised_total": sum(sups),
            "p50": totals[len(totals) // 2], "p95": totals[int(0.95 * len(totals))], "max": totals[-1],
            "share_over_cap": round(sum(t > cap for t in totals) / len(totals), 4),
        }
    out["cap"] = cap
    return out


@stage("fp16")
def fp16_stage(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        return {"skipped": "no CUDA"}
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    text = tok.apply_chat_template([{"role": "user", "content": "Name the capital of France."}], add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    m16 = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float16, device_map={"": 0})
    with torch.no_grad():
        l16 = m16(input_ids=ids.to(0)).logits[0, -1].float().cpu()
    finite = bool(torch.isfinite(l16).all())
    del m16
    torch.cuda.empty_cache()
    m32 = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float32)
    with torch.no_grad():
        l32 = m32(input_ids=ids).logits[0, -1].float()
    del m32
    lp16, lp32 = l16.log_softmax(-1), l32.log_softmax(-1)
    top16, top32 = lp16.topk(5).indices.tolist(), lp32.topk(5).indices.tolist()
    return {"finite": finite, "max_abs_logprob_diff_top5": round(float((lp16[top32] - lp32[top32]).abs().max()), 4),
            "top1_agrees": top16[0] == top32[0], "top5_overlap": len(set(top16) & set(top32)), "prompt_tokens": int(ids.shape[1])}


def pick_smoke_keys(rows: list[dict], n_longest: int, n_random: int, seed: int = 13) -> set:
    from ragharness import training

    ordered = sorted(rows, key=lambda r: -r["meta"]["est_tokens"])
    longest = ordered[:n_longest]
    rest = ordered[n_longest:]
    random.Random(seed).shuffle(rest)
    return {training.row_key(r["meta"]) for r in longest + rest[:n_random]}


@stage("train")
def train_stage(args, out: Path):
    import torch
    from agenttune.agentic.distill_trainer import create_distill_trainer

    from ragharness import mixture, qlora, training

    rows = mixture.annotate(read_rows("main"))
    keys = pick_smoke_keys(rows, args.longest, args.train_rows)
    logs, metas = training.load_episodes(ROOT / "data" / "processed" / "episodes", "train")
    logs, metas = training.episodes_for_keys(logs, metas, keys)
    system = training.read_system(ROOT / "data" / "processed" / "training" / "main.jsonl")
    cuda = torch.cuda.is_available() and not args.stand_in
    template = (ROOT / "tests" / "fixtures" / "tiny_aya_global_chat_template.jinja").read_text(encoding="utf-8") if args.stand_in else None
    factory = qlora.QLoraFactory(
        system, quantize=cuda, dtype="float16" if cuda else "float32", revision=None if args.stand_in else REVISION,
        lora_targets="all-linear" if cuda else ["q_proj", "v_proj"], max_length=args.cap_tokens, use_cpu=not cuda,
        gradient_checkpointing=cuda, chat_template=template, metas=metas, keep=keys,
    )
    kwargs = dict(output_dir=str(out / "train"), max_steps=args.steps, per_device_train_batch_size=1, gradient_accumulation_steps=args.grad_accum,
                  learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=0.0, seed=13, weight_decay=0.0, max_grad_norm=1.0)
    if cuda:
        kwargs["optim"] = "paged_adamw_8bit"
        torch.cuda.reset_peak_memory_stats()
    trainer = create_distill_trainer(STAND_IN if args.stand_in else MODEL, trajectories=logs, trainer_factory=factory, **kwargs)
    t0 = time.time()
    result = trainer.train()
    wall = time.time() - t0
    ds = factory.sft_trainer.train_dataset
    tokens = int(sum(len(x) for x in ds["input_ids"]))
    longest = max(len(x) for x in ds["input_ids"])
    truncated = sum(len(x) >= args.cap_tokens for x in ds["input_ids"])  # TRL truncates from the left at max_length
    seen = args.steps * args.grad_accum
    adapter = out / "adapter"
    factory.sft_trainer.model.save_pretrained(str(adapter))
    factory.tokenizer.save_pretrained(str(adapter))
    return {
        "rows_selected": len(keys), "rows_trained": factory.n_train_rows, "optimizer_steps": args.steps, "micro_batches": seen,
        "longest_row_tokens": longest, "rows_at_or_over_max_length": truncated, "mean_row_tokens": round(tokens / len(ds), 1),
        "wall_seconds": round(wall, 1), "seconds_per_micro_batch": round(wall / seen, 2),
        "tokens_per_second": round(tokens / len(ds) * seen / wall, 1),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if cuda else None,
        "train_loss": getattr(result, "training_loss", None),
        "loss_finite": bool(getattr(result, "training_loss", 0) == getattr(result, "training_loss", 0)),
    }


def load_generation_model(args, adapter: Path | None):
    from ragharness import generate

    cuda = not args.stand_in
    return generate.load_model(STAND_IN if args.stand_in else MODEL, None if args.stand_in else REVISION,
                               adapter=str(adapter) if adapter else None, quantize=cuda, dtype="float16" if cuda else "float32")


@stage("generate")
def generate_stage(args, out: Path):
    from agenttune.rag.retrieval.sqlite_fts import SQLiteFTSBackend

    from ragharness import corpus as corpus_mod, data, evaluation, generate, protocol, rollout, rows as rowmod
    from ragharness.retrieval import MaskedSearch
    from ragharness.tools import CorpusSearchTool

    man = json.loads((ROOT / "data" / "manifests" / "split_manifest.json").read_text(encoding="utf-8"))
    splits = {n: data.read_questions(ROOT / "data" / "processed" / f"questions_{n}.jsonl") for n in ("train", "dev", "test")}
    corp = corpus_mod.build_corpus([q for n in splits for q in splits[n]], seed=man["seed"])
    backend = SQLiteFTSBackend(str(ROOT / "data" / "index" / "corpus.sqlite"), match_all=False)
    tool, search = CorpusSearchTool(backend, top_k=3), MaskedSearch(backend)
    system = protocol.system_prompt(tool)
    jobs = evaluation.make_jobs(splits["dev"][: args.gen_questions], "dev", corp, tool, search, extra_questions=0)
    template = rowmod.FIXTURE.read_text(encoding="utf-8") if args.stand_in else None
    result = {}
    for label, adapter in (("base", None), ("adapter", out / "adapter")):
        if adapter is not None and not adapter.exists():
            result[label] = {"skipped": "no adapter (train stage failed)"}
            continue
        model, tok = load_generation_model(args, adapter)
        gen = generate.HFGenerator(model, tok, chat_template=template)
        t0 = time.time()
        res = evaluation.run_jobs(jobs, corp, tool, system, lambda job: gen, 3)
        scored = evaluation.score_results(res, corp, None, label, label)["summary"]
        kinds = [ep.final.turn.kind for _, ep, _ in res]
        result[label] = {
            "episodes": len(res), "final_kinds": {k: kinds.count(k) for k in set(kinds)}, "headline": scored["headline"],
            "mean_seconds_per_episode": round((time.time() - t0) / len(res), 2),
            "mean_model_calls": scored["cost"]["mean_model_calls"], "mean_output_tokens": scored["cost"]["mean_output_tokens"],
            "first_outputs": [res[i][1].steps[0].text[:160] for i in range(min(3, len(res)))],
        }
        del model
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    return result


@stage("reload")
def reload_stage(args, out: Path):
    from ragharness import generate, rows as rowmod

    adapter = out / "adapter"
    if not adapter.exists():
        return {"skipped": "no adapter"}
    template = rowmod.FIXTURE.read_text(encoding="utf-8") if args.stand_in else None
    prompt = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Who wrote Hamlet?"}]
    outs = []
    for _ in range(2):  # two fresh loads must produce the same greedy output
        model, tok = load_generation_model(args, adapter)
        outs.append(generate.HFGenerator(model, tok, max_new_tokens=24, chat_template=template)(prompt))
        del model
    return {"identical_outputs": outs[0] == outs[1], "sample": outs[0][:120]}


VLLM_CHILD = """
import json, sys, time
from vllm import LLM, SamplingParams
t0 = time.time()
llm = LLM(model=sys.argv[1], revision=sys.argv[2], dtype="float16", max_model_len=4096, gpu_memory_utilization=0.85, enforce_eager=True)
load = time.time() - t0
t1 = time.time()
out = llm.generate(["Name the capital of France."] * 8, SamplingParams(temperature=0, max_tokens=32))
print(json.dumps({"load_seconds": round(load, 1), "gen_seconds_8_prompts": round(time.time() - t1, 2), "sample": out[0].outputs[0].text[:80]}))
"""


@stage("vllm")
def vllm_stage(args):
    if args.stand_in:
        return {"skipped": "stand-in mode"}
    p = subprocess.run([sys.executable, "-c", VLLM_CHILD, MODEL, REVISION], capture_output=True, text=True, timeout=1500)
    last = (p.stdout.strip().splitlines() or [""])[-1]
    if p.returncode != 0 or not last.startswith("{"):
        return {"ok_child": False, "returncode": p.returncode, "stderr_tail": p.stderr[-900:]}
    return json.loads(last)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/smoke")
    ap.add_argument("--stand-in", action="store_true", help="CPU glue-code check with a tiny public model")
    ap.add_argument("--cap-tokens", type=int, default=3072)
    ap.add_argument("--longest", type=int, default=6)
    ap.add_argument("--train-rows", type=int, default=24)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--gen-questions", type=int, default=8)
    ap.add_argument("--skip", default="", help="comma-separated stage names to skip")
    args = ap.parse_args()
    skip = set(args.skip.split(","))
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    REPORT["args"] = vars(args)

    from transformers import AutoTokenizer

    from ragharness import rows as rowmod

    env_stage(args)
    tok = AutoTokenizer.from_pretrained(STAND_IN) if args.stand_in else AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    if args.stand_in:
        rowmod.fetch_template() if not rowmod.FIXTURE.exists() else None
        tok.chat_template = rowmod.FIXTURE.read_text(encoding="utf-8")
    plan = [
        ("tokens", lambda: tokens_stage(args, tok)), ("fp16", lambda: fp16_stage(args)), ("train", lambda: train_stage(args, out)),
        ("generate", lambda: generate_stage(args, out)), ("reload", lambda: reload_stage(args, out)), ("vllm", lambda: vllm_stage(args)),
    ]
    for name, fn in plan:
        if name not in skip:
            fn()
    REPORT["all_ok"] = all(s.get("ok") for s in REPORT["stages"].values())
    (out / "smoke_report.json").write_text(json.dumps(REPORT, indent=2, default=str), encoding="utf-8")
    print("ALL STAGES OK" if REPORT["all_ok"] else "SOME STAGES FAILED: " + ", ".join(k for k, v in REPORT["stages"].items() if not v.get("ok")))


if __name__ == "__main__":
    main()
