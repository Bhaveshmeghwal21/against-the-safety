# Running on Kaggle

Every GPU step runs in a private Kaggle script kernel with two T4s, started from this repository with the Kaggle CLI.
Nothing needs a Hugging Face token: the weights come from Kaggle Models (`cohere-labs/tiny-aya/transformers/global/1`)
and `scripts/verify_model.py` / the smoke test check every shard against the hashes of the pinned Hugging Face revision
(`af89d219b53ed9b13b8a4645f9c8028973510324`).

## One-time setup

- A Kaggle account with phone verification (GPU and internet), and `~/.kaggle/kaggle.json` (the launcher reads it into its own
  process only and keeps the CLI's config folder inside the workspace).
- The Kaggle CLI: `python -m pip install --no-deps -r requirements-launcher.txt`.

## Commands

```powershell
. .\scripts\env.ps1
python scripts/kaggle_launch.py dataset                    # bundle code + exact data, upload, wait until Kaggle serves it
python scripts/kaggle_launch.py run smoke                  # environment, tokens, 10 QLoRA steps, batched generation, adapter reload
python scripts/kaggle_launch.py run train-both             # main on GPU 0 and control on GPU 1, 2 epochs each
python scripts/kaggle_launch.py status ats-train-both
python scripts/kaggle_launch.py output ats-train-both      # -> runs/kaggle/ats-train-both
python scripts/kaggle_launch.py run eval-dev  --adapters-from ats-train-both [--extra "--ckpt adapter_ep1 --systems main,control"]
python scripts/kaggle_launch.py run eval-test --adapters-from ats-train-both
python scripts/kaggle_launch.py run eval-product --adapters-from ats-train-both

# round 2 (plan sections 20 and 21)
python scripts/kaggle_launch.py run train-cands            # candA on GPU 0, candB on GPU 1
python scripts/kaggle_launch.py run eval-dev --adapters-from ats-train-cands --tag cands "--extra=--systems candA,candB"
python scripts/kaggle_launch.py run eval-test2 --adapters-from "ats-train-both,ats-train-cands" "--extra=--systems main,candA,base,control,closed"

# external benchmarks (plan sections 22 and 23); two kernels, so the tags keep their slugs apart
python scripts/kaggle_launch.py run eval-bench --adapters-from "ats-train-both,ats-train-cands" --tag a "--extra=--systems base,main"
python scripts/kaggle_launch.py run eval-bench --adapters-from "ats-train-both,ats-train-cands" --tag b "--extra=--systems candA,control,closed"
```

`--extra` is passed to the stage. For `eval-*`: `--ckpt` selects `adapter` (final, default) or `adapter_ep1`; `--systems` selects
from `base,main,closed,control` (also `candA` and `candB` where their adapters are attached); anything else goes to
`scripts/run_eval.py` (for example `--limit-questions 6`). `eval-bench` also takes `--benches` (default
`squad2,rgb,rgb_int,2wiki,musique`). Each system runs the benchmarks in turn on one GPU, and the systems are spread over the two GPUs.

## What the launcher guards against (all found the hard way)

- **Stale dataset mounts.** Kaggle reports a dataset "ready" while the new version is still processing. The bundle carries a content
  id (`BUNDLE_ID`); `dataset` waits until Kaggle serves it, `run` refuses to push otherwise, and the kernel aborts if the mounted
  id differs from the one it was launched for.
- **SQLite without FTS5.** The kernel's system Python (3.13) loses FTS5 once torch is imported, and AgentTune's search backend then
  returns "no results" instead of an error. The kernel builds a uv-managed Python 3.12 environment, and every evaluation script
  opens the index through `ragharness.retrieval.open_verified_backend` (hash check plus a probe search).
- **bf16 under fp16.** TRL casts the LoRA weights of a quantized model to bf16; the T4's fp16 GradScaler cannot unscale them.
  `ragharness.qlora` casts them back to float32 and logs the dtype histogram.

## Notes

- Python 3.12 venv, `torch==2.11.0` (default CUDA 13 wheel on driver 580+, otherwise the cu126 build), then
  `requirements-gpu.lock.txt` with `--no-deps`.
- Main and control train at the same time on separate GPUs through `CUDA_VISIBLE_DEVICES`. No DDP, no `device_map="auto"`.
- A kernel only keeps `/kaggle/working/out`; the repository and the environment live in `/kaggle/temp`.
- Measured on one T4 (smoke test): 8 s per training row at about 1,640 tokens, 6.5 GB peak; batched generation 1.8 s per episode (base)
  and 3.5 s (adapter), 9.3 GB peak.
- Use the same hardware, precision and decoding for every system in the final evaluation.
