# Running on Kaggle

Every GPU step runs as a private Kaggle script kernel with two T4s, started from this repository by `scripts/kaggle_launch.py`. No Hugging Face token is needed: the weights come from Kaggle Models (`cohere-labs/tiny-aya/transformers/global/1`), and the smoke test checks every shard against the hashes of the pinned Hugging Face revision (`af89d219b53ed9b13b8a4645f9c8028973510324`).

## Setup

- A Kaggle account with phone verification (GPU and internet) and `~/.kaggle/kaggle.json`. The launcher reads it into its own process and keeps the CLI's config folder inside the workspace. For a second account, pass `--credentials-env <VARIABLE>` and keep the credentials out of reports.
- The Kaggle CLI: `python -m pip install --no-deps -r requirements-launcher.txt`.

## Try the demo (one T4, about 50 minutes)

```powershell
. .\scripts\env.ps1
python scripts/bundle_for_kaggle.py --adapter main=<main adapter folder>      # code, data, examples and the adapter
python scripts/kaggle_launch.py dataset --existing                            # upload; waits until Kaggle serves it
python scripts/kaggle_launch.py run demo --tag try --timeout 5400
python scripts/kaggle_launch.py status ats-demo-try
```

The `demo` stage answers the eight example questions (saved to `out/demo/demo_smoke.json`), then starts `scripts/demo_app.py` with a public gradio.live link. The kernel prints `DEMO LINK: https://...` every minute for the time given by `--minutes` (default 50). Open the kernel's page on kaggle.com and read its log while it runs; `kernels output` returns files only after it finishes.

## Train and evaluate

```powershell
python scripts/kaggle_launch.py dataset                    # bundle code and exact data, upload, wait until Kaggle serves it
python scripts/kaggle_launch.py run smoke                  # environment, tokens, 10 QLoRA steps, batched generation, adapter reload
python scripts/kaggle_launch.py run train-both             # main on GPU 0 and control on GPU 1, 2 epochs each
python scripts/kaggle_launch.py output ats-train-both      # -> runs/kaggle/ats-train-both
python scripts/kaggle_launch.py run eval-test --adapters-from ats-train-both
```

`--extra` is passed to the stage. For `eval-*`, `--ckpt` selects `adapter` (final, default) or `adapter_ep1`, `--systems` selects from `base,main,closed,control` (also `candA`, `candB` where attached), and anything else goes to `scripts/run_eval.py`. Each system runs its benchmarks in turn on one GPU and the systems spread over the two GPUs.

The stages, with one line each, are listed at the top of `kaggle/bootstrap.py`: `smoke`, `train-both`, `train-cands`, `patch`, `interp`, `eval-dev`, `eval-test`, `eval-test2`, `eval-product`, `eval-bench`, `eval-jobs`, `train-evidence`, `sample-r5`, `train-dpo`, `force-r5`, `train-dpo2`, `verify-r5`, `verify-main-r5`, `check-system`, `train-verifier`, `r6-system`, `demo`.

Later rounds ran on a second account with a bundle that carries the starting adapters:

```powershell
python scripts/bundle_for_kaggle.py --out runs/kaggle_bundle_account2 --adapter main=<dir> --adapter candA=<dir> --adapter candF1=<dir>
python scripts/kaggle_launch.py --credentials-env <VARIABLE> --bundle-dir runs/kaggle_bundle_account2 dataset --existing
python scripts/kaggle_launch.py --credentials-env <VARIABLE> --bundle-dir runs/kaggle_bundle_account2 run eval-jobs --tag r5test-ref "--extra=--jobs base:bench_r5t_hotpot:10,main:bench_r5t_hotpot:10"
python scripts/kstatus.py <kernel slugs> --wait-minutes 25      # progress of kernels; `kernels logs` shows nothing while a kernel runs
```

Round 6 (`train-verifier`, then `r6-system`) needs the pre-registered data in the bundle, and its sealed test must stay unopened until the development selection is recorded.

## Corrections of 13 November

The reported runs are historical. `python scripts/remediate_audit.py` regenerates the corrected reports on a CPU. The HotpotQA sampler now builds a separate training index and excludes held-out support by text fingerprints; the pair builders drop conflicting prompts and contaminated rows; `train_dpo.py` validates the whole pair file before loading weights. Use `dpo_r5_clean.jsonl` or `dpo_r5b_clean.jsonl` for any new, separately registered experiment. The old DPO files stay for audit. See [`reports/2026-11-13-evaluation-remediation.md`](../reports/2026-11-13-evaluation-remediation.md).

## What the launcher guards against (all found the hard way)

- **Stale dataset mounts.** Kaggle reports a dataset ready while the new version is still processing. The bundle carries a content id (`BUNDLE_ID`); `dataset` waits until Kaggle serves it, `run` refuses to push otherwise, and the kernel aborts if the mounted id differs from the one it was launched for.
- **SQLite without FTS5.** The kernel's system Python (3.13) loses FTS5 once torch is imported, and AgentTune's search backend then returns "no results" instead of an error. The kernel builds a uv-managed Python 3.12 environment, and every evaluation script opens the index through `ragharness.retrieval.open_verified_backend` (hash check plus a probe search).
- **bf16 under fp16.** TRL casts the LoRA weights of a quantized model to bf16, which the T4's fp16 GradScaler cannot unscale. `ragharness.qlora` casts them back to float32 and logs the dtype histogram.

## Measurements

- Python 3.12 venv, `torch==2.11.0` (CUDA 13 wheel on driver 580+, otherwise the cu126 build), then `requirements-gpu.lock.txt` with `--no-deps`.
- Main and control train at the same time on separate GPUs through `CUDA_VISIBLE_DEVICES`. No DDP, no `device_map="auto"`.
- A kernel keeps only `/kaggle/working/out`; the repository and the environment live in `/kaggle/temp`.
- One T4: 8 s per training row at about 1,640 tokens (6.5 GB peak); batched generation 1.8 s per episode for the base model and 3.5 s with the adapter (9.3 GB peak).
- Round 5, one T4: sampling 1,920 rollouts took about 63 minutes per GPU; the DPO loop took 1.6 to 1.7 hours for 84 steps (8.0 GB peak); a checker pass over 366 cases took about 4 minutes. TRL's `DPOTrainer` does not fit a pair longer than about 1,280 tokens on a T4 because it builds full-vocabulary logits (262k entries), so `scripts/train_dpo.py` computes logits only at the completion positions.
- Use the same hardware, precision and decoding for every system in a final evaluation.
