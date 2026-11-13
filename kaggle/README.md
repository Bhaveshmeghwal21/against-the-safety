# Running on Kaggle

**13 November 2026 correction:** The reported runs below are historical. Run `python scripts/remediate_audit.py` locally to regenerate corrected reports and clean preference files; this needs CPU only. The HotpotQA sampler now builds a separate training index and excludes held-out support by text fingerprints. Pair builders remove conflicting prompts and contaminated rows, and `train_dpo.py` validates the complete file before loading weights. Use `dpo_r5_clean.jsonl` or `dpo_r5b_clean.jsonl` for any future, separately registered training experiment. The old DPO artifacts remain available for audit. Main stays submitted, and no new GPU job is required for the current corrections. If later GPU work is needed, use the already authorized second account through `--credentials-env` and keep its credentials out of reports.

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

Rounds 3 to 5b use more stages (`patch`, `interp`, `eval-jobs`, for round 5 `sample-r5`, `train-dpo`, `verify-r5`, `verify-main-r5`, and for round 5b `force-r5`, `train-dpo2`, `check-system`; the list with a line each is at the top of `kaggle/bootstrap.py`). Round 5 ran on a second Kaggle account, with a bundle that carries the starting adapters:

```powershell
python scripts/bundle_for_kaggle.py --out runs/kaggle_bundle_account2 --adapter main=<main adapter dir> --adapter candA=<candA adapter dir>
python scripts/kaggle_launch.py --credentials-env <VARIABLE> --bundle-dir runs/kaggle_bundle_account2 dataset --existing
python scripts/kaggle_launch.py --credentials-env <VARIABLE> --bundle-dir runs/kaggle_bundle_account2 run sample-r5 --tag r5-pairs
python scripts/kaggle_launch.py --credentials-env <VARIABLE> --bundle-dir runs/kaggle_bundle_account2 run train-dpo --tag r5c
python scripts/kaggle_launch.py --credentials-env <VARIABLE> --bundle-dir runs/kaggle_bundle_account2 run verify-r5
python scripts/kstatus.py <kernel slugs> --wait-minutes 25     # progress of one or more kernels; `kernels logs` shows nothing while a kernel runs
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
- Measured in round 5 on one T4: sampling 1,920 rollouts took about 63 minutes per GPU; the DPO loop took 1.6 to 1.7 hours for 84 steps at 8.0 GB peak; one checker pass over 366 cases took about 4 minutes. Two GPU kernels of one account ran at the same time without trouble. TRL's `DPOTrainer` does not fit a pair longer than about 1,280 tokens on a T4 (it materialises the full 262k-entry vocabulary logits), which is why `scripts/train_dpo.py` computes logits at the completion positions only.
