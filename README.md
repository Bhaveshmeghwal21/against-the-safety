# Against-the-safety

We’re building a small document assistant for the Lexsi AI Safety and Alignment Hackathon. Our current entry is in the **Agentic Fine-tuning** track under **PS 4 — Fine-tuning for retrieval-augmented generation**. [Competition overview](https://hackathon.lexsi.ai/).

## Project

The agent learns to search a document corpus, identify missing evidence, answer with source citations, and abstain when the available documents cannot support an answer. The proposed demo uses fictional product manuals, troubleshooting notes and dated warranty policies.

We plan to fine-tune **Tiny Aya Global** with QLoRA on verified search and answer episodes. Matched examples vary whether supporting evidence is complete, incomplete, absent or mixed with distractors. AgentTune supplies retrieval and agent components, with AuditKIT supporting evaluation. [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) · [AgentTune](https://github.com/Lexsi-Labs/AgentTune) · [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT).

## Evaluation

We’ll compare the original and fine-tuned model on the same held-out questions with the same retrieval backend and call budget. Each part of PS 4 gets its own measurement: when the model searches, whether its queries find the full evidence, whether answers are supported by cited passages, and how much distractors hurt. We also measure unsupported answers when evidence is missing, false abstentions, calls and latency. Four core comparisons include a supported-only tool-episode SFT control with a matched training budget. Paired tests cover missing, partial, distracting and already-supplied evidence. AuditKIT's deterministic RAG checks score evidence-set recall, citation support, abstention and stale documents.

## Demo

The demo is a ZeroGPU Space if our Hugging Face account qualifies (free accounts must be older than 30 days), otherwise a static page of saved episodes. We'll also measure a local laptop CPU run with a GGUF build.

## Status

The CPU side is built and tested: data sampler, shared retrieval index, six-state matched episode builder, AgentTune-to-AuditKIT evaluation, QLoRA training path (rehearsed on CPU), baselines and Kaggle scripts (section 16 and 17 of the plan). The GPU smoke test is the next step and needs a Hugging Face token that has accepted the Tiny Aya gate. No GPU run or real training has happened yet.

## Run the CPU harness

Python 3.12. Everything, including caches and temp files, stays inside this folder.

```powershell
py -3.12 -m venv .venv
. .\scripts\env.ps1                      # redirects every cache and temp path into .cache/
python -m pip install --no-deps -r requirements-cpu.lock.txt
python -m pip install --no-deps -e .
python scripts/prepare_data.py           # sample HotpotQA splits, write manifests
python scripts/build_index.py            # shared SQLite FTS index
python scripts/retrieval_ceiling.py      # oracle retrieval ceiling on dev
python scripts/build_episodes.py         # six-state matched episodes, AgentTune trajectories, Tiny Aya rows
python scripts/make_training_set.py      # main, control and dev sets
python scripts/leakage_report.py         # dev/test overlap with SFT rows
python scripts/shortcut_report.py        # label-leak checks on training episodes
python scripts/gate_baseline.py          # retrieval-score gate baseline
python scripts/run_eval.py --system oracle --split test --out runs/eval/oracle_test
python scripts/dry_run_qlora.py          # whole training path on CPU with a tiny stand-in model
python scripts/smoke_test.py --stand-in --out runs/smoke_cpu --skip fp16,vllm
python -m pytest tests -q
```

GPU steps (smoke test, training, evaluation) are in [kaggle/README.md](kaggle/README.md).