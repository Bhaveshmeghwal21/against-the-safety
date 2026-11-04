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

The project is at the CPU-harness stage: the data sampler, shared retrieval index, masked search, AgentTune-to-AuditKIT evaluation slice and Tiny Aya row builder run and pass their checks (section 16 of the plan). No GPU run or training has happened yet. The [current paper review and plan](research/2026-11-04-fine-tuning-for-rag.md), rechecked twice on 4 November 2026, covers the literature, rules and judging, datasets, retrieval harness, training, evaluation, compute and submission milestones. Compute comes from the Kaggle T4 ×2 weekly quota, which should reset around Nov 7 and Nov 14, plus the $20 HF credit if it arrives. Final sample sizes depend on the throughput pilot; the allocation is not a measured runtime guarantee. The [earlier distillation note](research/2026-11-03-agentic-distillation.md) records the previous direction.

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
python scripts/audit_slice.py            # AgentTune tool and trajectories -> AuditKIT metrics
python scripts/check_rows.py             # AgentTune distill hook -> Tiny Aya rows, template-checked
python -m pytest tests -q
```
