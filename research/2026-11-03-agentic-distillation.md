# Research note: Agentic distillation for the Lexsi hackathon

**Date:** 3 November 2026 (revised the same day after checking the current rules, kickoff slides and tool source code)  
**Track:** Agentic Fine-tuning  
**Primary problem statement:** PS 3 — Agentic distillation  
**Decision:** Distill a stronger teacher's tool-use trajectories into a small, permitted Cohere Labs model. Use database analytics as the demonstration domain.

**Proposed stack:** Aya Expanse 32B teacher through the Cohere API (pending organizer confirmation, with a local Aya Expanse 8B fallback) → verified SQLite trajectories in one shared text tool format → Tiny Aya Global QLoRA student → paired execution evaluation with AgentTune trajectory metrics and AuditKIT reports. Kaggle is the training environment. The submission is a Hugging Face Article with public HF collections, a GitHub repository, an HF Space demo and a local CPU run of the student.

## Judging and submission

### How submissions are judged

The [kickoff slides](https://hackathon.lexsi.ai/kickoff-slides/) describe the process:

1. **Validity check.** The organizers verify the public artifacts, the submitted write-up, the checklist and that there are no duplicates. They then tag each submission to the closest track or sub-track, so a database agent may be grouped with PS 6 (data and database agents) as well as PS 3.
2. **Scoring.** A pool of judges scores submissions double-blind. Each team receives more than one score, and normalized averages select 10–20 teams for Demo Day.
3. **Score categories.** Write-up clarity /10, practical usefulness /10, code/notebook quality /10, and an overall review /100. The slides do not say how the categories are combined.

Three of the four published categories are about the write-up, usefulness and code quality, so each gets scheduled time instead of being left to the last day.

| What the organizers ask for | What we will show |
| --- | --- |
| A small student learns from a stronger teacher's agent trajectories | Verified teacher rollouts, an adapter, and teacher/base/student results with the same harness. |
| Multiple before-and-after metrics | Held-out execution success, tool-call validity, calls and latency per successful task, and a safety-drift check. |
| Practical usefulness | A return-rate or inventory question with visible SQL, a failure or abstention case, a blocked injection attempt, and the student running locally on a laptop CPU. |
| Write-up clarity | An HF Article that follows the organizer checklist below, drafted from Nov 8 instead of at the end. |
| Code/notebook quality | One runnable evaluation command, pinned versions, clean notebooks with outputs, and a clean-checkout reproduction. |
| Originality | Seeded retail data, original adversarial and counterfactual cases, and a comparison with existing community Tiny Aya tool-calling work. |
| Optional preferred ROI comparison | Self-hosted cost per successful task against a closed-source model's token billing, using measured throughput and a stated request volume. |

### Submission checklist

The slides ask for a Hugging Face Article with the sections below. The organizers say complete details will follow, and the submission form is emailed a couple of days before the deadline.

| Checklist section | Planned content |
| --- | --- |
| 1. Problem statement and approach | PS 3, database agent, teacher → verified traces → student. |
| 2. Hackathon repos used, for what, and which features | AgentTune: trajectory collection and normalization, `Project.distill`/SFT, `evaluate_agentic` metrics, `heal()` failure detection. AuditKIT: `agent_eval` with our deterministic scorer as the headline predicate, `compare_models` with bootstrap significance, red-team probes. |
| 3. Models used | Tiny Aya Global, Aya Expanse 32B (and 8B if used), and the closed-source ROI baseline. The checklist asks for each model's license; copy it from the model card. |
| 4. Datasets and benchmarks used | Our seeded retail databases, question templates and traces, plus any external portability set. The checklist asks for licenses here too; copy them from the source pages. |
| 5. Model artifacts as an HF collection | LoRA adapter(s), with the base model linked. |
| 6. Data artifacts as an HF collection | Retained traces, rejected-trace samples with failure labels, evaluation logs and the benchmark. |
| 7. Experiments run, findings, and what else was tested | Main run, controls, safety probes, multilingual slice and negative results. |
| 8. With more credits: approach, before and after results | GRPO on top of the SFT student with an execution-verified reward through AgentTune's `create_agentic_trainer`, plus a larger teacher trace set. This is also the plan for Nov 16–21 if shortlisted. |
| 9. Teammates tagged | All team members. |

### Dates

All organizer times are AoE.

| Dates (IST planning dates) | Milestone and exit criterion |
| --- | --- |
| Nov 3–5 | Register before registration closes (Nov 5 AoE). Join Discord, redeem the HF credit coupon and open a ticket about the API teacher. Build the CPU-only database and harness and lock the benchmark contract. Run the Kaggle environment check. Exit when a scripted fixture produces the expected answer and a rejected write, and AgentTune imports with CUDA working on Kaggle. |
| Nov 5–7 | Use the remaining week-1 GPU quota: base student on 40–60 development tasks, teacher on the same tasks, a short training run, adapter save and reload, and one real tool episode with the reloaded adapter. Exit when the teacher clearly beats the base student and the reloaded adapter completes a tool episode. |
| Nov 7–10 | Generate and filter teacher traces, run trajectory SFT and monitor validation. Keep a completed checkpoint before starting controls. Start drafting the HF Article. |
| Nov 10–12 | Run the fixed holdout for all models, the safety probes, the multilingual slice and latency. Freeze results and select demo traces. Build the HF Space and the local GGUF run. |
| Nov 12–14 | Publish collections, verify a clean-checkout reproduction, finish the article and submit. Aim for Nov 14 IST and use the AoE deadline (about Nov 15, 5:29 p.m. IST) only as a buffer. |
| Nov 16–22, if shortlisted | Use the $200 HF credits for the GRPO stage from checklist section 8, rehearse the live demo, and label any new results as post-submission. |

## Recommendation

Build a small business analytics agent that turns a natural-language question into database tool calls and answers from the returned rows. Distill verified teacher trajectories into Tiny Aya Global, and compare the student before and after training on held-out tasks.

Example: a user asks, "Which products had the highest return rate last month?" The agent calls `run_readonly_sql`, receives the rows, and ends with a `finish` call that carries the final SQL and a structured answer. Scoring then reads the `finish` arguments instead of parsing prose.

This is primarily **agentic distillation**, not a generic text-to-SQL model. The training examples contain the sequence of agent decisions, tool calls, tool responses and final answers. The database setting adds a secondary connection to PS 6 because results can be checked by executing queries.

**Why this is a good competition choice:** a small student fits the compute available, the demo is easy to follow, and correctness can be checked automatically. PS 3 asks for a student that "is cheap to run or runs locally", and a local CPU run shows that directly. These are reasons for choosing the project, not evidence that it will win.

## Competition requirements

- Use a permitted dense, open-weights Cohere Labs model: Tiny Aya Base, Global, Fire, Water and Earth, and Aya Expanse 8B and 32B. Tiny Aya Global is the student.
- Projects must be developed during the hackathon and chain the track's repositories into a real pipeline. The two-repo minimum exempts Track 3; we use AgentTune and AuditKIT together.
- Show more than one before-and-after metric. The write-up is an HF Article (plus an HF Space if relevant) covering the metrics and why they were chosen, the approach, the datasets, tasks and models, and how the result is better for real use. An ROI comparison against a closed-source model is optional but preferred.
- Teams have one to three members. Registration closes on 5 November (AoE).
- Host models, adapters, logs and datasets publicly on Hugging Face as collections, and deploy publicly (a demo or repository link) by the deadline.
- The submission deadline is **14 November 2026, end of day AoE** (about **15 November, 5:29 p.m. IST**). If more than one submission is made, only the last one before the deadline counts.
- HF credits: $20 for registered teams that join Discord, for the first 800 teams to redeem. HF usernames must match the registration form, HF may ask for a payment method, and credits expire 30 days after redemption. Shortlisted teams get a further $200.

Sources: [official hackathon page](https://hackathon.lexsi.ai/) and [kickoff slides](https://hackathon.lexsi.ai/kickoff-slides/). Recheck both before submitting.

## Student, teacher and tooling

### Student

[`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global) is a 3.35B model on the permitted list, with an 8K context. It is gated on Hugging Face: accept the access conditions on the model page and log in with a token before downloading.

AgentTune's README describes distillation into a "<3B" student. Every permitted Tiny Aya model is 3.35B, and `Project.distill` has no parameter-count check, so this is not treated as a blocker. [AgentTune implementation](https://github.com/Lexsi-Labs/AgentTune/blob/main/src/agenttune/agentic/project.py)

T4 settings to verify in the smoke test:

- Tiny Aya's vocabulary is about 262k tokens. The fp32 output logits alone take about 2.1 GB at 2,048 tokens and about 4.3 GB at 4,096, before the backward pass. Keep training sequences around 2–3k tokens, or use a chunked loss.
- T4 has no bf16 support. Run a few forward and backward passes in fp16 and check for NaN or inf loss before a long run.
- Start with 4-bit base weights, microbatch 1, gradient accumulation, gradient checkpointing and frequent checkpoints. [Transformers bitsandbytes hardware requirements](https://huggingface.co/docs/transformers/quantization/bitsandbytes)

### Teacher

**Primary: Aya Expanse 32B through the Cohere API** (`c4ai-aya-expanse-32b`).

- It is on the permitted list, so no non-listed teacher is involved. Confirm in a Discord ticket that generating teacher traces through a hosted endpoint of a permitted model is acceptable.
- Price: $0.50 per million input tokens and $1.50 per million output tokens ([Cohere pricing](https://cohere.com/pricing)). About 1,600 episodes at about 3 calls each, with roughly 2,000 input and 200 output tokens per call, comes to about $6. The model is also served through [Hugging Face Inference Providers via Cohere](https://huggingface.co/blog/inference-providers-cohere), which the HF credits may cover.
- Cohere trial keys are capped at 1,000 calls a month and 20 chat calls a minute ([Cohere FAQ](https://docs.cohere.com/docs/cohere-faqs)), which is too few for the full set. Use a production key or the HF router for generation.
- It uses no Kaggle quota, so GPU hours go to the student.
- Record the API model ID, call dates and decoding settings. A hosted model cannot be pinned to a weights revision, so state this in the write-up.

**Fallback: Aya Expanse 8B on Kaggle**, quantized to 4-bit. Use it if the API route is not allowed or not reliable. Rough, unmeasured estimate: 35–55 seconds per episode with single-stream Hugging Face `generate` on a T4, which is 8–25 GPU-hours for 1,600 attempts depending on batching and whether both GPUs run separate workers. Measure on the pilot and scale the trace count to the quota. Running Aya Expanse 32B on 2×T4 is not a practical fallback: the 4-bit weights must be split across both cards and generation is slow.

**Teacher gate.** Aya Expanse dates from October 2024 and its chat template ignores tools, while Tiny Aya Global is a newer model. A teacher advantage on SQL tool use is unproven. Benchmark the teacher and the base student on the same 40–60 development tasks before generating the full set. If the teacher does not clearly beat the base student, improve the prompt or verification, switch teacher, or stop before claiming distillation. [Aya Expanse report](https://arxiv.org/abs/2412.04261)

**One text tool format for every model.** Do not use the API's native tool calling. Send no `tools` parameter, put the tool schemas in the system prompt using AgentTune's Tiny Aya fallback format, parse tool calls from the assistant text with the same parser used for the student, and return tool results as user turns. Teacher, base student and distilled student then see and produce identical formats.

For every example, record the task, tool-schema revision, model tool calls and arguments, tool responses, final answer, teacher provenance and verification outcome. Execute queries against a sandbox and keep a trajectory only after its results check out. Programmatically generated gold trajectories can test the harness or form a separately labeled control. They do not show that a stronger model taught the student.

### Data

| Data candidate | Role and decision |
| --- | --- |
| Original seeded retail SQLite | Primary training and evaluation. Four tables for products, orders, line items and returns. Generate several database seeds with realistic joins, missing data, ties, cancellations and zero denominators. Keep canonical SQL, schema, generator and seeds under version control. |
| [Chinook](https://github.com/lerocha/chinook-database) | Optional portability test with a different schema and a small set of hand-written questions. |
| [Spider](https://yale-lily.github.io/spider) and [CoSQL](https://yale-lily.github.io/cosql) | Research references for schema transfer and conversational ambiguity. Keep any external evaluation separate from training. |
| [BIRD](https://bird-bench.github.io/) | Optional harder portability check on a labeled subset after the main pipeline works. |

Task design:

- Use seven families: aggregation, joins, time windows, ranking, missing evidence, ambiguous date ranges and unsafe requests.
- Write at least 40 question templates spread across the families, each with canonical SQL. Have the teacher paraphrase questions for variety.
- Split by template within each family, and by database seed, so every family appears in both training and testing but test templates are unseen. Do this before collecting any rollouts.
- Fix a reference date and state it in the system prompt. Define metric semantics, including what counts as a return and how ties and nulls are handled.
- Targets: 400–800 retained training traces, 60–100 validation tasks and 120–200 test tasks. These are planning choices, not validated optima.
- Log attempts, acceptance rates and failure categories. Reject traces with unsupported final claims, malformed tool calls, unauthorized queries or wrong results. Keep representative failures separately for reporting.
- Translate a subset of the test questions into one or two languages that both Aya Expanse and Tiny Aya cover (for example Hindi and Spanish) as an evaluation-only multilingual slice.

### AgentTune integration

- **Trace tiers.** AgentTune's `EventLog` has two tiers. `DictToolHarness` produces light-tier logs, which can be evaluated and healed but cannot be trained on; the training projections raise on them. Trainable full-tier logs come from `Project.collect_rollout` with a rollout engine. [AgentTune agentic spine](https://github.com/Lexsi-Labs/AgentTune/blob/main/src/agenttune/agentic/README.md)
- **One execution path.** Run teacher, base and student episodes through the same episode runner, so prompts, parsing and step limits are identical. Use `DictToolHarness` for evaluation only if it behaves the same as the training path.
- **API teacher traces.** `APIRolloutEngine` passes tool schemas natively through LiteLLM, returns no logprobs and has no local tokenizer ([source](https://github.com/Lexsi-Labs/AgentTune/blob/main/src/agenttune/agentic/rollout_engines/api_engine.py)). Its output will not match the text format and may not be full-tier. Run the API teacher through our own runner with the text format, then either add the trajectories to a `Project` if AgentTune accepts them or build the SFT dataset directly for TRL. Do not fabricate token spans or logprobs.
- **Tool turns in the student template.** Tiny Aya's template renders a `tool` role turn as empty. Training rows must carry tool results as user turns. Add an assertion that every rendered training example contains its tool output text.
- **Loss masking.** TRL's assistant-only loss needs a chat template with generation markers, which Tiny Aya's template lacks. Use one prompt/completion example per assistant turn so prior turns and tool results are masked. Check the rendered input and labels on one successful episode and one recovery episode before training. [TRL SFT trainer](https://huggingface.co/docs/trl/sft_trainer)
- **Verified scope.** AgentTune's GPU-verified distillation example is a single-turn task on SmolLM2-360M ([real examples](https://github.com/Lexsi-Labs/AgentTune/blob/main/examples/REAL_EXAMPLES.md)). Multi-turn tool trajectories on Tiny Aya are untested upstream, so keep a plain TRL/PEFT trainer factory ready. The same document warns to evaluate the saved adapter, not the trainer's in-memory model.
- **Extra features worth reporting.** `Project.evaluate_agentic` computes programmatic trajectory metrics on CPU, and `Project.heal()` detects looping and tool-crash failures. Neither needs GPU time.
- **Known issues.** Fetching any tool from the registry imports every built-in tool and can fail on a missing optional dependency, and the registry's SQL tool has no database configured. Use our own SQLite callables. [Known issues](https://github.com/Lexsi-Labs/AgentTune/blob/main/docs/community/known-issues.md)

### AuditKIT integration

- Use `auditkit.agent_eval` with our deterministic SQL-and-answer scorer as the custom outcome predicate, so the headline metric lives inside AuditKIT and the tool and trace scores serve as diagnostics. Convert `EventLog`s with `episode_from_eventlog` and test the conversion early. [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) · [offline agent example](https://github.com/Lexsi-Labs/AuditKIT/blob/main/examples/agent_eval_offline.py)
- Use `compare_models` for paired comparisons with bootstrap significance.
- Use the red-team probes (prompt injection, jailbreak, over-refusal) for the safety-drift check.

### Harness and trace contract

Tools: `get_schema`, `run_readonly_sql` and `finish(sql, answer)`. Keep the same tool declarations, tokenizer formatting, maximum calls and decoding policy for teacher, base and student.

Open the database with SQLite URI `mode=ro`, use an authorizer to restrict tables and operations, and deny writes, attachment, unsafe functions and arbitrary pragmas. Use a progress handler for deadlines, and bound SQL length, rows, output bytes and steps. Treat returned text as untrusted data. Include fixtures for a denied write, an unauthorized table, an oversized result, a slow query and an injected instruction inside a database row. Test on CPU before starting a GPU session. [SQLite opening modes](https://sqlite.org/c3ref/open.html) · [SQLite authorizer](https://sqlite.org/c3ref/set_authorizer.html) · [SQLite progress callbacks](https://sqlite.org/c3ref/progress_handler.html)

Give each trace a stable task ID, split, template ID, prompt and tool-schema revision, database seed and hash, model ID or revision, decoding config, ordered actions and observations, final response, validation result and timing.

## Kaggle compute plan

**Quota.** Kaggle gives about 30 GPU hours per week per account, sometimes more, and the quota resets weekly ([Kaggle GPU guidance](https://www.kaggle.com/docs/efficient-gpu-usage)). Unused hours do not carry over, so use the remaining week-1 quota for the pilot before the reset, and check the reset time on the quota panel. Each teammate has their own quota. GPU sessions are limited to 12 hours, so checkpoint and plan multiple sessions. [Kaggle notebooks documentation](https://www.kaggle.com/docs/notebooks)

**Hardware.** T4 ×2 gives two separate 16 GB cards. Memory is not pooled, and code does not use both automatically. Prefer a single-GPU trainer and use the second card as an independent worker for evaluation or rollout shards.

**Environment check (first session).**

1. The Kaggle GPU image runs Python 3.13 ([Kaggle Dockerfile](https://github.com/Kaggle/docker-python/blob/main/Dockerfile.tmpl)), which meets AgentTune's Python 3.12+ requirement.
2. AgentTune pins `torch==2.11.0`, `trl==1.7.1` and `transformers>=5.15` ([pyproject](https://github.com/Lexsi-Labs/AgentTune/blob/main/pyproject.toml)). The default PyPI wheels for torch 2.11 are built for CUDA 13.0 ([PyTorch announcement](https://dev-discuss.pytorch.org/t/transitioning-pypi-cuda-wheels-to-cuda-13-0-as-the-stable-release-2-11/3325)), which needs NVIDIA driver 580 or newer. Run `nvidia-smi` first. If the driver is older, install torch 2.11.0 from the cu126 or cu128 index before installing AgentTune, then confirm `torch.cuda.is_available()`.
3. Save the working wheels as a Kaggle dataset so later sessions install offline and quickly.
4. Assume Hugging Face `generate` for local inference. vLLM on T4 cannot use FlashAttention and falls back to Triton attention, with reported failures for some models ([vLLM issue](https://github.com/vllm-project/vllm/issues/38918)).
5. Confirm Hugging Face access to the gated models, Internet access and free disk space. Keep tokens in Kaggle Secrets and out of traces and notebook output.

**Allocation.** These are rough calculations, not measurements. Replace them with measured rates after the pilot.

| Activity | Estimated session hours | Week | Notes |
| --- | ---: | --- | --- |
| Environment check and smoke test | 2 | 1 | Install, load the student, a few training steps, adapter save and reload, one real tool episode. |
| Development pilot | 2 | 1 | Base student on 40–60 tasks. The API teacher runs off-GPU. |
| Local 8B teacher generation (fallback only) | 8–25 | 1–2 | Only if the API teacher is unavailable. Shrink the trace count to fit. |
| Main SFT run | 3–5 | 2 | About 2M tokens per epoch at an estimated 400–800 tokens/s, so 45–90 minutes per epoch, for 3 epochs. |
| Control run | 3–5 | 2 | Self-distillation or gold-trajectory control (see Comparisons). |
| Evaluation of base, student and control | 3–5 | 2 | About 25 seconds per episode single-stream, split across both cards. |
| Safety probes, multilingual slice and latency | 2 | 2 | |
| Contingency | 6+ | 2 | Setup failures, memory issues or reruns. |

With the API teacher, the plan needs roughly 21–27 session hours: the pilot fits in week 1 and the rest fits in week 2 on one account.

**Cutoffs.**

1. At the first pilot, measure generation seconds per episode, training tokens per second, GPU memory and evaluation seconds per task. Project each stage from the actual retained dataset size and context lengths.
2. If local teacher generation exceeds its share, cut the trace count or keep the most useful template families based on validation. Keep the test set fixed.
3. Reserve the last 4 hours of the week-2 quota for final evaluation and artifact upload.
4. If the teacher fails to beat the base student, stop and fix the teacher before claiming distillation.

## Experimental design

### Splits and verification

Prepare training, validation and test sets before tuning, split as described under Data. Keep the test set hidden until model selection is complete. Verify answers by executing reference queries and comparing query results, not SQL strings. Re-run each candidate query on additional database seeds and compare it against the reference query there, so a coincidentally correct result does not pass. The database enforces read-only access independently of model behaviour.

### Comparisons

1. **Base student:** Tiny Aya Global with the same tools and prompts, before training.
2. **Distilled student:** the same student after SFT on verified teacher traces.
3. **Teacher:** same tasks and harness. Disclose that it runs through an API on different hardware.
4. **Did the teacher matter?** Sample the base student several times per training task, keep only verified episodes, and train on those. If the distilled student beats this control, the teacher contributed beyond filtering. If the base student's yield is too low to build a usable set, report the yield as the result. If time allows, add a separately labeled gold-trajectory SFT control.
5. **Closed-model ROI baseline:** run one closed-source API model on the test set for the ROI comparison.
6. **Optional prior-art baseline:** evaluate one public community Tiny Aya tool-calling fine-tune, such as the work described in [Multilingual Tool Calling in 70+ Languages](https://huggingface.co/blog/Bronsn/tiny-facade).

Be ready to answer why a teacher is needed when canonical SQL exists. In deployment there is no canonical SQL for new questions, and teacher traces include schema exploration, error recovery and explanations that gold SQL does not.

Record model IDs, dataset versions, seeds, prompts, decoding settings, hardware, timing method and failures. Do not claim improvement until it has been measured.

### Metrics

| Metric | Measurement |
| --- | --- |
| Task success | Fraction of held-out tasks whose verified result and final answer are correct. |
| Tool-call validity | Fraction of calls with the correct tool name and parseable arguments. |
| SQL execution and safety | Valid queries, blocked unsafe requests and errors, with read-only access enforced outside the model. |
| Efficiency | Calls per successful task and p50/p95 latency under stated hardware, quantization, warm-up and concurrency. |
| Robustness | Paraphrased questions, changed values and injected instructions inside database rows. |
| Safety drift | AuditKIT prompt-injection, jailbreak and over-refusal probes on the base and distilled student. Fine-tuning on harmless data can weaken refusal behaviour ([Qi et al., 2023](https://arxiv.org/abs/2310.03693)). |
| Multilingual slice | Task success on the translated test subset. |
| Trajectory metrics | AgentTune `evaluate_agentic` metrics and `heal()` failure counts. |
| Cost and size | Adapter size, GPU session time, local CPU tokens per second and cost per successful task. |

### What makes the result credible

- Report both result correctness and answer faithfulness. Normalize types, handle NULLs and duplicate rows explicitly, and compare ordered rows for ranking questions and multisets otherwise. [Test suite accuracy paper](https://aclanthology.org/2020.emnlp-main.29/)
- Report paired differences on the same tasks, and bootstrap over templates instead of individual questions so paraphrases are not counted as independent samples. [AgentTuning research](https://arxiv.org/abs/2310.12823)
- Slice results by family and failure mode: malformed arguments, repeated calls, executable but wrong SQL, ungrounded answers and appropriate abstention. Report unsafe attempts, harness blocks, unsafe execution and false refusals separately. [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) · [AgentDojo](https://arxiv.org/abs/2406.13352)
- Report repeated-task reliability on a small subset if feasible. [τ-bench](https://arxiv.org/abs/2406.12045)
- Include real traces of both successes and failures. Avoid significance claims from a small test set.

## Demo and deployment

- **HF Space.** Host the demo as a Gradio Space on ZeroGPU, which the organizers' HF credits are meant for. Free personal accounts with a verified email that are older than 30 days can host two ZeroGPU Spaces ([ZeroGPU docs](https://huggingface.co/docs/hub/en/spaces-zerogpu)); other Gradio Spaces need a paid plan ([Spaces launch page](https://huggingface.co/spaces/launch)). Check eligibility on the team account this week. Store an HF token with access to the gated base model as a Space secret.
- **Local run.** Merge the adapter into the base model, convert it to GGUF and run it with llama.cpp on a laptop CPU, measuring tokens per second. Community Tiny Aya GGUF builds show the architecture is supported. This shows PS 3's "runs locally" directly and gives a near-zero marginal cost for the ROI comparison. Publish the adapter and the conversion script; the merged model is rebuilt locally.
- **Demo script.** One successful query with its SQL, one failure or abstention, one blocked injection attempt, and saved traces for rehearsal and debugging.

## ROI comparison

Compare the self-hosted student with a closed-source model's token billing, as the rules describe: a stated request volume, GPU runtime hours, per-hour GPU price and measured tokens per second. Report cost per successful task alongside success rate, so cheap failures do not count as a win. Include idle time and storage, and report teacher generation cost separately.

## Immediate next steps

1. Register before Nov 5 AoE, join Discord, redeem the HF credits and open a ticket about the API teacher.
2. Build the read-only database, templates and held-out split on CPU.
3. Run the Kaggle environment check and fix the torch/CUDA install if needed.
4. Run one end-to-end base-student episode and save a trace.
5. Run the teacher and base student on the development tasks and apply the teacher gate.
6. Test the full training path on a handful of verified traces, including the tool-turn assertion and adapter reload.
7. Generate the full trace set, train and evaluate, drafting the article in parallel.

Before submitting, verify the public repository on a clean checkout, including database regeneration, a smoke evaluation and loading the adapter from Hugging Face. Open the collection links while signed out and confirm they are public. Publish a model card, data card, training configuration, traces and a metrics report.

## Open questions

- Will the organizers accept teacher traces generated through a hosted API endpoint of Aya Expanse 32B?
- Which driver and torch version does the current Kaggle GPU image ship, and does AgentTune install cleanly on it?
- Does `Project` accept externally generated trajectories for distillation, or do we build the TRL dataset ourselves?
- How many traces and evaluation rollouts fit the measured quota?
- Is the team's HF account eligible for ZeroGPU Spaces?
- When does the Kaggle weekly quota reset for our accounts?

## Primary references

- [Lexsi hackathon: tracks, rules, schedule and submission](https://hackathon.lexsi.ai/) and [kickoff slides](https://hackathon.lexsi.ai/kickoff-slides/)
- [AgentTune repository](https://github.com/Lexsi-Labs/AgentTune), [agentic spine reference](https://github.com/Lexsi-Labs/AgentTune/blob/main/src/agenttune/agentic/README.md) and [known issues](https://github.com/Lexsi-Labs/AgentTune/blob/main/docs/community/known-issues.md)
- [AuditKIT repository](https://github.com/Lexsi-Labs/AuditKIT)
- [Tiny Aya Global model card](https://huggingface.co/CohereLabs/tiny-aya-global), [Aya Expanse 32B](https://huggingface.co/CohereLabs/aya-expanse-32b) and [Aya Expanse 8B](https://huggingface.co/CohereLabs/aya-expanse-8b)
- [Cohere pricing](https://cohere.com/pricing) and [Cohere on HF Inference Providers](https://huggingface.co/blog/inference-providers-cohere)
- [Kaggle notebook resources](https://www.kaggle.com/docs/notebooks), [GPU usage guidance](https://www.kaggle.com/docs/efficient-gpu-usage) and [Kaggle Docker image](https://github.com/Kaggle/docker-python)
- [PyTorch 2.11 CUDA 13.0 wheels](https://dev-discuss.pytorch.org/t/transitioning-pypi-cuda-wheels-to-cuda-13-0-as-the-stable-release-2-11/3325)
- [Hugging Face ZeroGPU Spaces](https://huggingface.co/docs/hub/en/spaces-zerogpu)
