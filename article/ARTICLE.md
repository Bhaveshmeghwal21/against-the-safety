# Teaching a small Cohere model to search, cite and say "the documents don't say"

**Team Against-the-safety** · Lexsi AI Safety and Alignment Hackathon · Track 3 (agentic fine-tuning) · PS 4: fine-tuning for retrieval-augmented generation

Code: https://github.com/Bhaveshmeghwal21/against-the-safety

## Abstract

A document assistant is only useful if it answers from its documents and stops when they do not contain the answer. We fine-tuned Tiny Aya Global (3.35B, a Cohere model) with QLoRA, through AgentTune, to run a three-search retrieval loop and then either answer with citations to the passages it saw or abstain and say what is missing. Training used 600 HotpotQA questions turned into scripted episodes in six evidence states (clean support, distractors, second hop, missing evidence, evidence already supplied, weak first search), 1,611 rows in total. We trained two adapters under an identical recipe: a main adapter on the full mixture and a control on supported tool episodes only. We evaluated on a held-out test split, opened once, with the same retrieval index, call budget and decoding for every system. Scoring was deterministic code plus AuditKIT's RAG and agent metrics.

On the test split the end-to-end good rate (a supported answer when the documents hold it, an abstention when they do not) went from 0.020 for the base model to 0.583 for the main adapter (+0.563, 95% interval +0.490 to +0.636), and from 0.404 for the control to 0.583 (+0.179, +0.103 to +0.252). The base model answered from missing evidence in 78% of unanswerable episodes; main does so in 9.8%, the control in 26.8%. Citations to a gold passage when answering rose from 12.6% to 99.4%. The improvement has a price: the main adapter abstains on 19% of answerable episodes, and its supported-answer rate on answerable questions (0.527) is lower than the control's (0.707). The control never abstains and, when evidence is missing, mostly loops until it runs out of searches. We report both and do not merge them into one score. A 24-task check on a fictional product's documents shows the same pattern and weaker results on two-document questions and on choosing the current edition of a policy. Training cost 3.9 T4 GPU-hours on the free Kaggle quota (about 1.36 USD at a reference price).

## 1. Problem and approach

PS 4 splits retrieval-augmented generation into five decisions: when to retrieve, what to retrieve, whether the answer is grounded, whether distractors are ignored, and whether the model abstains when the documents cannot answer. A small model usually gets the second step roughly right and fails the others: the base Tiny Aya answered 78% of questions whose evidence we had removed, and searched even when the evidence was already in the prompt (60 of 60 test episodes).

We treat this as behaviour cloning on matched evidence states. For each training question the episode builder uses the same scripted policy under different masks and pre-supplied evidence, so the only thing that tells the model whether to answer or abstain is what is visible in the conversation. The six states: clean support after one search (S1), a distractor ranked above the support (S2), second-hop recovery (S3), all support masked so the right move is abstain (S4), evidence given in the prompt fully or partly (S5), and a weak or masked first search recovered by a reworded query (S6). Masked episodes are rejected when the gold answer text still appears in a visible passage, and questions whose support the scripted search cannot reach within three searches are reported and never trained on.

The model sees a small protocol: a `search_corpus` call as JSON, passages with their chunk ids, then `<answer>…</answer><citations>…</citations>` or `<abstain>…</abstain>`. Budget: three searches, three passages per search, 800 characters shown per passage.

Two adapters share one recipe fixed before any result (QLoRA r=16, alpha=32, all linear layers, learning rate 2e-4, effective batch 16, 2 epochs, seed 13):

- **main**: 1,611 rows, 537 each of search, answer and abstain turns, from all six states.
- **control**: 1,697 rows of supported tool episodes only. It answers the same way but has never seen an abstention. It tests whether the abstention data, not just tool-use practice, produced the change.

The development split chose the epoch (epoch 2: 0.673 end-to-end good against 0.270 at epoch 1) and nothing else. The test split was opened once.

## 2. Lexsi repositories and features used

- **AgentTune**: `create_distill_trainer` with our QLoRA trainer factory for training; `SQLiteFTSBackend` and `SearchCorpusTool` for retrieval and the tool the model calls; `EventLog` and `TrajectoryStore` for every training episode and every evaluation rollout; `qa_metrics` F1 and exact match.
- **AuditKIT**: the RAG metrics `evidence_set_recall`, `citation_support`, `abstention` and `freshness`, and the agent metrics `tool_call_validity`, `redundant_tool_calls` and `agent_loop_detection`. Each episode is an `ak.Sample` with a `RAGTrace` and is scored with `ak.evaluate`.

Two integration problems we hit and documented in the repository: TRL casts LoRA weights of a quantized model to bf16, which the T4's fp16 gradient scaler rejects (we cast them back to fp32), and AgentTune's SQLite backend returns "no results" instead of an error when the Python build lacks FTS5, which happened on Kaggle's default Python (we run a uv-managed Python 3.12 and verify the index with a probe search before every evaluation).

## 3. Models and licenses

| model | what it is | license |
| --- | --- | --- |
| [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) (revision `af89d219`) | base and student model, Cohere | CC BY-NC 4.0, gated |
| [against-the-safety-tiny-aya-rag-main](https://huggingface.co/BMEscanor/against-the-safety-tiny-aya-rag-main) | LoRA adapter, full mixture | CC BY-NC 4.0 (inherited) |
| [against-the-safety-tiny-aya-rag-control](https://huggingface.co/BMEscanor/against-the-safety-tiny-aya-rag-control) | LoRA adapter, supported-only control | CC BY-NC 4.0 (inherited) |

## 4. Datasets and benchmarks, with licenses

| dataset | contents | license |
| --- | --- | --- |
| [against-the-safety-rag-episodes](https://huggingface.co/datasets/BMEscanor/against-the-safety-rag-episodes) | training rows, AgentTune event logs, dev and test evaluation traces, reports | CC BY-SA 4.0 (derived from HotpotQA) |
| [nimbus-home-support-suite](https://huggingface.co/datasets/BMEscanor/nimbus-home-support-suite) | 24 fictional product documents, 24 tasks, evaluation traces | CC BY 4.0 |
| HotpotQA (distractor) | source of questions and passages | CC BY-SA 4.0 |

## 5. Collections

- Models: https://huggingface.co/collections/BMEscanor/against-the-safety-tiny-aya-rag-adapters-6aec1fda812ed14d9f60bfbc
- Data, traces and demo: https://huggingface.co/collections/BMEscanor/against-the-safety-rag-episodes-traces-and-demo-6aec1fdb994e3b036c6c958d
- Static trace viewer (no model runs; it shows recorded episodes side by side): https://huggingface.co/spaces/BMEscanor/against-the-safety-trace-viewer

## 6. Experiments and findings

All numbers are from the test split, scored by deterministic code, the same questions and retrieval for every system. Headline episodes: 151 (86 answerable, 65 with the evidence masked). Intervals come from resampling whole questions.

| | Tiny Aya base | main | control |
| --- | ---: | ---: | ---: |
| end-to-end good (151) | 0.020 | **0.583** | 0.404 |
| supported correct answer, answerable (205) | 0.034 | 0.527 | **0.707** |
| cites a gold passage when answering | 0.126 | **0.994** | 0.990 |
| abstains when the evidence is missing (82) | 0.024 | **0.756** | 0.000 |
| answers anyway when the evidence is missing (82, lower is better) | 0.780 | **0.098** | 0.268 |
| abstains although the evidence was there (205, lower is better) | **0.000** | 0.190 | **0.000** |
| searches although the evidence was supplied (60, lower is better) | 1.000 | 0.283 | **0.067** |
| full evidence found by its own queries, open questions (86) | 0.360 | 0.709 | **0.826** |
| correct with a distractor passage present (30) | 0.067 | 0.667 | **0.833** |
| invalid or rule-breaking turns (287, lower is better) | 0.188 | **0.070** | 0.251 |
| AuditKIT loop-free (1.0 is best) | 0.941 | 0.934 | 0.784 |

What the numbers say:

- **Abstention training is what separates main from control.** Both adapters learn the search loop and citation format (citing a gold passage 99% of the time when they answer). Only main learns to stop: it abstains on 76% of missing-evidence episodes, the control on none. When evidence is missing the control keeps searching until the budget runs out; 58 of its 65 headline missing-evidence episodes end without a valid final turn.
- **It costs answers.** Main declines 39 of 205 answerable episodes (19%). The control answers more of them correctly (0.707 against 0.527; difference −0.180, interval −0.268 to −0.099). If a wrong answer costs little, the control is the better adapter; if it costs a lot, main is.
- **Strict format compliance.** Read strictly (exactly the trained protocol), the base model breaks the format on 279 of 287 episodes, because it tends to write a tool call and an invented answer in the same turn. We therefore use a lenient reading (first valid action wins) as the primary one, fair to the base model, and report the strict reading beside it. Both were fixed before the test split.
- **Closed book.** With no documents, the base model gets 16.3% of the 86 test questions right from memory, which is the level any retrieval system has to beat.
- **Epochs.** Epoch 1 of main was far worse than epoch 2 on the development split (0.270 against 0.673) because it had not yet learned to abstain; the control got no better with a second epoch (0.289 against 0.277) and developed loops.
- **Practical check (Nimbus Home, 24 tasks, exploratory).** Ordinary one-document questions: base 0/6, main 4/6, control 6/6. Missing-evidence tasks: correct abstain base 0/6, main 6/6, control 0/6. Two-document questions are weak (main 1/6, control 3/6), and on dated policy changes main cites a superseded edition in 4 of 6 tasks (control 3 of 6). The first run of this suite showed that F1 could not tell "30 days" from "45 days", so we shortened the gold answers to the discriminating tokens, fixed one question, re-ran every system, and report the second run; both runs' traces are in the dataset.
- **Cost.** Training took 3.9 T4 GPU-hours for main and 3.6 for the control, on the free Kaggle quota; about 1.36 USD at 0.35 USD per T4-hour. Per good outcome, GPU cost was 0.0135 USD for the base, 0.00084 for main and 0.00105 for the control, with main using more tokens and seconds per episode (5.0 s against 2.8 s for the base).

### Can decoding buy back the false abstains?

We tried the one lever that needs no retraining and no second look at the test split: a bias added at decoding time to the logit of `abstain` right after `<`. Dev only, nine settings, rule fixed beforehand (highest end-to-end good rate, ties toward 0). Setting 0 won, so the test result above stands as the only test number.

| bias | end-to-end good | false abstain (answerable, 86) | correct abstain (unanswerable, 73) | answered anyway (unanswerable, 73) |
| ---: | ---: | ---: | ---: | ---: |
| −9 and below | 0.264 | 0.000 | 0.000 | 0.740 |
| −6 | 0.547 | 0.116 | 0.630 | 0.356 |
| −3 | 0.654 | 0.233 | 0.918 | 0.068 |
| 0 (selected) | 0.673 | 0.233 | 0.959 | 0.027 |
| +1 | 0.673 | 0.256 | 0.959 | 0.027 |

Moving from 0 to −6 halves the false abstains but turns most of the correct abstentions into unsupported answers; at −9 abstaining is switched off. Between 0 and −3 almost nothing changes, which says the abstain decision is nearly all-or-nothing in this adapter, not a confidence that can be thresholded. Reducing the 19% false-abstain rate probably needs different training data (a smaller share of abstention episodes, or harder answerable examples); we did not test that.

### Alternatives we considered and did not run

An Aya Expanse teacher and distillation, and reinforcement learning as the core method: SFT on matched states was the smallest change that could test the abstention question. A second training round with a different mixture: that would have meant opening the test split again, so the decoding-time sweep above is the only post-hoc lever we examined.

### Limits

Small evaluation sets (86 answerable questions); differences of a few points are not established. F1 ≥ 0.5 against the gold answer can credit near-misses on short answers, so exact-match rates are in the reports (main 0.424, control 0.571, base 0.020 on answerable episodes). English, short factual questions. The check that masked episodes lack the answer is a text match, not a proof that no other passage could answer.

## 7. What more credits would enable

More than two T4 GPUs and a longer quota window would let us (1) train the same recipe on all of HotpotQA's training set instead of 600 questions, (2) run several seeds and give variance for the 19% false-abstain rate, (3) add a mixture between main and control (a smaller share of abstention episodes) and select it on the development split to move along the accuracy-coverage curve, and (4) evaluate on a document set from a real deployment, with several editions of the same policy, which is where our 24-task check was weakest.

## 8. Team

Team Against-the-safety. Add each teammate's Hugging Face username here before posting: `@username1`, `@username2`.
