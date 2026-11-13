# Teaching a small Cohere model to search, cite and say "the documents don't say"

**13 November 2026 correction:** This table preserves the original scorer and selection record. Evidence alternatives, binary answers, fixed eligibility and source offsets are corrected in the [dated remediation report](../reports/2026-11-13-evaluation-remediation.md). Later DPO prompts exposed held-out HotpotQA evidence. Annotation-based unsupported flags do not establish semantic hallucination; main remains submitted.


**Team Against-the-safety** · Lexsi AI Safety and Alignment Hackathon · Track 3 (agentic fine-tuning) · PS 4: fine-tuning for retrieval-augmented generation

Code: https://github.com/Bhaveshmeghwal21/against-the-safety

## Abstract

A document assistant is only useful if it answers from its documents and stops when they do not contain the answer. We fine-tuned Tiny Aya Global (3.35B, a Cohere model) with QLoRA, through AgentTune, to run a three-search retrieval loop and then either answer with citations to the passages it saw or abstain. Training drew from 600 sampled HotpotQA questions and selected 1,611 assistant decision rows from 488 parent questions across six evidence states, including missing evidence. A main adapter learned the full mixture; a control learned supported tool episodes only. Every system ran with the same index, call budget and decoding, and was scored by deterministic code plus AuditKIT's RAG and agent metrics. Later choices used dated protocols; the round-5b development gate was informed by earlier development results, as disclosed.

First round, on a held-out HotpotQA test split opened once: the share of episodes that end well (a supported answer when the documents hold it, an abstention when they do not) rose from 0.020 for the base model to 0.583 for main (+0.563, 95% interval +0.490 to +0.636), against 0.404 for the control. Answers given despite missing evidence fell from 78% to 9.8%. Main declines 19% of answerable questions. A second split of 100 fresh questions replicated the result (0.028 to 0.601). On five external benchmarks (1,131 episodes) main went from 0.079 to 0.499 and improved on four. It fails on two-part answers and refuses most single-passage SQuAD questions.

We tried four fixes, each judged by rules set beforehand, and adopted none. A SQuAD patch answered single-passage questions (supported answers 4 to 32 of 57) but raised unsupported SQuAD answers from 6 to 31 of 109, failing its safety guard. A matched-evidence curriculum improved controlled exercises, and no checkpoint passed the release rules. Preference training (DPO) raised SQuAD accuracy, but all four checkpoints stopped abstaining, because our 335 pairs rejected an abstention 91 times and chose one twice. A second DPO run with abstention pairs restored abstention; its one eligible checkpoint, candF1, scored 0.570 against 0.517 on the sealed test but failed the registered safety condition.

What worked needed no further training: two repairs on the main adapter's output. Fail-closed ends a rollout without a valid final action as an abstention. A self-check has the same model read each answer against its cited passages and withhold it if unsupported (one forward pass per answer). On a sealed test of 938 fresh SQuAD, HotpotQA and RGB episodes, opened twice, macro success rose from 0.022 for the base model to 0.517 (+0.495, +0.468 to +0.521). Answers given although the evidence was hidden fell from 276 of 369 episodes to 37 of 364, and supported correct answers on answerable episodes rose from 2.7% to 27.5%. Against the adapter alone, the repairs add +0.103 macro (+0.079 to +0.127) and cut unsupported answers from 29.7% to 10.2%. They cost caution: refusals of answerable questions rise from 22% to 39%, and supported correct answers do not increase (29.0% to 27.5%).

Training the main adapter cost 3.9 T4 GPU-hours on the free Kaggle quota.

## 1. Problem and approach

PS 4 splits retrieval-augmented generation into five decisions: when to retrieve, what to retrieve, whether the answer is grounded, whether distractors are ignored, and whether the model abstains when the documents cannot answer. A small model usually gets the second step roughly right and fails the others: the base Tiny Aya answered 78% of questions whose evidence we had removed, and searched even when the evidence was already in the prompt (60 of 60 test episodes).

We treat this as behaviour cloning on matched evidence states. For each training question the episode builder uses the same scripted policy under different masks and pre-supplied evidence, so the only thing that tells the model whether to answer or abstain is what is visible in the conversation. The six states are clean support after one search, a distractor ranked above the support, second-hop recovery, all support masked (the right move is to abstain), evidence given in the prompt fully or partly, and a weak first search recovered by a reworded query. Masked episodes are rejected when the gold answer text still appears in a visible passage, and questions whose support the scripted search cannot reach within three searches are never trained on.

The model sees a small protocol: a `search_corpus` call as JSON, passages with their chunk ids, then `<answer>…</answer><citations>…</citations>` or `<abstain>…</abstain>`. The budget is three searches, three passages per search, 800 characters per passage.

Two adapters share one recipe fixed before any result (QLoRA r=16, alpha=32, all linear layers, learning rate 2e-4, effective batch 16, 2 epochs, seed 13):

- **main**: 1,611 rows, 537 each of search, answer and abstain turns, from all six states.
- **control**: 1,697 rows of supported tool episodes only. It answers the same way but has never seen an abstention, so it tests whether the abstention data produced the change.

The development split chose the epoch (epoch 2: 0.673 end-to-end good against 0.270 at epoch 1) and nothing else. The test split was opened once.

This first-round adapter is the base of everything that follows. Rounds 2 to 4 tried to improve it with more training, and none replaced it under its pre-registered rules. Round 5 adopted two output-side repairs that need no training and rejected preference training. Round 5b repeated the preference training with the fix the failure called for. Its one eligible checkpoint scored higher on the sealed test but did not meet the safety condition of the replacement rule, so the main adapter with the two repairs is the submitted system (sections 6.5 and 6.6).

## 2. Lexsi repositories and features used

- **AgentTune**: `create_distill_trainer` with our QLoRA trainer factory for training; `SQLiteFTSBackend` and `SearchCorpusTool` for retrieval and the tool the model calls; `EventLog` and `TrajectoryStore` for every training episode and evaluation rollout; `qa_metrics` F1 and exact match.
- **AuditKIT**: the RAG metrics `evidence_set_recall`, `citation_support`, `abstention` and `freshness`, and the agent metrics `tool_call_validity`, `redundant_tool_calls` and `agent_loop_detection`. Each episode is an `ak.Sample` with a `RAGTrace`, scored with `ak.evaluate`.

Two integration problems are documented in the repository. TRL casts LoRA weights of a quantized model to bf16, which the T4's fp16 gradient scaler rejects (we cast them back to fp32). AgentTune's SQLite backend returns "no results" instead of an error when the Python build lacks FTS5, as on Kaggle's default Python (we run a uv-managed Python 3.12 and probe the index before every evaluation).

In rounds 5 and 5b every rollout still runs through AgentTune's tool, trajectory and evaluation stack, but the preference-training step does not use an AgentTune class. TRL's `DPOTrainer`, wrapped by AgentTune's `TrlAgenticDPO`, builds full-vocabulary logits (262k entries) for both sequences of a pair and ran out of memory on a 15 GB T4, so only 38 of 335 pairs would fit. We replaced it with a short loop (`scripts/train_dpo.py`) that computes logits at the completion positions only and uses the same loss. The round-5 headline numbers come from the outcome classes of our scorer version 2, not from AuditKIT metrics.

## 3. Models and licenses

| model | what it is | license |
| --- | --- | --- |
| [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) (revision `af89d219`) | base and student model, Cohere | CC BY-NC 4.0, gated |
| [against-the-safety-tiny-aya-rag-main](https://huggingface.co/BMEscanor/against-the-safety-tiny-aya-rag-main) | LoRA adapter, full mixture | CC BY-NC 4.0 (inherited) |
| [against-the-safety-tiny-aya-rag-control](https://huggingface.co/BMEscanor/against-the-safety-tiny-aya-rag-control) | LoRA adapter, supported-only control | CC BY-NC 4.0 (inherited) |
| [against-the-safety-tiny-aya-rag-cand-a](https://huggingface.co/BMEscanor/against-the-safety-tiny-aya-rag-cand-a) | LoRA adapter, round 2, 30% abstentions (fewer refusals; not the selected model) | CC BY-NC 4.0 (inherited) |
| [against-the-safety-tiny-aya-rag-patch](https://huggingface.co/BMEscanor/against-the-safety-tiny-aya-rag-patch) | LoRA adapter, round 3 patch (candD): answers single-passage questions, fails the safety guard; not the selected model | CC BY-NC 4.0 (inherited) |
| [against-the-safety-tiny-aya-rag-cand-f1](https://huggingface.co/BMEscanor/against-the-safety-tiny-aya-rag-cand-f1) | LoRA adapter, round-4 checkpoint candF1, the round-5b alternative (more answers, same repairs; not the submitted system) | CC BY-NC 4.0 (inherited) |

The round-4 checkpoint candF1, the documented alternative of round 5b, is published as [against-the-safety-tiny-aya-rag-cand-f1](https://huggingface.co/BMEscanor/against-the-safety-tiny-aya-rag-cand-f1) (same license). The preference-trained checkpoints (candG, candH, candI, candK) were not selected and are not published; their pair files, development traces and logs are in the dataset repositories.

## 4. Datasets and benchmarks, with licenses

| dataset | contents | license |
| --- | --- | --- |
| [against-the-safety-rag-episodes](https://huggingface.co/datasets/BMEscanor/against-the-safety-rag-episodes) | training rows, AgentTune event logs, dev, test and test2 evaluation traces, round 3 to 5b sets and traces, reports | mixed source licenses: HotpotQA/SQuAD CC BY-SA 4.0; RGB-derived verifier evidence CC BY-NC-SA 4.0 |
| [nimbus-home-support-suite](https://huggingface.co/datasets/BMEscanor/nimbus-home-support-suite) | 24 fictional product documents, 24 tasks, evaluation traces | CC BY 4.0 |
| [against-the-safety-benchmark-traces](https://huggingface.co/datasets/BMEscanor/against-the-safety-benchmark-traces) | evaluation traces and reports on the external benchmarks and the sealed round-5 test | per folder, as the source benchmark (below) |
| HotpotQA (distractor) | source of questions and passages | CC BY-SA 4.0 |
| 2WikiMultihopQA (dev) | external benchmark | Apache-2.0 |
| MuSiQue-Ans (dev, 2-hop) | external benchmark | CC BY 4.0 |
| SQuAD 2.0 (dev; training split for rounds 3 to 5b) | external benchmark; training, development and sealed-test episodes | CC BY-SA 4.0 |
| RGB (English refined and integration sets) | external benchmark | CC BY-NC-SA 4.0 |

## 5. Collections

- Models: https://huggingface.co/collections/BMEscanor/against-the-safety-tiny-aya-rag-adapters-6aec1fda812ed14d9f60bfbc
- Data, traces and demo: https://huggingface.co/collections/BMEscanor/against-the-safety-rag-episodes-traces-and-demo-6aec1fdb994e3b036c6c958d
- Static trace viewer (no model runs; it shows recorded episodes side by side): https://huggingface.co/spaces/BMEscanor/against-the-safety-trace-viewer

## 6. Experiments and findings

### 6.1 First round: abstention training works, and costs answers

Test split, scored by deterministic code, the same questions and retrieval for every system. Headline episodes: 151 (86 answerable, 65 with the evidence masked). Intervals come from resampling whole questions.

| | Tiny Aya base | main | control |
| --- | ---: | ---: | ---: |
| end-to-end good (151) | 0.020 | **0.583** | 0.404 |
| supported correct answer, answerable (205) | 0.034 | 0.527 | **0.707** |
| cites a gold passage when answering | 0.126 | **0.994** | 0.990 |
| answers anyway when the evidence is missing (82, lower is better) | 0.780 | **0.098** | 0.268 |
| abstains although the evidence was there (205, lower is better) | **0.000** | 0.190 | **0.000** |
| searches although the evidence was supplied (60, lower is better) | 1.000 | 0.283 | **0.067** |
| full evidence found by its own queries, open questions (86) | 0.360 | 0.709 | **0.826** |
| invalid or rule-breaking turns (287, lower is better) | 0.188 | **0.070** | 0.251 |

Main minus base on end-to-end good: +0.563 (95% interval +0.490 to +0.636). Main minus control: +0.179 (+0.103 to +0.252).

- **Abstention training is what separates main from control.** Both learn the search loop and the citation format. Only main learns to stop: it abstains on 76% of missing-evidence episodes, the control on none. When evidence is missing the control keeps searching until the budget runs out (58 of its 65 headline missing-evidence episodes end without a valid final turn).
- **It costs answers.** Main declines 39 of 205 answerable episodes (19%). The control answers more of them correctly (difference −0.180, interval −0.268 to −0.099).
- **Strict reading.** Read as exactly the trained protocol, the base model breaks the format on 279 of 287 episodes, because it writes a tool call and an invented answer in the same turn. We use a lenient reading (first valid action wins) as the primary one, fair to the base model, and report the strict reading beside it. Both were fixed before the test split.
- **Closed book.** With no documents, the base model gets 16.3% of the 86 test questions right from memory.
- **Decoding cannot buy back the refusals.** A bias on the logit of `abstain`, nine settings, dev only, rule fixed beforehand: setting 0 won. At −6 the false abstains halve but unsupported answers on unanswerable questions rise from 2 to 26 of 73, and the abstain decision is nearly all-or-nothing between 0 and −3.
- **Practical check (Nimbus Home, 24 fictional-product tasks, exploratory).** Missing-evidence tasks: correct abstain base 0/6, main 6/6, control 0/6. One-document questions: 0/6, 4/6, 6/6. Two-document questions are weak (main 1/6), and on dated policy changes main cites a superseded edition in 4 of 6 tasks. The first run showed that F1 could not tell "30 days" from "45 days", so we shortened the gold answers, re-ran every system and report the second run; both runs' traces are published.
- **Cost.** Training took 3.9 T4 GPU-hours for main (about 1.36 USD at 0.35 USD per T4-hour). GPU cost per good outcome was 0.0135 USD for the base and 0.00084 for main; main takes 5.0 s per episode against 2.8 s for the base.

### 6.2 Round 2: a fresh test split and an attempt to cut the refusals

Choosing a new mixture after opening the test split would have made it a selection set, so round 2 drew **test2**, 100 new HotpotQA questions whose supporting passages appear in no training row. Every rule was pushed before training. In main's training set, abstentions are half of the final decisions and 81% of those after three searches, and main copies this. Two candidates cut the share: **candA** lowers abstentions to 30% of final decisions, **candB** also equalises the abstain rate after two and three searches. The pre-registered rule picked candA among candidates on dev but kept main, because candA was below main there (end-to-end good 0.629 against 0.673).

| test2 | base | **main** | candA | control |
| --- | ---: | ---: | ---: | ---: |
| end-to-end good (143) | 0.028 | **0.601** | 0.657 | 0.280 |
| false abstain (199, lower is better) | **0.000** | 0.221 | 0.146 | **0.000** |
| unsupported answer (74, lower is better) | 0.878 | **0.041** | 0.081 | 0.189 |

Main minus base: +0.573 (+0.503 to +0.643). Main minus candA: −0.056 (−0.122 to +0.007). The first-round result replicates, and fewer abstentions in training means fewer refusals and more unsupported answers. The two are operating points rather than better and worse models, so both are published. Retrieval did not improve: re-weighting existing examples does not teach better follow-up queries.

### 6.3 External benchmarks

Do the behaviours hold on data we neither built nor used for any choice? Five public sets, settings and metrics were fixed before any model ran on them, each with its own index and the same harness. The scores compare our systems with each other and cannot be compared with official leaderboards, which hand the model the gold paragraphs.

| ends well | base | **main** | candA | control | main minus base (95% interval) |
| --- | ---: | ---: | ---: | ---: | --- |
| 2WikiMultihopQA (166) | 0.018 | **0.566** | 0.620 | 0.241 | +0.548 (+0.479 to +0.619) |
| MuSiQue 2-hop (168) | 0.018 | **0.536** | 0.577 | 0.250 | +0.518 (+0.455 to +0.581) |
| SQuAD 2.0 (202) | 0.010 | **0.475** | 0.559 | 0.040 | +0.465 (+0.404 to +0.528) |
| RGB (453) | 0.157 | **0.618** | 0.660 | 0.530 | +0.461 (+0.406 to +0.518) |
| RGB integration (142) | 0.070 | **0.028** | 0.049 | 0.007 | −0.042 (−0.097 to +0.008) |
| pooled (1,131) | 0.079 | **0.499** | 0.547 | 0.293 | +0.420 (+0.386 to +0.453) |

- **Abstention transfers.** With the support hidden, main abstains in 85%, 85%, 71% and 52% of episodes on the first four sets (base: 4% or fewer), and declines 57 of SQuAD 2.0's 80 human-written unanswerable questions (base: 0).
- **Two-part answers do not.** On RGB integration main cites both passages but answers with the first part only; accuracy with clean passages falls from 0.383 for the base model to 0.017. Every training answer was a single HotpotQA span.
- **Single-passage questions are refused.** On SQuAD 2.0 main declines 54 of 73 answerable questions, and in all 54 the answer was already in its first search results. Every training question needed two passages.
- **On-topic passages that don't answer are often trusted.** At RGB noise 1 main declines only 26 of 100 times.

### 6.4 Rounds 3 and 4: fixes that failed their rules

**Round 3 patch.** Two candidates continued from main for one epoch (93 steps) on new scripted episodes: 240 SQuAD 2.0 training-split questions, 120 compound questions with the answer "A1; A2", and prompt variants on the HotpotQA training questions, plus rehearsal rows. On a development set both beat main on the macro average (candC 0.657, candD 0.633, main 0.501) and both failed the guard fixed beforehand: answers when no passage in reach holds the answer rose from 17 to 70 and 72 of 282, against a limit of 25. They answer from on-topic passages even when the answer is absent, and answer the visible half of a compound question while inventing the other half. A weight blend of main and candD (20%, 35%, 50%) raised unsupported answers steadily (34, 39, 46 of 282), so no blend gains the skill inside the guard. On a fresh test (six sets, 739 episodes per system) candD gained +0.066 (+0.031 to +0.103) on the macro average and answered 32 of 57 fresh SQuAD questions against main's 4, but raised unsupported answers by +0.120 (limit +0.05). Main stayed. CandA tied candD on the macro score with a smaller safety cost, which is why we report candA as an alternative.

**Round 4 matched evidence decisions.** The same question and search depth, with the required fact present in one example and missing in the other (721 rows including replay of the search policy), starting from main and from candA, two epochs each: four checkpoints (candE1, candE2, candF1, candF2) for about 2 hours of dual-T4 time. The best development checkpoint, candF1, raised the single and two-part evidence-success macro from 39.2% to 52.7% and cut unsupported answers from 18 to 13 of 132 hidden-evidence episodes, but its HotpotQA success fell from 67.3% to 60.4% and its unsupported HotpotQA answers rose from 2 to 9 of 73. No checkpoint passed the release rules. A frozen diagnostic of candE1 on a final holdout did not show an upgrade (macro success 35.7% for main, 37.7% for candE1, interval crossing zero). A trace review found masking errors, where a hidden-evidence episode still showed another passage holding the answer, so we do not read raw unsupported counts as human-verified hallucination rates.

### 6.5 Round 5: two repairs that need no training, and a preference-training negative result

Round 4's traces suggested two cheap repairs. A rollout that never reaches a valid action (a repeated query, a search over budget, malformed output) was scored as a failure although the model had asserted nothing. And some answers were not backed by the passages the model itself cited. The plan put both repairs ahead of more training, with preference training (DPO) as the training step. Every rule was committed before the data it applied to existed, and the departures from the plan are dated in the pre-registration.

**The two repairs.**
- **Fail-closed.** A rollout that does not end in a valid answer or abstention ends as an abstention. This is a protocol repair: it changes the score because an abstention is the right ending on a hidden-evidence episode, and it does not make the model know more.
- **Self-check.** The main adapter reads the question, the cited passages and the answer, and replies `SUPPORTED` or `NOT`. The answer is withheld, and the episode ends as an abstention, when the probability of `SUPPORTED` is below 0.5, when the cited chunk was never shown, or when a part of the answer does not appear in the cited passages. It costs one forward pass of about 770 tokens per answer and generates no text.

Both repairs act on a finished episode, so we applied them offline to the saved rollouts. We then packaged them (`src/ragharness/system.py`, `scripts/ask.py`) and ran the packaged system on 182 sealed-test episodes (all 82 HotpotQA episodes and the first 100 RGB episodes). Every one ended as the offline evaluation had ended it, and the model's own ending before the repairs was identical in all 182. This confirms that the code reproduces the offline numbers, and is not a new accuracy test.

**Choosing the self-check, on development data only.** Four configurations (the base model or the main adapter as the checker, and two wordings), threshold fixed at 0.5. The gate, fixed beforehand: block at least 40% of main's 20 unsupported development answers and at most 10% of its 76 supported correct answers.

| checker | unsupported answers blocked (of 20) | supported correct blocked (of 76) | gate |
| --- | ---: | ---: | --- |
| base model, `SUPPORTED` / `NOT` | 60% | 24% | no |
| base model, `Yes` / `No` | 75% | 21% | no |
| **main adapter, `SUPPORTED` / `NOT`** | **70%** | **8%** | **yes** |
| main adapter, `Yes` / `No` | 100% | 87% | no, blocks nearly everything |

Twenty unsupported answers is a small sample. On the round-4 final-test traces, which were not used to choose the checker, it blocks 59% of main's 39 unsupported answers at a cost of 14% of its 51 supported correct ones.

**The sealed test.** 938 episodes built before any training from data no earlier round used: 385 SQuAD episodes (single and two-part questions and the round-4 evidence variants), 82 HotpotQA episodes and 471 RGB episodes (noise 0, 0.8 and 1). The oracle's 1.0 was the weaker success metric; strict scorer-v2 evidence success was 372/471 on RGB and 78/82 on HotpotQA. Citation and source-offset errors prevent that result from establishing that every answer was visible. Scorer version 2 was fixed before any round-5 result. An answer counts as correct when its token F1 and token precision against the gold answer are both at least 0.5 and it cites every required piece of evidence. Macro success is the mean of four component rates: SQuAD single, SQuAD two-part, HotpotQA and RGB. The set was opened for the reference systems (base, main, candA), and round 5b opened it once more, for one candidate.

| sealed test (938 episodes) | Tiny Aya base | main | main + fail-closed | **main + fail-closed + self-check** |
| --- | ---: | ---: | ---: | ---: |
| **macro success** | 0.022 | 0.414 | 0.474 | **0.517** |
| SQuAD single | 0.018 | 0.545 | 0.600 | 0.591 |
| SQuAD two-part | 0.036 | 0.224 | 0.327 | 0.394 |
| HotpotQA | 0.000 | 0.524 | 0.561 | 0.561 |
| RGB | 0.034 | 0.363 | 0.407 | 0.524 |
| answers although the evidence was hidden ↓ | 276 / 369 | 108 / 364 | 108 / 364 | **37 / 364** |
| supported correct, answerable episodes | 0.027 | 0.290 | 0.290 | 0.275 |
| refusals of answerable episodes ↓ | **0.011** | 0.222 | 0.270 | 0.391 |

Paired 95% intervals, resampling whole questions within each part (3,000 draws). Final system minus base: macro **+0.495 (+0.468 to +0.521)**, unsupported-answer rate **−0.646 (−0.698 to −0.591)**, supported correct **+0.249 (+0.203 to +0.295)**. Final system minus main: macro **+0.103 (+0.079 to +0.127)**, unsupported-answer rate **−0.195 (−0.231 to −0.159)**, supported correct −0.014 (−0.025 to −0.005). Fail-closed alone adds +0.059 (+0.041 to +0.079) macro and changes no unsupported-answer count. These scores use a stricter definition of success than rounds 1 to 3, on a different and harder mix of questions, so they do not compare with the 0.583 of the first round.

- **The cut in unsupported answers is the result.** The self-check removes 71 of main's 108 unsupported answers (29.7% to 10.2% of hidden-evidence episodes).
- **The accuracy gain over the base model comes from training.** Supported correct answers rise from 2.7% to 29.0% with the adapter, and the repairs leave them at 27.5%.
- **The price is more refusals.** Refusals of answerable episodes rise from 22% to 39%. Most of the added abstentions replace wrong answers, but a user sees more "I cannot answer".
- **candA with the same repairs** scores 0.024 higher on macro (+0.003 to +0.045) and is not different on unsupported answers (+0.022, −0.014 to +0.057). It was a reference and not a candidate under the rule, so main remains the default. It is the better choice if you prefer answers to caution.

**Preference training (DPO), judged and rejected.** We sampled 1,920 rollouts on training episodes (main and candA, temperature 0.8) and built 335 preference pairs: 186 favour a cited correct answer, 149 favour searching again or abstaining. Two LoRA continuations, from main (candG) and from candA (candH), two epochs of 84 steps each with the chosen-text likelihood as an anchor, gave four checkpoints. Development selection ran with both repairs applied to every system, and the first gate required strictly fewer unsupported answers than main with the repairs. We kept it as written. No checkpoint passed.

| development, both repairs on every system | macro | SQuAD single | supported correct | unsupported / hidden | HotpotQA unsupported | eligible |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| main | 0.538 | 0.603 | 0.228 | 5 / 131 | 1 | reference |
| candG2 (DPO from main) | 0.558 | 0.700 | 0.381 | 9 / 132 | 9 | no |
| candH2 (DPO from candA) | 0.557 | 0.678 | 0.386 | 12 / 129 | 8 | no |
| candF1 (round 4) | 0.583 | 0.612 | 0.312 | 7 / 131 | 2 | no |

DPO raised accuracy on SQuAD-style questions and lost the ability to abstain. None of the four checkpoints ends a single development episode on the round-4 set with an abstention, while main does so in 160 of 321. The cause is in the pair file: it rejects an abstention 91 times and chooses one twice, because on-policy sampling almost never produces an unsupported answer at the last search, which is where the reference action is an abstention. Training pushed abstentions down by about 17 nats and nothing pushed them up. With the repairs applied, the checkpoints' looping and unsupported answers turn into abstentions, so the wrapped numbers look reasonable, but the abstaining is then done by the repairs and not by the model.

**Departures from the plan, dated in the pre-registration.** Two samples per episode instead of four, 335 pairs instead of 1,500 to 3,000, and a learning rate of 2e-5 instead of 5e-6, all set by measured T4 throughput. The candidate pool was widened from four to eight checkpoints by adding round 4's, after an informed look at their test traces. The optimiser step is our own loop and not an AgentTune class (section 2). Compute was about 8 hours on a second Kaggle account's two T4s, including two failed launches.

### 6.6 Round 5b: preference training again, with the missing direction added

Round 5's preference pairs almost never chose an abstention, and the checkpoints stopped abstaining. Round 5b added that direction ([pre-registration](https://github.com/Bhaveshmeghwal21/against-the-safety/blob/main/research/2026-11-07-round5b-preregistration.md), committed before training, with three dated amendments).

- **New pairs.** We took 195 hidden-evidence contexts where the model itself had abstained after at least one search, and forced the same system to answer there (an `<answer>` prefix, temperature 0.7, at most 160 tokens). Each context gives a pair that chooses the abstention and rejects the forced answer. After caps and filters this gave 163 new pairs. Merged with the 335 round-5 pairs, the file has 498 pairs (SHA256 recorded before training). An abstention is the chosen side 165 times (round 5: twice) and the rejected side 91 times. A forced answer is sometimes right from the model's memory, so the pairs can teach it to withhold answers it could give; we recorded this before training.
- **Two runs, same recipe as round 5.** candI starts from main (2.4 hours on one T4). candK starts from candF1, the round-4 checkpoint that beat candA on the development table (2.9 hours). We chose that start after seeing the table, and say so in the pre-registration.
- **A new development gate.** Round 5's gate (strictly fewer unsupported answers than main with the repairs) was close to unpassable. Before training we replaced it with seven conditions measured against main with the repairs: the model's own abstention share on hidden-evidence episodes at least 0.75 of main's; unsupported rate within 3 points; HotpotQA unsupported answers within 2; supported correct at least +0.05; invalid turns within 1 point; HotpotQA success within 3 points; macro at least +3 points. Before registering it we ran it on the eight existing checkpoints and found that candF1 passes. **The gate is therefore informed by the development table.** The sealed-test rule was left exactly as in round 5.

| development, both repairs on every system | macro | HotpotQA success | supported correct | unsupported / hidden | own abstentions on hidden | eligible |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| main | 0.538 | 0.623 | 0.228 | 5 / 131 | 0.775 | reference |
| candF1 (round 4) | 0.583 | 0.642 | 0.312 | 7 / 131 | 0.775 | **yes** |
| candI1 (from main, epoch 1) | 0.528 | 0.560 | 0.233 | 3 / 132 | 0.829 | no |
| candI2 (epoch 2) | 0.566 | 0.579 | 0.360 | 7 / 132 | 0.634 | no |
| candK1 (from candF1, epoch 1) | 0.574 | 0.522 | 0.492 | 16 / 130 | 0.167 | no |
| candK2 (epoch 2) | 0.591 | 0.509 | 0.519 | 13 / 130 | 0.158 | no |

- **The added pairs restored abstention when training started from main.** candI1 and candI2 end 83% and 63% of hidden-evidence episodes in their own abstention. All four round-5 checkpoints ended none.
- **They still failed the gate.** candI1 fails on supported correct, HotpotQA success and macro. candI2 fails on HotpotQA unsupported answers (5 against a limit of 3), HotpotQA success and, by 0.002, macro.
- **Starting from candF1 did not keep abstention** (17% and 16%), and candK has 12 unsupported HotpotQA answers. We have no tested explanation for why the same pairs worked from main and not from candF1.

**The sealed test was opened for the one eligible checkpoint, candF1.** Both repairs on, the same 938 episodes:

| sealed test | macro | SQuAD single | SQuAD two-part | HotpotQA | RGB | unsupported / hidden | supported correct | refusals of answerable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **main** (submitted) | 0.517 | 0.591 | 0.394 | 0.561 | 0.524 | 37 / 364 | 0.275 | 0.391 |
| candF1 | 0.570 | 0.614 | 0.527 | 0.617 | 0.521 | 34 / 364 | 0.323 | 0.378 |

candF1 minus main, paired 95% intervals: macro **+0.052 (+0.021 to +0.085)**; unsupported-answer rate −0.008 (−0.047 to +0.030); supported correct +0.048 (+0.009 to +0.087). Without the repairs candF1 scores 0.449 macro with 114 of 364 unsupported answers (main: 0.414 and 108).

**The registered rule fails.** It needed the macro interval above zero, which holds, and an unsupported-rate difference below zero with an upper bound of at most +0.02. The point estimate is −0.008 and the upper bound is +0.030. **Main with both repairs stays the submitted system.** candF1 with the same repairs is the alternative for a reader who values answers over caution. It answers more correctly (0.323 against 0.275) and does better on two-part questions (0.527 against 0.394), and the data cannot say whether it is safer or less safe than main. Two cautions: a gate written after candF1's development results were known selected it, so this is one checkpoint tested once and not a family, and the unsupported counts (34 and 37) are small.

### 6.7 Limits

Evaluation sets are small (86 answerable questions per first-round split), so differences of a few points are not established. F1 ≥ 0.5 against the gold answer can credit near misses on short answers, so exact-match rates are in the reports. The work is English and short factual questions; the main adapter does not handle two-part answers or single-passage questions, and no later round fixed that safely. The check that masked episodes lack the answer is a text match, not a proof that no other passage could answer, and round 4's trace review found cases where it was wrong.

Rounds 5 and 5b add their own limits. The final system answers fully and correctly in only 27.5% of answerable sealed-test episodes (29.0% for the adapter alone) and refuses 39%; the gain is fewer unsupported answers, not more correct ones. The sealed test mixes harder variants than the first-round split and uses a stricter scorer, so its numbers do not compare with the first round's. The self-check was chosen among four configurations on 20 unsupported development answers. It held on round-4 test traces (59% to 67% of unsupported answers blocked) and on the sealed test (71 of 108), so the direction is established and the exact rate is not. The packaged system reproduced the offline decisions on 182 of 182 sealed-test episodes. It has been run on Kaggle only, with a local copy of the main adapter and not the Hub repository id, one question at a time. The sealed test has now been opened for the reference systems and for one candidate, so a further model would need a new sealed set. Round 5b's development gate was written after the earlier development table was seen, and its one eligible checkpoint was known to pass. Whether candF1 is safer, equal or less safe than main is not established.

## 7. What more credits would enable

More than two T4 GPUs and a longer quota window would let us:
1. Train the same recipe on all of HotpotQA's training set instead of 600 questions.
2. Run several seeds, so that systems that swap places between splits within their intervals can be ranked.
3. Generate training episodes for the failures that re-weighting could not fix: follow-up searches that fail to reach the second document (the model's own queries find the full evidence in only about 70% of open questions), single-passage questions taught without teaching the model to answer from any on-topic passage, and answers with several parts.
4. Evaluate on a document set from a real deployment with several editions of the same policy, which is where our 24-task check was weakest.
5. Train with a reward that scores the final decision directly (+1 for a supported correct answer, 0 for an abstention, −1 for a wrong or unsupported answer), with a KL penalty to the main adapter and a guard on the abstention rate. Round 5 showed that pairs built from errors alone remove the abstain action, and a reward on the final decision is the on-policy way to keep it.

## 8. Team

Team Against-the-safety. Add each teammate's Hugging Face username here before posting: `@username1`, `@username2`.
