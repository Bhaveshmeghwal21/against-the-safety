# Teaching a small Cohere model to search, cite and say "the documents don't say"

**Team Against-the-safety** · Lexsi AI Safety and Alignment Hackathon · Track 3 (agentic fine-tuning) · PS 4: fine-tuning for retrieval-augmented generation

Code: https://github.com/Bhaveshmeghwal21/against-the-safety

## Abstract

A document assistant is only useful if it answers from its documents and stops when they do not contain the answer. We fine-tuned Tiny Aya Global (3.35B, a Cohere model) with QLoRA, through AgentTune, to run a three-search retrieval loop and then either answer with citations to the passages it saw or abstain. Training used 600 HotpotQA questions turned into 1,611 scripted episodes in six evidence states, including missing evidence. A main adapter learned the full mixture; a control learned supported tool episodes only. Every system ran with the same index, call budget and decoding, and was scored by deterministic code plus AuditKIT's RAG and agent metrics. Later choices were fixed in dated pre-registrations before their data was seen.

First round, on a held-out HotpotQA test split opened once: the share of episodes that end well (a supported answer when the documents hold it, an abstention when they do not) rose from 0.020 for the base model to 0.583 for main (+0.563, 95% interval +0.490 to +0.636), against 0.404 for the control. The base model answered from missing evidence in 78% of unanswerable episodes; main does so in 9.8%. Main declines 19% of answerable questions. A second split of 100 fresh questions replicated the result (0.028 to 0.601). On five external benchmarks (1,131 episodes) main went from 0.079 to 0.499 and improved on four. It fails on two-part answers and refuses most single-passage SQuAD questions.

We tried three fixes, each judged by rules set beforehand, and rejected all three. A SQuAD patch answered single-passage questions (supported answers 4 to 32 of 57) but raised unsupported SQuAD answers from 6 to 31 of 109, failing its safety guard. A matched-evidence curriculum improved controlled exercises, and no checkpoint passed the release rules. Preference training (DPO) raised SQuAD accuracy, but all four checkpoints stopped abstaining, because our 335 pairs rejected an abstention 91 times and chose one twice.

What worked needed no further training: two repairs on the main adapter's output. Fail-closed ends a rollout without a valid final action as an abstention. A self-check has the same model read each answer against its cited passages and withhold it if unsupported (one forward pass per answer). On a sealed test of 938 fresh SQuAD, HotpotQA and RGB episodes, opened once, macro success rose from 0.022 for the base model to 0.517 (+0.495, +0.468 to +0.521). Answers given although the evidence was hidden fell from 276 of 369 episodes to 37 of 364, and supported correct answers on answerable episodes rose from 2.7% to 27.5%. Against the adapter alone, the repairs add +0.103 macro (+0.079 to +0.127) and cut unsupported answers from 29.7% to 10.2%. They cost caution: refusals of answerable questions rise from 22% to 39%, and supported correct answers do not increase (29.0% to 27.5%).

Training the main adapter cost 3.9 T4 GPU-hours on the free Kaggle quota.

## 1. Problem and approach

PS 4 splits retrieval-augmented generation into five decisions: when to retrieve, what to retrieve, whether the answer is grounded, whether distractors are ignored, and whether the model abstains when the documents cannot answer. A small model usually gets the second step roughly right and fails the others: the base Tiny Aya answered 78% of questions whose evidence we had removed, and searched even when the evidence was already in the prompt (60 of 60 test episodes).

We treat this as behaviour cloning on matched evidence states. For each training question the episode builder uses the same scripted policy under different masks and pre-supplied evidence, so the only thing that tells the model whether to answer or abstain is what is visible in the conversation. The six states: clean support after one search (S1), a distractor ranked above the support (S2), second-hop recovery (S3), all support masked so the right move is abstain (S4), evidence given in the prompt fully or partly (S5), and a weak or masked first search recovered by a reworded query (S6). Masked episodes are rejected when the gold answer text still appears in a visible passage, and questions whose support the scripted search cannot reach within three searches are reported and never trained on.

The model sees a small protocol: a `search_corpus` call as JSON, passages with their chunk ids, then `<answer>…</answer><citations>…</citations>` or `<abstain>…</abstain>`. Budget: three searches, three passages per search, 800 characters shown per passage.

Two adapters share one recipe fixed before any result (QLoRA r=16, alpha=32, all linear layers, learning rate 2e-4, effective batch 16, 2 epochs, seed 13):

- **main**: 1,611 rows, 537 each of search, answer and abstain turns, from all six states.
- **control**: 1,697 rows of supported tool episodes only. It answers the same way but has never seen an abstention. It tests whether the abstention data, not just tool-use practice, produced the change.

The development split chose the epoch (epoch 2: 0.673 end-to-end good against 0.270 at epoch 1) and nothing else. The test split was opened once.

This first-round adapter is the base of everything that follows. Rounds 2 to 4 tried to improve it with more training and none passed its pre-registered rules. Round 5 adopted two output-side repairs that need no training, and rejected preference training (section 6).

## 2. Lexsi repositories and features used

- **AgentTune**: `create_distill_trainer` with our QLoRA trainer factory for training; `SQLiteFTSBackend` and `SearchCorpusTool` for retrieval and the tool the model calls; `EventLog` and `TrajectoryStore` for every training episode and every evaluation rollout; `qa_metrics` F1 and exact match.
- **AuditKIT**: the RAG metrics `evidence_set_recall`, `citation_support`, `abstention` and `freshness`, and the agent metrics `tool_call_validity`, `redundant_tool_calls` and `agent_loop_detection`. Each episode is an `ak.Sample` with a `RAGTrace` and is scored with `ak.evaluate`.

Two integration problems we hit and documented in the repository: TRL casts LoRA weights of a quantized model to bf16, which the T4's fp16 gradient scaler rejects (we cast them back to fp32), and AgentTune's SQLite backend returns "no results" instead of an error when the Python build lacks FTS5, which happened on Kaggle's default Python (we run a uv-managed Python 3.12 and verify the index with a probe search before every evaluation).

In round 5, every rollout still runs through AgentTune's tool, trajectory and evaluation stack, but the preference-training step does not use an AgentTune class. TRL's `DPOTrainer`, wrapped by AgentTune's `TrlAgenticDPO`, builds full-vocabulary logits (262k entries) for both sequences of a pair and ran out of memory on a 15 GB T4, so only 38 of 335 pairs would fit. We replaced it with a short loop (`scripts/train_dpo.py`) that computes logits at the completion positions only and uses the same loss. The round-5 headline numbers come from the outcome classes of our scorer version 2, not from AuditKIT metrics.

## 3. Models and licenses

| model | what it is | license |
| --- | --- | --- |
| [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) (revision `af89d219`) | base and student model, Cohere | CC BY-NC 4.0, gated |
| [against-the-safety-tiny-aya-rag-main](https://huggingface.co/BMEscanor/against-the-safety-tiny-aya-rag-main) | LoRA adapter, full mixture | CC BY-NC 4.0 (inherited) |
| [against-the-safety-tiny-aya-rag-control](https://huggingface.co/BMEscanor/against-the-safety-tiny-aya-rag-control) | LoRA adapter, supported-only control | CC BY-NC 4.0 (inherited) |
| [against-the-safety-tiny-aya-rag-cand-a](https://huggingface.co/BMEscanor/against-the-safety-tiny-aya-rag-cand-a) | LoRA adapter, round 2, 30% abstentions (fewer refusals; not the selected model) | CC BY-NC 4.0 (inherited) |
| [against-the-safety-tiny-aya-rag-patch](https://huggingface.co/BMEscanor/against-the-safety-tiny-aya-rag-patch) | LoRA adapter, round 3 patch (candD): answers single-passage questions, fails the safety guard; not the selected model | CC BY-NC 4.0 (inherited) |

The submitted system is the main adapter plus two output-side repairs from round 5 (a fail-closed rule and a self-check, section 6). They are code in the repository and add no weights. Round 4's checkpoints and round 5's four preference-trained checkpoints were not selected and are not published.

## 4. Datasets and benchmarks, with licenses

| dataset | contents | license |
| --- | --- | --- |
| [against-the-safety-rag-episodes](https://huggingface.co/datasets/BMEscanor/against-the-safety-rag-episodes) | training rows (main, control, round-2 candidates), AgentTune event logs, dev, test and test2 evaluation traces, round-3 patch sets and development traces, reports | CC BY-SA 4.0 (derived from HotpotQA) |
| [nimbus-home-support-suite](https://huggingface.co/datasets/BMEscanor/nimbus-home-support-suite) | 24 fictional product documents, 24 tasks, evaluation traces | CC BY 4.0 |
| [against-the-safety-benchmark-traces](https://huggingface.co/datasets/BMEscanor/against-the-safety-benchmark-traces) | evaluation traces and reports on the five external benchmarks | per folder, as the source benchmark (below) |
| HotpotQA (distractor) | source of questions and passages | CC BY-SA 4.0 |
| 2WikiMultihopQA (dev) | external benchmark | Apache-2.0 |
| MuSiQue-Ans (dev, 2-hop) | external benchmark | CC BY 4.0 |
| SQuAD 2.0 (dev; train split for the round-3 patch rows) | external benchmark; round-3 training and development episodes | CC BY-SA 4.0 |
| RGB (English refined and integration sets) | external benchmark | CC BY-NC-SA 4.0 |

Round 4 and 5 data come from the same sources (SQuAD 2.0 training-split articles no earlier round used, HotpotQA, RGB). Round 5's reports, selection tables, the preference-pair file hash and the sealed-test manifest with file hashes are in the repository under `reports/`.

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

Moving from 0 to −6 halves the false abstains but turns most of the correct abstentions into unsupported answers; at −9 abstaining is switched off. Between 0 and −3 almost nothing changes, which says the abstain decision is nearly all-or-nothing in this adapter, not a confidence that can be thresholded. Reducing the 19% false-abstain rate needs different training data, not a decoding setting; round 2 below tests a smaller share of abstention episodes.

### Alternatives we considered and did not run

An Aya Expanse teacher and distillation, and reinforcement learning as the core method: SFT on matched states was the smallest change that could test the abstention question. Round 5 later added preference training (DPO) on top of the SFT adapters, with the result below. Reinforcement learning with an abstention-aware reward (GRPO) was planned and not run.

### Round 2: a fresh test split and an attempt to cut the refusals

Choosing a new training mixture after opening the test split would have turned the test split into a selection set. So round 2 drew **test2**, 100 new HotpotQA questions (80 bridge, 20 comparison, all hard). The sampling rule skipped any question whose supporting title is the title of any passage already in the corpus, so no test2 support passage appears in any training row. The test2 passages were appended to the index without renumbering anything. Every rule below was written and pushed before training.

**Diagnosis from saved data.**
- In main's training set, abstentions are half of the final decisions, and 81% of the final decisions after three searches are abstentions.
- On dev, main copies this: it abstains on 86% of its final turns after three searches. 16 of its 36 false abstentions come after all the support was on screen.

**Two candidates.** Both use main's recipe, questions and token budget (within 0.1%).
- **candA** lowers abstentions to 30% of final decisions.
- **candB** does the same, and also:
  - sets an equal 40% abstain rate after two and after three searches;
  - pairs answers and abstentions from the same question;
  - spreads search examples evenly over the first, second and third search.

**Selection on dev, by the pre-registered rule.**
- Eligible candidates: at most 11 of 73 unsupported answers, and at most 10% of episodes without a valid final turn.
- Among the eligible, the higher end-to-end good rate wins.
- The winner replaces main only if it is at least as good as main.

| dev | main | candA | candB |
| --- | ---: | ---: | ---: |
| end-to-end good (159) | **0.673** | 0.629 | 0.604 |
| false abstain (205) | 0.176 | **0.088** | 0.166 |
| unsupported answers (73) | **2** | 9 | 5 |

CandA won the candidate comparison but scored below main, so main stayed the model. Main and candA then each ran once on test2:

| test2 | base | **main** | candA | control |
| --- | ---: | ---: | ---: | ---: |
| end-to-end good (143) | 0.028 | **0.601** | 0.657 | 0.280 |
| supported correct (199) | 0.025 | 0.462 | 0.528 | **0.598** |
| false abstain (199) ↓ | **0.000** | 0.221 | 0.146 | **0.000** |
| unsupported answer (74) ↓ | 0.878 | **0.041** | 0.081 | 0.189 |
| invalid turns (273) ↓ | 0.205 | 0.037 | **0.011** | 0.293 |

Main minus base: +0.573 (95% interval +0.503 to +0.643). Main minus control: +0.322 (+0.246 to +0.392). Main minus candA: −0.056 (−0.122 to +0.007).

What round 2 established:
- **The first-round result replicates on fresh questions.** Main's end-to-end good rate is 0.601, against 0.583 on the first test split.
- **Fewer abstentions in training lowers refusals on both splits.** CandA refuses less and gives more supported answers, and it answers somewhat more often from missing evidence. Its end-to-end rank against main flips between dev and test2 within the intervals. At this sample size, the two are different operating points rather than better and worse models, and both are published.
- **The search-count pattern was a symptom, not the cause.** CandB was trained without it, yet it still abstains mostly after many searches, and its refusals did not fall (34 against 36 on dev). Unanswerable episodes use up more searches.
- **Re-weighting the existing examples did not improve retrieval.** The model's own queries reached the full evidence in 56, 58 and 59 of 86 open dev questions (main, candA, candB).
- **Cost:** about 4 T4-hours of training for both candidates together, and about 2.5 more hours of evaluation.

### External benchmarks

We then asked whether the behaviour holds on data we neither built nor used for any choice. Five public sets were fixed before any model ran on them, with their settings and metrics (plan section 22 in the repository). Each benchmark gets its own index and the same harness as HotpotQA: three searches, three passages, 800 characters. The scores therefore compare our systems with each other and are not comparable with official leaderboards, which hand the model the gold paragraphs.

- **2WikiMultihopQA** (Apache-2.0): 100 dev questions, 87 reachable within three searches.
- **MuSiQue-Ans 2-hop** (CC BY 4.0): 200 dev questions, 88 reachable.
- **SQuAD 2.0** (CC BY-SA 4.0): 80 answerable dev questions and 80 human-written unanswerable ones.
- **RGB** (CC BY-NC-SA 4.0): 100 questions, run with search and also in RGB's own settings with five passages in the prompt (noise 0, 0.8 and 1).
- **RGB integration:** 60 questions whose answer has two parts in different passages.

The RGB counterfactual set was left out, because it rewards overriding the documents.

| ends well | base | **main** | candA | control | main minus base (95% interval) |
| --- | ---: | ---: | ---: | ---: | --- |
| 2WikiMultihopQA (166) | 0.018 | **0.566** | 0.620 | 0.241 | +0.548 (+0.479 to +0.619) |
| MuSiQue 2-hop (168) | 0.018 | **0.536** | 0.577 | 0.250 | +0.518 (+0.455 to +0.581) |
| SQuAD 2.0 (202) | 0.010 | **0.475** | 0.559 | 0.040 | +0.465 (+0.404 to +0.528) |
| RGB (453) | 0.157 | **0.618** | 0.660 | 0.530 | +0.461 (+0.406 to +0.518) |
| RGB integration (142) | 0.070 | **0.028** | 0.049 | 0.007 | −0.042 (−0.097 to +0.008) |
| pooled (1,131) | 0.079 | **0.499** | 0.547 | 0.293 | +0.420 (+0.386 to +0.453) |

What transfers and what does not:
- **Abstention transfers.** With the support hidden, main abstains in 85%, 85%, 71% and 52% of episodes on the first four sets; the base model in 4% or fewer. It declines 57 of SQuAD 2.0's 80 human-written unanswerable questions (base: 0).
- **Two-part answers do not.** On RGB integration, main cites both passages but answers with the first part only ("Wyndham Clark" for the winners of two golf majors). Accuracy with clean passages falls from 0.383 for the base model to 0.017. All training answers were single HotpotQA spans.
- **Single-passage questions are refused.** On SQuAD 2.0, main declines 54 of 73 answerable questions, and in every one of those 54 episodes the answer text was already in the results of its first search. Every training question needed two passages, and the adapter appears to have learned that one passage is never enough. On HotpotQA test2, by comparison, 16 of its 30 false abstentions had the evidence on screen.
- **On-topic passages that don't answer, supplied in the prompt, are often trusted.** At RGB noise 1, main declines 26 of 100 times; when the same answers are hidden from search instead, it abstains on 52%.
- **CandA transfers better than main.** It is ahead on all five sets, by 0.049 pooled (0.026 to 0.072), with more supported answers (133 against 105 of 346) and more unsupported ones (128 against 102 of 465). Main stays the selected model by the pre-registered rule, and candA is the better choice when an answer matters more than caution.
- **Cost:** two kernels of 2.8 and 2.5 hours on T4 x2.

### Round 3: a patch for the gaps the benchmarks found, judged by rules fixed first

The benchmarks showed gaps that the first-round data could not close because it never contained them: questions that one passage answers, answers with two parts, and on-topic passages without the answer. Every rule below was written and pushed before training (plan sections 24 to 26).

**The patch.** Two candidates continue from the main adapter for one epoch (93 steps, 2.2 M tokens). The new episodes are scripted under real search, like the first round's:
- 240 single-passage questions from the SQuAD 2.0 *training* split, and 120 compound questions made of two such questions from one article, with the answer "A1; A2";
- prompt variants, also on the 600 HotpotQA training questions: the support plus three on-topic passages without the answer, supplied in the prompt (answer without searching); on-topic passages only, support masked (search, then abstain); one of two supports masked (abstain);
- the rest of each set is rehearsal from main's training rows. candC is the stronger patch (65% new rows), candD the careful one (45%).

SQuAD's natural unanswerable questions were never used for training, so declining them stays a transfer test. Because SQuAD 2.0 dev is no longer zero-shot for these models, the evidence on fresh questions comes from six sets, of which RGB stays zero-shot.

**Development result.** On held-out SQuAD training-split articles plus the HotpotQA dev split, both candidates beat main on the macro average of four components (candC 0.657, candD 0.633, main 0.501) and cut invalid turns from 67 to 4 of 578. Both also failed the guard. Answers given when no passage in reach holds the answer rose from 17 to 70 and 72 of 282, against a limit of 25. The patched models answer from on-topic passages even when the answer is absent, and answer the visible half of a compound question while inventing the other half ("Constitutional Reform Act 2005; 2004"). **No candidate was eligible, and main stayed.** A probe added afterwards blended the main and candD weights (20%, 35%, 50%). Unsupported answers rose steadily (34, 39, 46 of 282), so no blend gains the skill inside the guard.

**Fresh test result** (six sets, 739 episodes per system, questions disjoint from every earlier set, opened once):

| | main | candD (round 3) | candA (round 2) |
| --- | ---: | ---: | ---: |
| macro average, ends well | 0.489 | **0.555** | 0.549 |
| difference from main, 95% CI | | +0.066 [+0.031, +0.103] | +0.061 [+0.028, +0.093] |
| answers when no passage in reach holds the answer ↓ | **0.203** | 0.322 | 0.233 |
| difference from main, 95% CI | | +0.120 [+0.076, +0.163] | +0.031 [−0.003, +0.064] |
| SQuAD 2.0, supported answers on open search (of 57) | 4 | **32** | 6 |
| SQuAD 2.0, answers when the passage cannot answer (of 109) ↓ | **6** | 31 | **6** |

- **The patch worked as built.** Fresh SQuAD questions went from 4 to 32 supported answers of 57, invalid turns from 30 to 3 of 166. Two-part answers improved only from 0 to 4 of 27.
- **It failed the safety criterion, fixed beforehand** (unsupported answers within 0.05 of main's; measured +0.120). So candD is published as a higher-answering variant, and main remains the headline model.
- **Round 3 found no better operating point than round 2.** CandA ties candD on the macro score with a smaller safety cost. It gains by abstaining correctly and by fewer invalid turns, and it still declines single-passage questions. On three independent splits (test2, the benchmark suite, this one) candA is 0.05 to 0.06 ahead of main on ends well and gives 3 to 6 points more unsupported answers. The round-2 rule chose main on development data, and we did not change that after seeing test data. A switch would need its own pre-registered rule and a new split, and we had neither the quota nor the time.

Four things the procedure cost us or taught us. The first probe launch failed after 7 minutes because the blended adapters lacked the tokenizer files. Compute used was 5.5 of the 6 hours left. The interpolation probe and the test of non-eligible candidates were decided after the development result, and the plan labels them. A bootstrap ordering detail changed an interval in the third decimal until we sorted its keys.

### Round 4: matched evidence decisions, not adopted

Round 3's patch taught answering without teaching when the evidence is missing. Round 4 trained a curriculum of matched decisions: the same question and search depth, with the required fact present in one example and missing in the other. The 721-row mixture includes replay of the search policy. Two initialisations, main and candA, each saved two epochs, which gives four checkpoints (candE1, candE2, candF1, candF2) for about 2 hours of dual-T4 time.

On development data the best checkpoint, candF1, raised the macro of single-passage and two-part evidence success from 39.2% to 52.7% and cut unsupported answers from 18 to 13 of 132 hidden-evidence episodes. Its HotpotQA success fell from 67.3% to 60.4%, and its unsupported HotpotQA answers rose from 2 to 9 of 73. No checkpoint passed the release rules, so main stayed. A frozen diagnostic of candE1 on a final holdout did not show an upgrade either: macro success was 35.7% for main and 37.7% for candE1, with an interval that crosses zero, and candE1 had more query-loop failures. A review of the traces found masking errors, where a hidden-evidence episode still showed another passage holding the answer. We therefore do not read the raw unsupported counts as human-verified hallucination rates.

### Round 5: two repairs that need no training, and a preference-training negative result

Round 4's traces suggested two cheap repairs. A rollout that never reaches a valid action (a repeated query, a search over budget, malformed output) was scored as a failure although the model had asserted nothing. And some answers were not backed by the passages the model itself cited. The plan put both repairs ahead of more training, with preference training (DPO) as the training step. Every rule was committed before the data it applied to existed, and the departures from the plan are listed below with their dates.

**The two repairs.**
- **Fail-closed.** A rollout that does not end in a valid answer or abstention ends as an abstention. This is a protocol repair. It changes the score because an abstention is the right ending on a hidden-evidence episode. It does not make the model know more.
- **Self-check.** The main adapter reads the question, the cited passages and the answer, and replies `SUPPORTED` or `NOT`. The answer is withheld, and the episode ends as an abstention, when the probability of `SUPPORTED` is below 0.5, when the cited chunk was never shown, or when a part of the answer does not appear in the cited passages. It costs one forward pass of about 770 tokens per answer and generates no text.

Both repairs act only on a finished episode. We applied them offline to the saved rollouts, which gives the same result as running them at inference time. The repository has no packaged inference wrapper yet.

**Choosing the self-check, on development data only.** Four configurations (the base model or the main adapter as the checker, and two wordings), threshold fixed at 0.5. The gate, fixed beforehand: block at least 40% of main's 20 unsupported development answers and at most 10% of its 76 supported correct answers.

| checker | unsupported answers blocked (of 20) | supported correct blocked (of 76) | gate |
| --- | ---: | ---: | --- |
| base model, `SUPPORTED` / `NOT` | 60% | 24% | no |
| base model, `Yes` / `No` | 75% | 21% | no |
| **main adapter, `SUPPORTED` / `NOT`** | **70%** | **8%** | **yes** |
| main adapter, `Yes` / `No` | 100% | 87% | no, blocks nearly everything |

Twenty unsupported answers is a small sample. On the round-4 final-test traces, which were not used to choose the checker, it blocks 59% of main's 39 unsupported answers at a cost of 14% of its 51 supported correct ones (candA: 67% at 5%).

**The sealed test.** 938 episodes built before any training from data no earlier round used: 385 SQuAD episodes (25 unused training-split articles, with single and two-part questions and the round-4 evidence variants), 82 HotpotQA episodes (unused validation questions) and 471 RGB episodes (noise 0, 0.8 and 1). A scripted oracle scores 1.0 on all three parts, so every episode can be solved. Scorer version 2 was fixed before any round-5 result. An answer counts as correct when its token F1 and token precision against the gold answer are both at least 0.5 and it cites every required piece of evidence. Hidden-evidence episodes whose shown passages already hold the whole answer are dropped. Macro success is the mean of four component rates: SQuAD single, SQuAD two-part, HotpotQA and RGB. The set was opened once, for the reference systems only. No preference-trained checkpoint was eligible, so none was run on it.

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

What the numbers say:
- **The cut in unsupported answers is the result.** The self-check removes 71 of main's 108 unsupported answers (29.7% to 10.2% of hidden-evidence episodes).
- **The accuracy gain over the base model comes from training.** Supported correct answers rise from 2.7% to 29.0% with the adapter. The repairs leave them at 27.5%.
- **The price is more refusals.** Refusals of answerable episodes rise from 22% to 39%. Most of the added abstentions replace wrong answers, but a user sees more "I cannot answer".
- **candA with the same repairs scores 0.024 higher on macro** (+0.003 to +0.045) and is not different on unsupported answers (+0.022, −0.014 to +0.057). It was a reference and not a candidate under the rule, so main remains the default. It is the better choice if you prefer answers to caution.

**Preference training (DPO), judged and rejected.** We sampled 1,920 rollouts on training episodes (main and candA, temperature 0.8, two samples each) and built 335 preference pairs: 186 favour a cited correct answer, 149 favour searching again or abstaining. Two LoRA continuations from main (candG) and from candA (candH), two epochs of 84 steps each, with the chosen-text likelihood as an anchor, gave four checkpoints. Training loss fell from 0.67 to 0.19. Development selection ran with both repairs applied to every system, so the first gate (strictly fewer unsupported answers than main with the repairs) is hard to pass. We kept it as written. No checkpoint passed.

| development, both repairs on every system | macro | SQuAD single | supported correct | unsupported / hidden | HotpotQA unsupported | eligible |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| main | 0.538 | 0.603 | 0.228 | 5 / 131 | 1 | reference |
| candG2 (DPO from main) | 0.558 | 0.700 | 0.381 | 9 / 132 | 9 | no |
| candH2 (DPO from candA) | 0.557 | 0.678 | 0.386 | 12 / 129 | 8 | no |
| candF1 (round 4) | 0.583 | 0.612 | 0.312 | 7 / 131 | 2 | no |

DPO raised accuracy on SQuAD-style questions and lost the ability to abstain. None of the four checkpoints ends a single development episode on the round-4 set with an abstention, while main does so in 160 of 321. The cause is in the pair file. It rejects an abstention 91 times and chooses one twice, because on-policy sampling almost never produces an unsupported answer at the last search, which is where the reference action is an abstention. Training pushed abstentions down by about 17 nats and nothing pushed them up. With the repairs applied, the checkpoints' looping and unsupported answers turn into abstentions, so the wrapped numbers look reasonable, but the abstaining is then done by the repairs and not by the model. A preference set must contain both directions for every action it touches. We did not run a repaired set.

**Departures from the plan, all dated in the pre-registration.**
- Two samples per episode instead of four, 335 pairs instead of 1,500 to 3,000, and a learning rate of 2e-5 instead of 5e-6, all set by measured T4 throughput.
- The candidate pool was widened from four to eight checkpoints by adding round 4's, after an informed look at their test traces (stated at the time). Selection used development data only.
- The optimiser step is our own loop and not an AgentTune class (section 2).
- Compute: about 8 hours on a second Kaggle account's two T4s, including two failed launches (a bug in a memory probe, and the out-of-memory error above).

### Limits

Small evaluation sets (86 answerable questions per split); differences of a few points are not established. A second fresh split (test2, round 2) confirms the main result. F1 ≥ 0.5 against the gold answer can credit near-misses on short answers, so exact-match rates are in the reports (main 0.424, control 0.571, base 0.020 on answerable episodes). English, short factual questions; the main adapter does not handle two-part answers or single-passage questions, and the round-3 patch that does fails the safety guard. Round 4's curriculum and round 5's preference training did not pass their release rules either. The pooled unsupported-answer rates are dominated by RGB's on-topic-passage setting, where every adapter is weak. The check that masked episodes lack the answer is a text match, not a proof that no other passage could answer, and round 4's trace review found cases where it was wrong.

Round 5 adds its own limits. The final system answers fully and correctly in only 27.5% of answerable sealed-test episodes (29.0% for the adapter alone) and refuses 39%; the gain is fewer unsupported answers, not more correct ones. The sealed test mixes harder variants than the first-round split and uses a stricter scorer, so its numbers do not compare with the first round's. The self-check was chosen among four configurations on 20 unsupported development answers. It held on round-4 test traces (59% to 67% of unsupported answers blocked) and on the sealed test (71 of 108), so the direction is established and the exact rate is not. Both repairs were applied offline to saved rollouts, which is equivalent to running them at inference time, and there is no packaged wrapper. Preference training failed, and we do not know whether a repaired pair set would help.

## 7. What more credits would enable

More than two T4 GPUs and a longer quota window would let us do five things:
1. Train the same recipe on all of HotpotQA's training set instead of 600 questions.
2. Run several seeds, so that main and candA, which swap places between dev and test2 within their intervals, can actually be ranked.
3. Generate new training episodes for the failures that re-weighting could not fix. Round 3 tried the second and third of these with a 93-step patch; it fixed single-passage questions but not safely, so the full recipe, from scratch and with more steps, is what a larger budget would test:
   - follow-up searches that fail to reach the second document (the model's own queries find the full evidence in only about 70% of open questions);
   - questions that one passage answers, taught without teaching the model to answer from any on-topic passage;
   - answers with several parts, where the patch gained only 4 of 27.
4. Evaluate on a document set from a real deployment with several editions of the same policy, which is where our 24-task check was weakest.
5. Train with a reward that scores the final decision directly (+1 for a supported correct answer, 0 for an abstention, −1 for a wrong or unsupported answer), with a KL penalty to the main adapter and a guard on the abstention rate, or rebuild the preference pairs so that they also choose abstentions. Round 5 showed that pairs built from errors alone remove the abstain action. Either run needs about 4 T4 hours per attempt and a fresh pre-registered test.

## 8. Team

Team Against-the-safety. Add each teammate's Hugging Face username here before posting: `@username1`, `@username2`.
