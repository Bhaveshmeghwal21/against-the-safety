# Teaching a small Cohere model to search, cite and say "the documents don't say"

**Team Against-the-safety** · Lexsi AI Safety and Alignment Hackathon · Track 3 (agentic fine-tuning) · PS 4: fine-tuning for retrieval-augmented generation

Code: https://github.com/Bhaveshmeghwal21/against-the-safety

## Abstract

A document assistant is only useful if it answers from its documents and stops when they do not contain the answer. We fine-tuned Tiny Aya Global (3.35B, a Cohere model) with QLoRA, through AgentTune, to run a three-search retrieval loop and then either answer with citations to the passages it saw or abstain and say what is missing. Training used 600 HotpotQA questions turned into scripted episodes in six evidence states (clean support, distractors, second hop, missing evidence, evidence already supplied, weak first search), 1,611 rows in total. We trained two adapters under an identical recipe: a main adapter on the full mixture and a control on supported tool episodes only. We evaluated on a held-out test split, opened once, with the same retrieval index, call budget and decoding for every system. Scoring was deterministic code plus AuditKIT's RAG and agent metrics.

On the test split the end-to-end good rate (a supported answer when the documents hold it, an abstention when they do not) went from 0.020 for the base model to 0.583 for the main adapter (+0.563, 95% interval +0.490 to +0.636), and from 0.404 for the control to 0.583 (+0.179, +0.103 to +0.252). The base model answered from missing evidence in 78% of unanswerable episodes; main does so in 9.8%, the control in 26.8%. Citations to a gold passage when answering rose from 12.6% to 99.4%. The improvement has a price: the main adapter abstains on 19% of answerable episodes, and its supported-answer rate on answerable questions (0.527) is lower than the control's (0.707). The control never abstains and, when evidence is missing, mostly loops until it runs out of searches. In a pre-registered second round we drew 100 fresh questions (test2) that no training row or choice had touched. Main replicated there: 0.028 to 0.601, +0.573 (+0.503 to +0.643). A re-trained variant with fewer abstention examples cut the refusals (15% against 22%), at the cost of more unsupported answers (8% against 4%). The selection rule kept main, and both adapters are published. On five external benchmarks never used for training or selection (2WikiMultihopQA, MuSiQue, SQuAD 2.0, RGB and RGB integration; 1,131 episodes), main went from 0.079 to 0.499 (+0.420, +0.386 to +0.453) and improved on four of the five. It fails on two-part answers, and it declines most single-passage SQuAD questions even with the answer shown; neither question type was in its training data. A pre-registered third round patched both gaps. On fresh questions the patch answered single-passage SQuAD questions (supported answers 4 to 32 of 57) but raised pooled unsupported answers from 0.20 to 0.32 (limit 0.25), so it failed a safety guard fixed in advance, and main remains the model. A 24-task fictional-product check shows the same pattern, weak on two-document and policy-edition questions. Training cost 3.9 T4 GPU-hours on the free Kaggle quota (about 1.36 USD at a reference price).

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
| [against-the-safety-tiny-aya-rag-cand-a](https://huggingface.co/BMEscanor/against-the-safety-tiny-aya-rag-cand-a) | LoRA adapter, round 2, 30% abstentions (fewer refusals; not the selected model) | CC BY-NC 4.0 (inherited) |
| [against-the-safety-tiny-aya-rag-patch](https://huggingface.co/BMEscanor/against-the-safety-tiny-aya-rag-patch) | LoRA adapter, round 3 patch (candD): answers single-passage questions, fails the safety guard; not the selected model | CC BY-NC 4.0 (inherited) |

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

An Aya Expanse teacher and distillation, and reinforcement learning as the core method: SFT on matched states was the smallest change that could test the abstention question.

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

### Limits

Small evaluation sets (86 answerable questions per split); differences of a few points are not established. A second fresh split (test2, round 2) confirms the main result. F1 ≥ 0.5 against the gold answer can credit near-misses on short answers, so exact-match rates are in the reports (main 0.424, control 0.571, base 0.020 on answerable episodes). English, short factual questions; the main adapter does not handle two-part answers or single-passage questions, and the round-3 patch that does fails the safety guard. The pooled unsupported-answer rates are dominated by RGB's on-topic-passage setting, where every adapter is weak. The check that masked episodes lack the answer is a text match, not a proof that no other passage could answer.

## 7. What more credits would enable

More than two T4 GPUs and a longer quota window would let us do four things:
1. Train the same recipe on all of HotpotQA's training set instead of 600 questions.
2. Run several seeds, so that main and candA, which swap places between dev and test2 within their intervals, can actually be ranked.
3. Generate new training episodes for the failures that re-weighting could not fix. Round 3 tried the second and third of these with a 93-step patch; it fixed single-passage questions but not safely, so the full recipe, from scratch and with more steps, is what a larger budget would test:
   - follow-up searches that fail to reach the second document (the model's own queries find the full evidence in only about 70% of open questions);
   - questions that one passage answers, taught without teaching the model to answer from any on-topic passage;
   - answers with several parts, where the patch gained only 4 of 27.
4. Evaluate on a document set from a real deployment with several editions of the same policy, which is where our 24-task check was weakest.

## 8. Team

Team Against-the-safety. Add each teammate's Hugging Face username here before posting: `@username1`, `@username2`.
