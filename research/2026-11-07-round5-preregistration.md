# Round 5, priority 2: preference-pair sampling and DPO design, fixed before any sample is read

Written 7 November 2026, 01:35 IST. This note is the dated record for section 27 of `research/2026-11-04-fine-tuning-for-rag.md`.
No round-5 sample, pair or dev result existed when it was written. Anything that differs from section 27 is listed under "Deviations".

## Priority 0a result (already computed, no new model calls)

`scripts/wrapper_round4.py` on the round-4 final-test traces of main and candA (pre-registered in commit `00e1c25`):

| system | macro | unsupported / 187 | supported success | false refusals | invalid |
| --- | ---: | ---: | ---: | ---: | ---: |
| main | 0.357 | 40 | 0.202 | 0.302 | 0.111 |
| FB(main) | 0.427 | 40 | 0.202 | 0.374 | 0.000 |
| G(main, candA) | 0.446 | 25 | 0.176 | 0.531 | 0.000 |

All three pre-registered claims hold: FB(main) raises macro by +0.070 [+0.046, +0.099] with no extra unsupported answers; G lowers the unsupported rate by 0.080 [-0.126, -0.042] and does not lower macro (+0.089 [+0.058, +0.122]). Per section 27, **FB joins every round-5 system**; G stays a non-headline option (two models per answer, and its false-refusal rate rises to 0.531). The FB gain comes entirely from ending invalid turns as abstentions on hidden-evidence episodes, so the write-up must say it is a protocol repair, not a model improvement.

## Data (all training-split material)

Article allocation (`scripts/prepare_round5.py`, seed 53, `reports/round5_allocation.json`): the 442 SQuAD v2 training-split articles are partitioned into round-3 dev (25), round-3 train (80), round-4 train/dev/test (60/15/20), and the 242 never used. Of these, 50 become **r5train** (pair sampling), 25 become **r5test** (sealed final test), 167 stay in reserve. Group overlap is asserted to be zero.

Sampling sets, each rolled out k = 2 times per system at temperature 0.8, top-p 0.95, top-k off, lenient parser, three-search budget, 160 new tokens:

- `r5train`: 49 questions (32 single, 17 compound), 311 episodes (130 hidden-evidence): open, support_masked and the round-4 prompt variants (full, missing, partial, recoverable).
- `hotpot_train`: 32 HotpotQA training questions (stratified, seed 53), 169 episodes (44 hidden-evidence): open, support_masked, first_search_masked, prefill_sufficient, prefill_partial, prefill_partial_masked.
- Systems: main (GPU 0, seed 5301) and candA (GPU 1, seed 5302). Total 1,920 rollouts.

## Pair construction (fixed now)

For every sampled rollout, the context is the message list before its final step and the rejected text is the final step. The reference action depends only on what the context shows:

1. *Complete*: every required evidence aspect (the round-4 `aspect_coverage` facets) has a shown chunk.
2. Answerable and complete: the reference is `<answer>gold</answer><citations>shown support ids</citations>`. A final step is rejected when it is an abstention, a wrong answer, an answer that cites an unshown or non-support chunk, or an invalid turn.
3. Not complete (answerable or hidden): the reference is the scripted teacher's next unused public query (`policy.next_query`) while fewer than three searches have been made, otherwise `<abstain>` with the training-set abstention sentence (`episodes.ABSTAIN_TEXT`). A final step is rejected when it is an answer or an invalid turn, or an abstention made before the third search on an answerable episode. An abstention at the third search is never rejected.
4. Hidden-evidence episodes in which any shown passage contains an accepted gold answer form (visible text, whole word) are dropped, because the answer is then supported. This is the training-side form of the priority-0c false-unsupported fix.
5. Pairs are dropped when the prompt plus the longer completion exceeds 2,560 real tokens, when the chosen and rejected texts are equal, or when the same (context, rejected) pair appears twice. At most 2 pairs per (question, variant) and 4 per question.
6. Pro-answer pairs (rule 2) and pro-caution pairs (rule 3) are balanced: the larger group is subsampled (seeded) to at most 1.25 times the smaller. Within each group, candidates are ordered by a seeded shuffle, not by length or outcome.
7. Everything is a function of the sampled files and the local corpora; no test or development episode is read.

## Training (priority 2 proper)

AgentTune `TrlAgenticDPO` (offline mode, TRL `DPOTrainer`) on 4-bit Tiny Aya Global. LoRA continues from the starting adapter; TRL adds a frozen copy named `ref` as the reference policy, so the reference is the starting adapter, not the bare base.

- candG starts from main (GPU 0), candH from candA (GPU 1); the same pair file.
- beta 0.1, one epoch, per-device batch 1 pair, gradient accumulation 8, maximum length 2,560, seed 53, cosine schedule, 4-bit paged AdamW.
- Loss: sigmoid DPO plus 0.5 times a length-normalized NLL on the chosen text (`loss_type=["sigmoid", "sft"]`), which keeps the strict output format from drifting.
- Snapshots at half the steps and at the end: four selectable checkpoints (candG-half, candG, candH-half, candH).
- Learning rate 2e-5.

Development selection is the section-27 rule, applied to the checkpoints above on `r4dev` plus HotpotQA dev (480 episodes), after the priority-0c scorer fixes, with FB applied to every system on both sides: strictly fewer unsupported answers (pooled and per family); HotpotQA-dev unsupported not above main's; supported-correct not below main's; invalid not above main's plus 1 point; HotpotQA success not below main's minus 3 points; macro at least main's plus 3 points. Ties go to fewer unsupported answers, then the main-initialised candidate.

## Deviations from section 27

| item | section 27 | now | reason |
| --- | --- | --- | --- |
| rollouts per episode and system | k = 4 | k = 2 | measured throughput of about 4 s per rollout per T4: 1,920 rollouts already cost about 1.2 h; more questions beat more samples per question |
| pairs | 1,500 to 3,000 | about 450 to 500 (final count set by the 6-point balance rule) | measured SFT throughput on a T4 was 330 processed tokens per second; a pair is two sequences of about 2,000 tokens plus a reference pass, about 14 s each, so 450 pairs cost about 1.75 h per GPU. Section 27's budget of 4 h assumed a cost per pair that was too low |
| learning rate | 5e-6 | 2e-5 | 450 pairs and 8-pair batches give about 56 optimizer steps; 5e-6 on a rank-16 LoRA would barely move the adapter. The final-decision preference sits at one token (`answer` against `abst`), so a small number of steps is enough |
| loss | DPO | DPO plus 0.5 chosen-NLL | format stability (invalid turns are a gate) |
| checkpoints | final | half and final | cheap guard against overshooting into over-abstention; four candidates, the same count as round 4 |
| near-miss negatives (answer sentence deleted) | in the first pass | deferred to an optional second sampling pass | needs a doctored corpus and index; the round-4 hard-negative variants already put on-topic passages without the asked fact in front of the model |
| MiniCheck filter on chosen answers | optional | not used | references are gold answers with real support ids |

## Priority 0b, 0c and 1: fixed before the checker was run (7 November 2026, 01:10 IST)

### 0b. Deterministic checks on the development traces (computed already, labels from scorer v2)

`scripts/build_verifier_cases.py` on main's answered episodes in `r4dev` and HotpotQA dev: 183 answers (76 supported correct, 87 wrong, 20 unsupported on hidden evidence).

| deterministic check | supported correct blocked | wrong answers blocked | unsupported blocked |
| --- | ---: | ---: | ---: |
| cited chunk was never shown | 0 / 76 | 0 / 87 | 0 / 20 |
| some answer part (split at ";") absent from the cited passages | 1 / 76 | 10 / 87 | 3 / 20 |

The deterministic checks alone cannot reach the 40% gate (15% of unsupported answers). Priority 1 therefore rests on the model-based check.

### 0c. Scorer version 2 (`src/ragharness/rescore.py`, `scripts/rescore_round5.py`)

Version 1 reproduces the stored scores exactly in all 12 development runs (0 mismatches). Version 2 adds the token-precision rule (at least 0.5 against the closest accepted gold form) and excludes a hidden-evidence episode only when the shown passages hold the whole answer, meaning every part of a compound answer. A first draft excluded an episode when any single part was visible; it removed 27 of 132 hidden episodes, almost all of them the designed "partial" family where one half is deliberately missing, so it was corrected before any result was used. With the corrected rule one hidden episode per system is excluded on `r4dev` (`5709b503200fba1400368289:support_masked`, where another passage holds the answer; `r4_missing_0` too for candA) and none on HotpotQA dev. `reports/round5_rescore_r4.{md,json}` list both versions for every development and final-test run of round 4. The precision rule lowers HotpotQA supported success for every system (main 0.314 to 0.256).

### 1. The checker (Tiny Aya itself, no generation)

- Cases: the 183 answered episodes above, with the cited passages rendered exactly as the tool rendered them (`data/processed/verifier/cases_dev.jsonl`, SHA256 `bc38c8ee2c0a6c596ff4bca09778b364068bb5f5b9697932f0f585f23ca5b47f`).
- Configurations: checker = base Tiny Aya Global or main's adapter; wording A (reply SUPPORTED or NOT) or wording B (reply Yes or No). Four configurations, one forward pass per case, score = probability of the positive first token against the negative one. An answer is blocked when that probability is below 0.5. No threshold is tuned.
- Gate on development data, unchanged from section 27: block at least 40% of main's 20 unsupported answers and at most 10% of its 76 supported-correct answers. V is the deterministic checks first, then the model check. Among configurations that pass, the one with the largest (unsupported blocked minus supported-correct blocked) wins. If none passes, V is dropped.
- Also reported, not gated: the share of main's 87 wrong answers blocked, and the effect of V on macro success and false refusals with FB applied.
- Cost: 366 forward passes per configuration of about 770 tokens, a few minutes on one T4. The run uses account 2 (`vibhutimeghwal`) alongside the sampling job.

## Result of priority 1 (7 November 2026, 01:20 IST)

Kaggle kernel `vibhutimeghwal/ats-verify-r5-dev` (about 4 minutes of GPU time per configuration). Full table: `reports/round5_verifier.md`.

| configuration | unsupported blocked (of 20) | supported correct blocked (of 76) | wrong answers blocked (of 87) | passes the gate |
| --- | ---: | ---: | ---: | --- |
| base, wording A | 60% | 24% | 53% | no |
| base, wording B | 75% | 21% | 62% | no |
| main adapter, wording A | 70% | 8% | 47% | **yes** |
| main adapter, wording B | 100% | 87% | 99% | no (blocks nearly everything) |

**V = the deterministic checks, then main's adapter with wording A, is adopted** and joins every round-5 system. On main's development traces (scorer v2, FB applied on both sides) it changes unsupported answers from 18 to 5 of 131 hidden-evidence episodes, supported-correct from 0.243 to 0.228, macro from 0.515 to 0.538, and false refusals from 0.471 to 0.635. The false-refusal rise is mostly wrong answers that become abstentions, which were failures already. The sample is small (20 unsupported answers), so the 70% is a development-set estimate only; the sealed final test decides.

## Priority 0d: sealed final test (7 November 2026, 01:30 IST, before any training)

`reports/round5_test_manifest.json` holds the file hashes. Three parts, all disjoint from every earlier set (asserted at build time), the scripted oracle scoring 1.0 on each:

| part | questions | episodes | hidden-evidence episodes |
| --- | ---: | ---: | ---: |
| `r5t_squad` (25 unused SQuAD training-split articles, round-4 variants) | 65 | 385 | 158 |
| `r5t_hotpot` (unused HotpotQA validation questions) | 60 (44 reachable) | 82 | 38 |
| `r5t_rgb` (RGB questions no earlier set used, noise 0, 0.8 and 1) | 100 | 471 | 179 |

Combined hash `cb368b0d1f917461015d446dee5aa054a012e9ef6e2623898043dd08c5cbba23`, 938 episodes (the plan said about 700; RGB's three noise settings and the round-4 variants made it larger). The set is opened once, at the final test.

## Selection with the checker, an out-of-sample check, and the final-test definition (7 November 2026, 01:40 IST)

- **Selection with V on both sides.** The training kernel also builds checker cases for the four candidates' development answers and scores them with main's adapter and wording A, exactly as for main. `scripts/select_round5.py` then applies FB and V to main and to every candidate before the six gates. Main with FB and V has 5 unsupported answers of 131 hidden-evidence episodes on `r4dev` and 1 on HotpotQA dev, so a candidate must beat those counts, not main's raw 18 and 2. This makes the first gate hard to pass; it is kept as written. If no candidate passes, main with FB and V is the round-5 system and the DPO runs are reported as a negative result.
- **Out-of-sample check of V.** V's configuration was chosen on development traces. The round-4 final-test traces of main, candA and candE1 were never used to choose it. `data/processed/verifier/cases_r4test.jsonl` holds their 664 answered episodes. The same configuration (main adapter, wording A, threshold 0.5) is run on them once, and the blocked shares of unsupported, supported-correct and wrong answers are reported next to the development numbers. Nothing is selected on this check; those traces were already opened in round 4.
- **Final-test macro.** On `r5test` the macro success is the mean of four component rates under scorer v2, each on its own episodes: SQuAD single questions, SQuAD compound questions, `r5t_hotpot`, `r5t_rgb`. Success means a correct answer that cites every required evidence aspect, or an abstention on a hidden-evidence episode. The unsupported rate is pooled over all hidden-evidence episodes (version-2 exclusions removed). Intervals resample whole questions within each part and question type, 3,000 draws, seed 47.
- **Systems opened on `r5test`, once:** base (reference), main, candA (reference), and the round-5 winner if one exists; FB and V are applied offline and with one checker pass over each system's answers. The recommendation rule is the one in section 27: a winner with FB and V replaces main with FB and V only if the macro difference has an interval above zero and the unsupported-rate difference is below zero as a point estimate with an upper bound of at most +0.02. The comparison of main with FB and V against raw main is reported as the effect of the adopted wrappers.

## Amendment to the pair rules, still before any sample was read (7 November 2026, 01:50 IST)

The per-question caps (2 per question and variant, 4 per question) bound the file at about 4 pairs for each of roughly 81 questions. If those caps leave fewer than 450 balanced pairs, the builder now applies the caps (3, 6) instead. This is a fixed fallback, applied mechanically (`caps_used` is written to `reports/round5_pairs.json`). A development look at main's saved traces explains why pro-answer pairs will be plentiful: on `r4dev`, with the full evidence shown, main abstains in 41 of 54 `open` episodes (and answers correctly and with support in 1), and across all variants it abstains in 70 of 165 complete-evidence episodes. Pairs that reject those abstentions are the accuracy side of the preference data; pairs that reject unsupported answers are the safety side.

## Result of the out-of-sample check of V, and a widened candidate pool (7 November 2026, 02:00 IST)

**V on the round-4 final-test traces** (`reports/round5_verifier_r4test.json`; kernel `ats-verify-main-r5-r4test`; same configuration as on development data, run once):

| system | unsupported blocked | supported correct blocked | wrong answers blocked |
| --- | ---: | ---: | ---: |
| main | 23 / 39 (59%) | 7 / 51 (14%) | 55 / 113 (49%) |
| candA | 36 / 54 (67%) | 3 / 64 (5%) | 74 / 137 (54%) |
| candE1 | 21 / 32 (66%) | 8 / 85 (9%) | 43 / 86 (50%) |

The effect holds out of sample: about 60 to 67% of unsupported answers are blocked at a cost of 5 to 14% of supported-correct ones (the development gate was 40% and 10%; main sits just above 10% here). `reports/round5_r4test_wrappers.md` reports the wrappers on these already-opened traces: main + FB + V against raw main, macro +0.111 [+0.076, +0.147], unsupported rate -0.127 [-0.178, -0.076], supported correct -0.027 [-0.030, -0.004].

**Widened candidate pool, declared before any development verdict for these systems exists.** The same table shows candE1 with FB and V ahead of main with FB and V on all three measures (macro 0.555 against 0.461, unsupported 11 against 16, supported correct 0.294 against 0.168). That look was at the round-4 test traces, so it influenced this decision, and I am saying so. To keep the test out of the selection, the round-4 checkpoints (candE1, candE2, candF1, candF2) join the candidate pool for the development gates, which now has eight members: the four DPO checkpoints (candG1, candG2, candH1, candH2) and the four round-4 checkpoints. Their development traces already exist (`r4dev`, HotpotQA dev); only the checker has to be run on their answers (`data/processed/verifier/cases_dev_r4cands.jsonl`, 1,073 cases). The gates and the tie-break are the section-27 rules, unchanged, with FB and V on both sides and scorer v2. Selection uses development data only. The sealed `r5test` is opened once and decides by the section-27 recommendation rule, which is where the extra comparisons are paid for: a winner must show an interval above zero on fresh data. If a round-4 checkpoint wins, it is evaluated and, if it passes, released as a new adapter; main stays the default otherwise.

## Sampling result and the pair file (7 November 2026, 02:15 IST; written before training starts)

Kernel `vibhutimeghwal/ats-sample-r5-pairs`, 1,920 rollouts, about 63 minutes on each T4. Sampled outcomes (main, temperature 0.8): `r5train` 622 rollouts: 59 supported correct, 168 wrong answers, 117 false abstentions, 170 correct abstentions, 61 unsupported answers, 47 invalid; `hotpot_train` 338 rollouts: 128 supported correct, 78 wrong, 39 false abstentions, 65 correct abstentions, 21 unsupported, 7 invalid. candA's summary is in the same folder.

The builder applied the rules above to all 1,920 rollouts (`reports/round5_pairs.json`): 858 candidate pairs after the template, length and duplicate checks; the (2, 4) caps left too few, so the fallback (3, 6) was used (425 pairs); balancing gave **335 pairs** over 77 questions, SHA256 `8214a2c62bf7b0483fead2d32a6758c05849a1bd01cbef4fec7ba8f9c4681508`. Of these, 186 are pro-answer (chosen = the cited correct answer; rejected = 83 abstentions, 93 wrong answers, 10 invalid turns) and 149 are pro-caution (chosen = a further search in 147 cases and an abstention in 2; rejected = 119 premature answers, 20 invalid turns, 8 early abstentions, 1 answer at the third search plus 1 invalid). 224 pairs come from `r5train` and 111 from `hotpot_train`; 167 rejected texts were sampled from main and 168 from candA. Mean length 1,626 tokens, maximum 2,485.

One limitation is visible already. Only two chosen texts are abstentions: the model rarely answers wrongly at the third search on hidden evidence, so on-policy sampling produces almost no "answer at the last decision" errors to correct. The pairs therefore teach "answer when the evidence is shown" and "search again before answering when it is not", and say nothing new about the last-decision abstention. The development gates (unsupported answers per family, with V applied) are what would show an erosion of that behavior.

### Amendment to the training schedule, made before training starts (7 November 2026, 02:20 IST)

335 pairs with 8 pairs per optimizer step give 41 steps for one epoch, and the half-epoch snapshot would then sit after 20 steps. Training therefore runs **two epochs (82 steps)**, with the snapshots after epoch 1 and epoch 2 (candG1, candG2, candH1, candH2 — the same naming as round 4's candE1 and candE2). The learning rate stays 2e-5 (same order of total parameter movement as round 4's 3e-5 over 92 steps). The pair-fitting probe now divides the time budget by the number of epochs; the budget is 2.4 hours of training per GPU, which fits all 335 pairs if a pair takes at most about 25 seconds. If the timing probe says fewer pairs fit, the first N pairs of the shuffled, balanced file are used and `train_report.json` records N.
