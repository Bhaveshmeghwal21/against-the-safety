# Round 4 results: evidence-grounded accuracy

Experiment started 6 November 2026; all GPU stages completed by 00:04 IST on 7 November. Team: Against-the-safety. Track 3, PS 4: fine-tuning for RAG.

**Decision: retain main.** The friend's Kaggle account successfully ran both T4 GPUs and trained four usable checkpoints. None met the predefined accuracy and unsupported-answer requirements. The strongest development variant improved the new controlled exercises but failed the existing HotpotQA checks. No zero-unsupported-answer model was produced.

This report records the completed experiment rather than a competition leaderboard result. All four adapters, raw traces, scores and data provenance are saved locally. The original published model was not replaced.

## What was trained

- Base: Cohere Tiny Aya Global, 3.35B, revision `af89d219b53ed9b13b8a4645f9c8028973510324`.
- E continues the original main LoRA; F continues round-2 candA. The suffix 1 or 2 identifies the saved epoch.
- Both runs used identical 721-row data: 152 matched evidence-present/evidence-missing pairs plus 417 replay decisions. New and replay data each contribute about half the processed-token budget.
- The pairs keep the same question, passage count and search depth while changing whether a required fact is available. They teach answer versus search before the search budget is exhausted, and answer versus abstain at the last decision. Actual tool calls, neutral IDs and matched retrieval novelty are retained.
- New SQuAD training-split articles were divided by article into training, development and final-test groups, excluding round-3 training/development articles. This prevents overlap with the new curriculum; it does not establish absence from foundation-model pretraining.
- QLoRA NF4, rank 16, alpha 32, dropout 0.05, learning rate `3e-5`, seed 47, maximum training length 2,560, gradient accumulation 16, two epochs. Each run completed 92 optimizer steps.
- AgentTune supplies the actual training factory and event/tool traces. AuditKIT evaluates the same episodes. Existing tool and decoding budgets stay fixed.

Training data SHA256: `64edf966c462f4b2ad1f01657334d20215c17b7d241fb92bd8f34122c24bd83d`.

The [prospective plan and execution log](2026-11-06-round4-evidence-training.md) records the thresholds, data provenance and timing of pre-evaluation clarifications.

## Development results

Every system saw the same 321 new-suite episodes and 159 original HotpotQA episodes. New-suite macro success gives equal weight to single and compound questions. Success requires the existing answer-correctness rule and every required evidence aspect cited, or an abstention on a labelled unanswerable case. The HotpotQA column retains the existing legacy criterion.

These are automatic episode scores, not a human-judged factual-accuracy rate. See the concrete scoring limitations below.

| system | new-suite macro success | supported answers / 189 answerable | unsupported / 132 missing-evidence | HotpotQA success | HotpotQA unsupported / 73 |
| --- | ---: | ---: | ---: | ---: | ---: |
| main | 39.2% | 46 | 18 | 67.3% | 2 |
| candA | 43.5% | 53 | 20 | 62.9% | 9 |
| E1 | 39.0% | 60 | 17 | 62.3% | 5 |
| E2 | 43.1% | 70 | 21 | 62.9% | 8 |
| F1 | **52.7%** | 74 | **13** | 60.4% | 9 |
| F2 | 49.6% | **76** | 22 | 62.3% | 15 |

F1 has the highest three-component development macro, 55.23% against main's 48.53%. That selection statistic includes new single, new compound and original HotpotQA components; it is different from the new-suite-only macro in the table.

F1 still fails multiple requirements:

1. Its unsupported rate is 13/132 = 9.85%, above the fixed 5% ceiling. Fully missing-evidence failures are 6 against main's 5, although partial-evidence failures improve from 13 to 7.
2. HotpotQA success falls by 6.92 percentage points against main, exceeding the allowed three-point decline.
3. HotpotQA unsupported answers are 9 against main's 2.

No other checkpoint qualifies either. The [selection JSON](../reports/round4_selection.json) preserves every gate, including false outputs and invalid turns; the [selection table](../reports/round4_selection.md) preserves all four candidates.

The parent comparison matters. F1 starts from candA, whose new-suite macro is 43.49% and unsupported count is 20. F1 improves these to 52.66% and 13. CandA already has 9 HotpotQA unsupported answers, and F1 also has 9. Thus the 2-to-9 release-baseline comparison is not evidence that the new training itself introduced all seven extra failures. It failed to recover main's calibration while preserving its utility.

An exploratory question-grouped bootstrap gives F1 minus main a new-suite macro difference of +13.51 percentage points, 95% interval [+4.29, +22.73], and an unsupported-rate difference of -3.79 points, interval [-10.37, +2.99]. The latter does not establish a population-level reduction. These development intervals are unadjusted for choosing the best of four checkpoints and are not independent confirmation. Full values and procedure: [development intervals](../reports/round4_dev_intervals.json).

## What the traces show

The development review packet is `runs/round4_review_dev_F1/index.json`. It contains 61 episodes selected before reading their answer text: all 13 candidate answers on missing-evidence cases, all 21 answerable citation/coverage failures, ten scored successes per question type, and five paired wins and losses, with overlapping categories. This is a targeted audit with repeated question variants, not a random estimate of human accuracy.

Observed improvements include answering two independently required facts with both source passages cited, where main refused. For example, F1 correctly combines New Haven's household count with the Knights of Columbus headquarters fact when both supports are present.

Remaining failures are concrete:

- When the household evidence is hidden, F1 substitutes **129,779 people** for the requested **47,094 households**, while correctly answering the other half. A plausible number from a related paragraph is not the required fact.
- On Kerry's 1991--1993 committee, a missing-evidence answer instead selects the Foreign Relations Committee from a paragraph describing 2009.
- A correct Everton answer adds citation `c00014`, which was never shown. Correct answer text alone does not establish valid grounding.
- Several partial-evidence compound answers fill the missing half with a related gene, place or publication. Having one supported half is insufficient.
- E1 produces 77 repeated-query stops on the new suite, against main's 40. F1 produces 31. Repeated searches remain failures; they do not become successful abstentions. Parsing the actual training targets found zero repeated-next-query labels among all 211 selected search targets, so this is not a direct duplicate-query teacher-label bug. Its mechanism needs a separate ablation.

## Limits uncovered in the evaluation

All required evidence aspects being cited is necessary for this metric, but does not establish that every generated claim follows from the text.

- One scored-success answer to a **non-meat** agricultural-product question includes poultry and cattle as well as soybeans. The substring-based answer-part rule accepts the gold item without rejecting wrong additions.
- One source question says **George II**, while the supplied passage says **George III**. Matching its expected answer is rewarded despite that inconsistency.
- Another scored-success compound question says **NFL**, while its cited nine-preseason-game rule is explicitly about the **National Hockey League**. A matching date does not repair the false premise.
- A labelled missing-evidence Viacom/Mediaset case still contains other passages about Viacom's lawsuit. Its strict gold-paragraph-based unsupported label is debatable; Mediaset's action is still unestablished. Singular/plural answer matching introduces another limitation.
- The final trace review confirms an error in **our constructed masking**, separate from source-question mistakes: question `56e78b7a37bdd419002c40fd` is labelled missing evidence in `r4_missing_0` and `support_masked`, but shown passage `c00739` explicitly gives the National Archives Building's location. Both E1 answers are supported by that alternate passage despite receiving unsupported labels. For `572690b9f1498d1400e8e3f8`, `r4_missing_0`, shown passage `c00312` likewise explicitly places DDT in the organochlorine family; the model's plural answer also exposes the exact-token matching limitation. These are confirmed false-positive unsupported labels. Hiding the annotated gold paragraph did not hide all semantically equivalent evidence.
- Dated source passages, such as an article's use of “current manager,” define the document-relative task. They are not evidence about the current world.

The frozen counts were retained. We did not repair selected labels or change the scorer after seeing which model benefited. In particular, **33/187 is an automatic gold-mask protocol count, not 33 manually verified hallucinations**. These findings prevent interpreting either the automatic success or unsupported rate as semantic ground truth. Genuine wrong-entity/quantity answers and repeated-query failures are also present, so no reliable replacement has been established. Before another training run, revise and independently audit the constructed negative-evidence labels, accepted answers and claim-level checks equally for every model. Keep the original frozen results as an experiment record and use a newly sealed evaluation for the revised protocol.

## Runtime and saved artifacts

Both training runs took about 7,257 seconds concurrently: **2.02 hours of dual-T4 configuration time**, about 4.03 device-hours in aggregate. Peak training memory was 6.04 GiB per GPU. This is not a measurement of Kaggle's weekly quota deduction. Development evaluation took about another 2.06 configuration-hours, including environment setup.

All four adapters are saved locally under `runs/account2_round4/ats-train-evidence-r4/out/`. Each has 504 float32 tensors and 30,228,480 parameters, all finite; file hashes are distinct. The [checkpoint manifest](../reports/round4_checkpoints.json) records exact paths and SHA256 hashes. The private training notebook is [ats-train-evidence-r4](https://www.kaggle.com/code/vibhutimeghwal/ats-train-evidence-r4).

The runtime audit found zero truncated model inputs and zero empty observations across 4,795 executed development searches. This rules against the earlier SQLite/FTS fault explaining these development results. Training-length checks reject truncation, and the focused data/selection/transfer tests passed.

## Frozen final diagnostic

Completed 6 November at approximately 23:55 IST. With no eligible development candidate, the predeclared fallback is E1. Base, main, candA and E1 each completed 449 episodes over 74 fresh article-held-out questions: 262 answerable and 187 labelled missing-evidence episodes. E1 cannot be promoted using this test. F1 was not evaluated on this final set; its development score is not a demonstrated final-test result. The conditional six-suite deployment comparison was skipped because development selection failed.

| system | macro evidence success | supported answers / 262 | unsupported / 187 | invalid / 449 |
| --- | ---: | ---: | ---: | ---: |
| original Tiny Aya Global, no LoRA | 1.6% | 0 | 112 | 145 |
| main | 35.7% | 53 | 40 | 50 |
| candA | 37.5% | 66 | 55 | 19 |
| E1, diagnostic | 37.7% | 88 | 33 | 96 |

E1 minus main is +2.02 percentage points of macro success, with a 95% interval of [-2.48, +6.35]. Its unsupported-rate difference is -3.74 points, interval [-8.11, +1.04]. Both intervals include no improvement. E1 answers more supported questions but also doubles the repeated-query failures: all 96 invalid E1 episodes end on repetition, against 50 for main. Its 33 unsupported answers remain far from the target.

The primary numerical recommendation remains **main**. The [final report](../reports/round4_test.md) and [raw summary](../reports/round4_test.json) preserve all metrics. Intervals use 3,000 paired resamples of whole question IDs, stratified by single/compound, seed 47. They account for repeated variants of a question; they are not article-clustered or estimates of competition-wide generalization.

All four final jobs have zero truncated model inputs and zero empty observations across 2,289 executed searches. The final diagnostic kernel took approximately 1.10 configuration-hours including setup.

The final review packet is `runs/round4_review_final_E1/index.json`, retaining 105 episodes with overlapping reasons. The review inspected all 33 hidden-evidence question/answer outputs, the cited passages of the 20 seeded scored-success examples, and selected full evidence traces. It confirmed the alternate-evidence masking errors above as well as genuine errors such as substituting a different year's statistic or answering only one half. This is a qualitative diagnostic; the remaining stored citation-failure traces are not a certification of semantic correctness.

## Selective Bright-Pro use

Its evidence-aspect coverage idea is already used as an additional compound-answer check. That addresses missing facts and incomplete citations; the review above shows why coverage must be paired with semantic accuracy.

The evaluation-only pilot compares main and diagnostic E1 on ten fixed robotics sample queries with the full 63,920-document robotics corpus, using official aspect labels and normalized weights. No Bright-Pro queries or answers enter training. The pinned dataset revision, selected IDs and corpus checksum are recorded in the [manifest](../reports/brightpro_robotics_manifest.json).

The pilot completed on 7 November at about 00:04 IST. It uses our lexical search, three-search budget and 800-character passage display.

| system | weighted aspect recall | cited aspect recall | invalid / 10 | input truncations |
| --- | ---: | ---: | ---: | ---: |
| main | 17.03% | 7.31% | 3 | 0 |
| E1, diagnostic | 7.03% | 2.31% | 5 | 0 |

The ten-point retrieval difference comes entirely from query ID 4; the other nine recall values are identical. The sample gives no evidence for adopting E1 or for a significant improvement in winning chances. F1 was not tested in this pilot.

These are **document-ID coverage** measures: retrieving a labelled supporting document counts toward an aspect even when its relevant fact lies beyond the 800-character display. Neither that score nor a citation to the document proves answer entailment. This is a custom ten-query transfer diagnostic, not reproduction of the official 175-query benchmark or an answer-quality evaluation. See the [per-query pilot report](../reports/round4_brightpro.md) and [machine-readable results](../reports/round4_brightpro.json). Primary reference: [Bright-Pro repository](https://github.com/yale-nlp/Bright-Pro).

## Next focused experiment

Keep main as the submission model. Before spending more GPU time, correct and independently audit the next training/development data for entity, date, quantity-type and extra-claim errors. Then test a small preference-based grounding objective that prefers a supported answer or abstention over a plausible but unsupported completion, while retaining query-policy replay and explicitly training recovery from stagnant search results. This is a proposed follow-up, not something demonstrated by this run. It needs a newly sealed holdout; the inspected development cases and final diagnostic cannot become a fresh test again.
