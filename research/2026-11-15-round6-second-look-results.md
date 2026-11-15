# Round 6 sealed test: results of the second look checked by V

The pre-registration is `research/2026-11-15-round6-second-look-preregistration.md` (commit e667f3c, 15 November 2026, before any model output on this test). The run was launched once, at 09:39 IST, on Kaggle account 1 (kernel `ats-r6-system-sl`) and finished without a rerun. The registered report is `scripts/report_round6_second_look.py`, unchanged after the run. Outputs: `reports/round6_second_look.{json,md}`. Post hoc numbers (not part of the decision): `scripts/report_round6_second_look_extra.py`, `reports/round6_second_look_extra.{json,md}`.

Integrity: 679 episodes (75 + 60 + 29 questions), all eligible and shared by every system. No prompt was truncated (0 of 679 plus 0 of 304 forced prompts; the input limit was the default 4,096 tokens). Stored and replayed outcomes agree on every episode. The sealed inputs matched the hashes in `reports/round6_test_manifest.json` before the run.

## Decision

**Not adopted. The submitted system stays main + fail-closed + V.**

| Pre-registered condition | Result | Met |
| --- | --- | --- |
| Macro-success difference has a 95% interval above zero | +0.9 points, interval −0.3 to +2.1 | no |
| Hidden-evidence answer-rate difference has an upper bound of at most +0.02 | +0.7 points, interval +0.0 to +1.8 | yes |

The development estimate was +1.4 points of macro success. The test gave +0.9, and its interval includes zero. The development analysis found no hidden-evidence answer added at 0.90. The test added two.

## All systems (scorer v3, 679 shared episodes, 291 of them hide the evidence)

| System | Macro success | Correct and cited, of answerable | Answers on hidden evidence | Answered at all | Correct among answers shown |
| --- | ---: | ---: | ---: | ---: | ---: |
| main (raw) | 44.6% | 22.7% | 61 / 291 | 47.3% | 27.4% |
| main + FB | 50.2% | 22.7% | 61 / 291 | 47.3% | 27.4% |
| **main + FB + V (submitted)** | **54.1%** | 21.1% | 29 / 291 | 34.8% | 34.7% |
| main + FB + V + second look at 0.90 | 55.0% | 23.5% | 31 / 291 | 37.4% | 35.8% |
| refuse every question | 43.8% | 0.0% | 0 / 291 | 0.0% | none |

Paired differences (3,000 cluster draws, seed 47; SQuAD by article, HotpotQA and RGB by question):

| Comparison | Macro success | Correct and cited | Hidden-answer rate |
| --- | ---: | ---: | ---: |
| second look minus main + FB + V (the decision) | +0.9 pp [−0.3, +2.1] | +2.3 pp [+0.7, +4.1] | +0.7 pp [+0.0, +1.8] |
| main + FB + V minus main + FB | +3.9 pp [+1.9, +6.0] | −1.5 pp [−2.9, −0.5] | −11.0 pp [−15.2, −7.4] |
| main + FB + V minus main | +9.5 pp [+7.1, +11.9] | −1.5 pp [−2.9, −0.5] | −11.0 pp [−15.2, −7.4] |
| main + FB + V minus refuse everything (post hoc) | +10.3 pp [+7.5, +13.1] | +21.1 pp [+17.5, +25.0] | +10.0 pp [+6.3, +13.7] |

The second look raised correct and cited answers by 2.3 points with an interval above zero. That is a gain on one of four measures, and the registered rule was about macro success. Macro success did not clear zero because the gain is small: the second look added 9 correct and cited answers and 2 hidden-evidence answers, a net 7 successful episodes out of 679, before weighting by component. The 7 wrong answers it showed do not lower macro success (each replaces a refusal that already counted as a failure on an answerable episode), but they are visible mistakes, and they pull the share of correct answers among those shown down.

## What the second look did

Main refused 304 times after it had seen passages (128 answerable episodes, 176 with hidden evidence). All 304 forced continuations parsed as an answer with citations. 238 of them passed the deterministic checks, and 111 of those scored at least 0.50 with V.

| V at least | Correct and cited added | Wrong answers shown | Hidden-evidence episodes turned into answers (of 176) |
| ---: | ---: | ---: | ---: |
| 0.60 | 27 | 36 | 25 |
| 0.70 | 22 | 30 | 13 |
| 0.80 | 16 | 19 | 6 |
| 0.85 | 13 | 13 | 5 |
| **0.90 (registered)** | **9** | **7** | **2** |
| 0.95 | 4 | 3 | 1 |

At 0.90: SQuAD 12 shown (6 correct, 5 wrong, 1 hidden-evidence), HotpotQA 4 (2, 1, 1), RGB 2 (1, 1, 0). The two hidden-evidence answers: a HotpotQA `support_masked` episode (V gave 0.98; the answer matched the gold answer without the support, so it is flagged as unsupported but correct) and a SQuAD `r4_missing_0` episode (V gave 0.93; wrong answer). On development data the same threshold had added nine correct answers, four wrong ones and no hidden-evidence answer; the test added nine, seven and two. The count of correct answers was the same. The count of mistakes was higher, which is what the development estimate could not see.

## The submitted system, measured a second time

This is the second fresh measurement of main + FB + V (the first is the Round 5 test). The two tests are drawn differently, so the pooled columns are not comparable. Round 6 has 448 SQuAD episodes out of 679 (66%), where Round 5 has 471 RGB episodes out of 934 and RGB is the easiest part. The macro components are comparable:

| Success per component | SQuAD single | SQuAD two-part | HotpotQA | RGB |
| --- | ---: | ---: | ---: | ---: |
| main + FB + V, Round 6 test | 43.3% | 36.2% | 58.9% | 78.0% |
| main + FB + V, Round 5 test | 59.1% | 40.4% | 58.5% | 74.1% |
| refuse every question, Round 6 test | 41.1% | 44.3% | 50.0% | 39.7% |
| refuse every question, Round 5 test | 40.0% | 43.5% | 46.3% | 38.0% |

What it shows:

- Refusing carries over. The hidden-evidence answer rate is 10.0% (29 of 291), against 11.2% (42 of 375) on the Round 5 test. HotpotQA success is the same (58.9% against 58.5%) and RGB success is higher (78.0% against 74.1%).
- Macro success is 54.1%, 10.3 points above refusing everything (interval +7.5 to +13.1). On the Round 5 test the margin was 16.0 points.
- SQuAD single-passage success fell from 59.1% to 43.3% and is now close to refusing everything (41.1%). This part has 75 questions from 25 articles, so its interval is wide, and we have not looked for the cause. SQuAD two-part success is below refusing everything on both tests (36.2% and 40.4% against 44.3% and 43.5%). SQuAD single-passage and two-part questions were already listed as weak spots.
- Of the answers it shows, 34.7% are correct on this test. The pooled figure was 67.3% on the Round 5 test. The two pooled figures are not comparable, because the mix of SQuAD, HotpotQA and RGB episodes differs, and we did not compute the split by part. The SQuAD single-passage drop above may account for part of the difference.
- We did not run the base model or the hosted models on this test.

## What stays the same, and what this changes in the claims

- The submitted system does not change.
- The Round 6 sealed test is now opened and used. Further model selection needs another fresh test.
- The README, the Article and the cards now report this test next to the Round 5 test. The claim that two thirds of the shown answers are correct stays tied to the Round 5 test; on the Round 6 test the share is 34.7%.
- The refusal behaviour, the main result of the project, replicated: 10.0% of hidden-evidence episodes were answered, against 75% (Round 5) for the base model, which was not run here.

## Disclosures

- The 0.90 threshold was chosen after the development table (see the pre-registration). It was the only value adopted from the test, and nothing was tuned on the test.
- The registered report and its tests were committed before the run and not changed afterwards. The post hoc script was written after the run and is labelled as such.
