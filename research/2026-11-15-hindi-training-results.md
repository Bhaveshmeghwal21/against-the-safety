# Hindi training of main and candF1: results

Registered in `research/2026-11-15-hindi-training-preregistration.md` (commit cd74be7, amendment 1 in commit 3f32d0f). Report script `scripts/report_hindi.py` (commit a5a256c, before any score was read). Saved outputs: `reports/hindi_results.json` and `reports/hindi_results.md`. 3,000 paired cluster draws, seed 47, scorer v3.

## What was run

| step | where | result |
| --- | --- | --- |
| Train `hiMain` (from `main`) and `hiF` (from `candF1`) | Kaggle account 1, T4 x 2, `ats-hi-train-a`, 11:25 IST to about 15:10 IST (3.3 h of training, then the `hi_dev` runs) | 956 and 899 rows, 120 and 114 optimizer steps, final loss 0.036 and 0.042, peak 9.98 GB, nothing truncated |
| Baselines `main` and `candF1` on `xq2_hi`, `xq2_en`, `hi_dev`, and `candF1` on `r6t_*` | Kaggle account 2, `ats-r6-system-hi-base`, then `ats-verify-cases-hi-base` | generation finished; V ran out of memory and was rerun on the 561 saved cases (amendment 1) |
| `hiMain` and `hiF` on `xq2_hi`, `xq2_en`, `r6t_*`, with V | Kaggle account 1, `ats-r6-system-hi-new` | complete; 1,497 of 1,501 answer cases scored, the other 4 had no passages and are withheld by the registered `valid()` rule |

The 4 x L4 machine in the request cannot be selected from the Kaggle command line (it offers T4, P100, TPU; a request for it ran on T4 x 2), so everything ran on T4 x 2.

## Decision (registered rules, each adapter against its parent, both with fail-closed and V)

| Registered rule (difference in points, 95% interval) | hiMain minus main | hiF minus candF1 |
| --- | ---: | ---: |
| A1. Hindi, correct and cited (lower end above 0) | +33.3 [+27.9, +38.9]: met | +31.3 [+25.5, +37.3]: met |
| A2. Hindi, answers on hidden evidence (upper end at most +2.0) | +4.8 [+2.1, +7.8]: not met | +3.4 [+0.7, +6.6]: not met |
| A3a. English, macro success (lower end at least -3.0) | +1.0 [-2.0, +4.1]: met | +1.2 [-1.5, +4.0]: met |
| A3b. English, answers on hidden evidence (upper end at most +2.0) | +8.2 [+4.1, +12.2]: not met | +1.7 [-0.7, +4.6]: not met |

**Neither adapter is published as a Hindi-capable variant under the registered rule.** Both fail A2, and both fail A3 on its hidden-evidence part. The rule says the adapters are still published, labelled by what failed. The English submission (`main` + fail-closed + self-check) is unchanged.

## Numbers

Fresh Hindi test (300 XQuAD questions never used before, 291 with both variants after exclusions), every system with both repairs:

| System | Hindi: correct and cited (291 questions) | Hindi: answers when evidence was hidden | English: macro success (second sealed test) | English: answers when evidence was hidden |
| --- | ---: | ---: | ---: | ---: |
| Main + both repairs (submitted) | 2.1% | 4 / 291 | 54.1% | 29 / 291 |
| hiMain + both repairs | 35.4% | 18 / 291 | 55.1% | 53 / 291 |
| candF1 + both repairs | 3.8% | 3 / 291 | 62.6% | 28 / 291 |
| hiF + both repairs | 35.1% | 13 / 291 | 63.9% | 33 / 291 |
| Refuse every question | 0.0% | 0 / 291 | 43.8% | 0 / 291 |

Other measures (all in `reports/hindi_results.md`):
- Hindi success (a correct, cited answer, or a refusal when the evidence is hidden; refusing everything scores 50.0%): `main` 50.3%, `hiMain` 64.6% (+14.3 points, interval +11.2 to +17.3), `candF1` 51.4%, `hiF` 65.3% (+13.9, +10.8 to +17.2).
- Refusals of answerable Hindi questions: `main` 96.6%, `hiMain` 47.4%, `candF1` 91.4%, `hiF` 48.1%.
- Invalid raw turns in Hindi: `main` 75.6%, `candF1` 72.3%, `hiMain` 0.2%, `hiF` 0.9%. The repeated-search failure found in the first check is gone.
- The same 300 questions in English (`xq2_en`), correct and cited with both repairs: `main` 1.4%, `hiMain` 12.7%, `candF1` 10.3%, `hiF` 24.1%. Answers on hidden evidence there: 1, 0, 1 and 3 of 291.
- Answers on hidden evidence before V on Hindi: `main` 5, `hiMain` 36, `candF1` 12, `hiF` 27 of 291. V removes about half of them, which is why the repaired numbers above are lower.
- `main` on `r6t` reproduces the Round 6 figures (54.1% macro, 21.1% correct and cited, 29 of 291 hidden answers), as it should: that run is reused.

## Reading

- **Training worked on what it targeted.** Correct, cited Hindi answers rose from 2% and 4% to 35% in both adapters, and the broken search loop disappeared. The gain does not come from Hindi alone: the same questions in English also rose (1.4% to 12.7% for `hiMain`, 10.3% to 24.1% for `hiF`), so part of it is better single-passage answering, the weak spot we had already recorded. Hindi stays well above English for `hiMain` (35.4% against 12.7%).
- **It cost caution.** Both adapters answer on hidden evidence more often. In Hindi the repaired rate went from 1.4% to 6.2% (`hiMain`) and from 1.0% to 4.5% (`hiF`). On the English second sealed test `hiMain` answered 53 hidden-evidence episodes against 29 for `main`; `hiF` answered 33 against 28 for `candF1`, within noise of the bound but not inside it (upper end +4.6 points against +2.0 allowed). We did not test why. One untested possibility is that the English replay (half the Hindi token count) was too small to hold the English refusal habit.
- **`hiF` is the closer of the two** on the English safety rule, and `hiMain` is far from it. Neither is a drop-in replacement for the English system.
- **Macro success on English did not drop** (lower ends -2.0 and -1.5 points against a -3.0 limit), so the English cost is in caution and not in accuracy.

## What could be wrong

- One training run per adapter, one seed. No second attempt was made or is registered; a new attempt (more English replay, a lower learning rate) would need its own registration and a test set that has not been opened. `xq2_hi` and `r6t` are now both used.
- The training questions are machine translated (IndicSQuAD Hindi); the test questions are human translated (XQuAD). Transfer from one to the other is what A1 measures. `hi_dev`, which is machine translated, shows a much higher raw rate of answers on hidden evidence (48 of 101 for `hiMain`, 47 for `hiF`, against 17 for both parents) than `xq2_hi` does (36 and 27 of 291). The hidden-evidence flags are annotation based, and after machine translation the hidden passage may still hold the answer, so `hi_dev` is a sanity check and not evidence about safety.
- V verdicts for `main` on `r6t` come from the Round 6 test run (batch 6, full-sequence logits, generator input limit 4,096; no prompt was truncated). All other V verdicts in this experiment use batch 1 and last-position logits (amendment 1). A probability within rounding of 0.50 could flip between the two; I count that as noise.
- The test questions share articles and passages with the first 300 XQuAD questions, whose results had been seen. Hidden-evidence answers are annotation-based flags.
- The intervals resample questions (by article for XQuAD) and do not cover training variance.
