# Multilingual check: results (Hindi, English, Spanish)

Run: one Kaggle kernel (`ats-r6-system-xq`, account 1, two T4s), 15 November 2026, 02:23 to 05:40 IST. Registered in `research/2026-11-15-multilingual-plan.md` (commit 792e5fc, amendments in section 14, commit 046eccd). Scored by `scripts/report_xquad.py`; machine-readable numbers in `reports/xquad_results.json`, tables in `reports/xquad_results.md`, figure `docs/figures/xquad_results.svg`.

No rerun, no infrastructure failure, no episode dropped. All six runs completed, including Spanish. Replaying every transcript reproduced the scores stored by the run (0 differences in all six runs), and no prompt was truncated (limit 8,000 tokens, see amendment 1).

## What was tested

The same 300 XQuAD questions in each language, over a search index of that language's passages (328 Hindi, 339 English, 377 Spanish). The oracle reached the answer passage for 285 Hindi, 295 English and 285 Spanish questions; those are the questions that count. Each has two episodes: evidence available, and evidence hidden. Systems: Tiny Aya base, the main adapter, and main + FB + V (the submitted system). Totals: 570 Hindi, 590 English and 570 Spanish episodes per system.

## Results

Success counts a correct, cited answer when the evidence is available and a refusal when it is hidden. Refusing every question scores exactly 50.0% in every language, because half the episodes are hidden-evidence ones.

| language | system | success | correct and cited (evidence available) | answers although evidence hidden | refuses although evidence available | invalid turns |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Hindi | base | 4.0% | 0.0% | 188/285 (66.0%) | 7.7% | 27.2% |
| Hindi | main | 13.5% | 2.5% | 6/285 (2.1%) | 21.8% | 73.0% |
| Hindi | main + FB + V | 50.5% | 2.1% | 3/285 (1.1%) | 96.1% | 0 |
| English | base | 2.4% | 0.0% | 205/295 (69.5%) | 0.0% | 24.1% |
| English | main | 38.8% | 2.7% | 0/295 (0.0%) | 75.3% | 23.6% |
| English | main + FB + V | 51.4% | 2.7% | 0/295 (0.0%) | 97.3% | 0 |
| Spanish | base | 1.2% | 0.0% | 237/285 (83.2%) | 1.1% | 14.4% |
| Spanish | main | 48.6% | 4.9% | 2/285 (0.7%) | 87.7% | 6.5% |
| Spanish | main + FB + V | 52.1% | 4.6% | 1/285 (0.4%) | 94.4% | 0 |

Paired differences on the questions both sides have, with 95% intervals over the 48 articles (3,000 draws, seed 47):

| comparison | correct and cited | answers although evidence hidden |
| --- | ---: | ---: |
| main + FB + V, Hindi minus English (282 questions) | -0.7 pp [-3.2, +2.1] | +1.1 pp [0.0, +2.2] |
| main + FB + V, Spanish minus English (285) | +1.8 pp [-0.4, +3.8] | +0.4 pp [0.0, +1.1] |
| Hindi, main + FB + V minus base | +2.1 pp [+0.7, +3.9] | -64.9 pp [-72.1, -58.3] |
| Spanish, main + FB + V minus base | +4.6 pp [+2.4, +6.9] | -82.8 pp [-87.5, -77.4] |
| English, main + FB + V minus base | +2.7 pp [+1.0, +4.4] | -69.5 pp [-75.2, -64.0] |

## What it shows

1. **The refusal behaviour carries over. The answering does not, in any language.** With the evidence hidden, the submitted system answers 1.1% of Hindi, 0.0% of English and 0.4% of Spanish episodes, against 66% to 83% for the base model. But with the evidence available it gives a correct, cited answer in only 2.1% (Hindi), 2.7% (English) and 4.6% (Spanish). Its success rate is 50.5%, 51.4% and 52.1%, which is what refusing every question scores (50.0%). On this test the system behaves almost like a refuse-everything system.
2. **This is mostly a weak spot we already knew, not a language effect.** XQuAD asks single-passage extractive questions. The English control already gives 2.7%, in line with the 5.5% the raw adapter scored on SQuAD 2.0 earlier, and the README already lists single-passage questions as a weak spot. Hindi and Spanish are not measurably worse than English on correct and cited answers (the intervals include zero).
3. **Hindi breaks the search loop much more often.** The adapter's raw invalid-turn rate is 73.0% in Hindi, 23.6% in English and 6.5% in Spanish. Almost all of it is one error: after the first search the model sends the same query again, which the harness counts as invalid (403 of 570 Hindi episodes, 138 of 590 English, 37 of 570 Spanish). Search itself works: the evidence-set recall of the raw adapter is 0.99 in Hindi and 1.00 in English and Spanish, and AuditKIT's loop-free share is 0.26 in Hindi, 0.76 in English and 0.93 in Spanish. We did not test why it loops in Hindi; the Hindi passages cost about 3.4 times as many tokens (median 432 against 128), which makes the prompts far longer than anything the adapter was trained on, but that is a guess.
4. **The base model answers in the right script but never cites correctly.** 92% of its Hindi answers contain Devanagari letters; the 8% without are mostly numbers and names, and 15% of the gold answers contain none. It scored 0.0% correct and cited in all three languages because it does not produce valid citations, as in earlier rounds. In Hindi it also fails the tag format often: 27% of its turns are invalid, nearly all untagged answers that run to the 160-token limit.
5. **The wrong-language check is clean for the submitted system.** Of the 14 Hindi answers it showed, 4 held no Devanagari letter (29%), against 15% for the gold answers; with 14 answers that proportion is too small to read.

## Limits of this check

- One sample of 300 questions per language from one benchmark, with Spanish and Hindi translations of the English text. Translated questions are not the same as questions written by native speakers.
- Only the oracle-reachable questions count (95.0% to 98.3% of the 300).
- The system was not changed or tuned for this check. It was designed and trained on English HotpotQA, and the check cannot say how an adapter trained in Hindi would do.
- FB and V are applied offline to the saved transcripts, as in rounds 5 and 6. V used the English checker prompt unchanged.
- The hosted-model comparison in section 11 of the plan was not run: it needs the Azure chat endpoint URL, which is not stored on this machine, and Hindi prompts cost about three times as many tokens as English ones.

## Deviations from the registered plan

All of these were decided before the run and are labelled in section 14 of the plan: the input limit of 8,000 tokens (instead of 4,096) for all three languages and both systems; an empty language tag as the default so the old English path is untouched; the exact normalization (MLQA rule, NFC, Devanagari digits read as ASCII); passage counts that differ from the plan's expectation. One further point after the run: the plan's timeline had a fallback to drop Spanish; it was not needed.
