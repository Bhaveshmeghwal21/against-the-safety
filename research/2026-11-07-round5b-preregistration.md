# Round 5b: DPO with abstention-chosen pairs, a feasible development gate, one opening of the sealed test

Written 7 November 2026, 11:05 IST, before any forced answer exists and before any round-5b model is trained. This note continues `research/2026-11-07-round5-preregistration.md` and `research/2026-11-07-round5-results.md`. Anything that departs from section 27 of `research/2026-11-04-fine-tuning-for-rag.md` or from the round-5 note is listed here, with the reason.

## Why a second DPO run

Round 5 DPO raised answer accuracy on SQuAD-style questions (candG2 supported correct 0.381 against main's 0.228 with the repairs on) and removed the abstain action: none of the four checkpoints ended a development episode in an abstention, while main ends 88 of 131 hidden-evidence `r4dev` episodes and 70 of 73 hidden-evidence HotpotQA episodes that way. The pair file rejected an abstention 91 times and chose one twice. The remedy registered here changes only the data: add pairs that choose the abstention at hidden-evidence decision points, whose rejected side is an answer the model itself gives when forced to answer there.

## What I knew when I wrote this (disclosure)

1. The sealed test `r5test` (938 episodes) has been opened for base, main and candA, with the repairs. Their results are in `reports/round5_final.md`. No other system has been run on it.
2. The development table for all eight earlier candidates is known (`reports/round5_selection.md`). Before writing the gate below I ran it on those eight checkpoints (output in `.cache/`, not committed). **candF1, a round-4 checkpoint, passes every condition of the new gate** and would be selected. The gate is therefore informed by the development table: it is a margin I judged reasonable (three points of unsupported rate is about four episodes of 131), not a prospective test of candF1. candF1 stays in the pool because it is eligible by a rule written before the new checkpoints exist, and because the sealed test, which nobody has used to choose anything, still decides.
3. The old gate 1 (strictly fewer unsupported answers than main with both repairs) can only be passed by chance with 131 hidden-evidence episodes (main has 5). The new gate replaces it with a margin, and adds what the first DPO run lacked: a check that the model itself still abstains.

## Pairs

Round-5 pair file, unchanged: 335 pairs, SHA256 `8214a2c62bf7b0483fead2d32a6758c05849a1bd01cbef4fec7ba8f9c4681508`.

New abstention-chosen pairs, from the same 1,920 sampled round-5 training rollouts (no development or test episode is read). `scripts/build_pairs_round5b.py`, seed 53:

1. **Contexts.** A sampled rollout qualifies when its episode has hidden evidence, no shown passage holds the whole answer (the training-side false-unsupported fix of round 5), the model itself abstained at its final step, and it had made at least one search. The context is the message list before that abstention. Identical contexts are kept once. At most 2 per (question, variant) and 4 per question, at most 120 per source set, in a seeded shuffle that ignores outcome and length. Prompt plus 160 new tokens must fit 2,560 tokens.
2. **Chosen** is the canonical abstention text of the first-round training set (`episodes.ABSTAIN_TEXT` formatted with the question), the same text the round-5 pairs use.
3. **Rejected** is a forced answer. The system that produced the context (main or candA) sees exactly those messages, its reply starts with the tokens of `<answer>`, and it continues by sampling (temperature 0.7, top-p 0.95, at most 160 new tokens, one sample per context; `scripts/force_answers_round5.py`, seeds 5303 and 5304). The text must parse as a complete answer with at least one citation, must differ from the chosen text, and must fit the length limit. At most 90 new pairs per source set.
4. The new pairs are appended to the unchanged round-5 pairs and the whole file is shuffled with seed 53 (`data/processed/training/dpo_r5b.jsonl`). Its SHA256 is recorded in the next amendment, before training starts.

Context selection ran on CPU before this note was written (the numbers do not use any model output): 362 qualifying contexts, 195 after the caps (r5train 120, hotpot_train 75; main 86, candA 109; depth 1: 11, depth 2: 125, depth 3: 59; 71 questions). The context file is `data/processed/training/force_contexts_r5.jsonl`, SHA256 to be recorded with the forced answers. About 165 new pairs are expected, which would raise the HotpotQA share of the pair file from 33% to about 36% and bring chosen abstentions to about a third of the total.

## Training

Same recipe as round 5, changed only in the data and the second starting point: sigmoid DPO with beta 0.1 plus 0.5 times the chosen-text NLL, reference = the starting adapter, AdamW 2e-5, cosine schedule with 5% warm-up, 8 pairs per step, two epochs, seed 53, snapshots after epoch 1 and epoch 2 (`scripts/train_dpo.py`, unchanged).

- **candI** starts from main (GPU 0): candI1, candI2.
- **candK** starts from candF1 (GPU 1): candK1, candK2. Round 5 started its second run from candA. candF1 replaces candA because it beat candA on every development measure that matters here (macro 0.583 against 0.544, supported correct 0.312 against 0.249, unsupported answers 7 against 8 of 131 with the repairs). This choice is also informed by the development table.

Estimated cost: forced answers about 0.3 hour, training about 2.5 hours per GPU for about 500 pairs at 9 seconds per pair pass, development evaluation and the checker about 2.2 hours, final test about 1 hour. All on Kaggle account 2 (`vibhutimeghwal`).

## Development gate (replaces the six round-5 gates)

Pool: the twelve checkpoints candE1, candE2, candF1, candF2, candG1, candG2, candH1, candH2, candI1, candI2, candK1, candK2. Data: `r4dev` plus HotpotQA dev, scorer version 2, the fail-closed rule and the checker V applied to every system (reference main). A checkpoint is eligible only if all seven hold (`scripts/select_round5b.py`):

1. **Abstention retained.** The share of hidden-evidence episodes that end in the model's own abstention (before the repairs) is at least 0.75 of main's (main: 158 of 204, 0.775).
2. **Unsupported rate within 3 points.** Unsupported answers per hidden-evidence episode, with the repairs, at most main's rate plus 0.03 (main: 5 of 131).
3. **HotpotQA unsupported within 2 episodes** of main's (main: 1).
4. **Supported correct at least main's plus 0.05** (main: 0.228).
5. **Invalid turns** before the repair at most main's plus 1 point.
6. **HotpotQA success** at least main's minus 3 points.
7. **Macro** at least main's plus 3 points (main: 0.538).

The highest macro among the eligible wins. Macro ties within 0.005 go to fewer unsupported answers, then to the name. If none is eligible, main with both repairs stays and no model is opened on the sealed test.

## Sealed test: one opening

Only the winner is run on `r5test`, with the same repairs and the same checker as main (main adapter, wording A, threshold 0.5), alongside nothing else new. The recommendation rule is the one of section 27, unchanged: the winner with the repairs replaces main with the repairs only if the macro difference has a 95% interval above zero, and the unsupported-rate difference is below zero as a point estimate with an upper bound of at most +0.02 (intervals from resampling whole questions within each part, 3,000 draws, seed 47, appended after the existing comparisons). Otherwise main with both repairs stays the headline and the result is reported as it falls, with the candidate as a documented alternative.

Because a candidate that is a little more accurate but not safer than main cannot pass this rule (candA with the repairs scored +0.024 macro and +0.022 unsupported on the sealed test and would not have passed), a pass means strictly that the candidate keeps unsupported answers at or below main's while improving the macro score.

## Dates and limits

Latest start of the sealed-test run: 10 November. Hard freeze for training: 11 November, 23:59 IST. Account 2's remaining quota is unknown to me (about 14 hours of its probably 30-hour weekly allowance are used); no step starts without a reason to believe it can finish, and a failed launch is reported.

Every step is committed before it runs. The next commits are the forced-answer results with the pair-file hash (before training), the development selection (before the sealed test is touched), and the final report.
