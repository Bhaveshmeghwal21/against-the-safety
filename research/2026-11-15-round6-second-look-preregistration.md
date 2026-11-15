# Round 6 sealed test: a second look checked by V

Written on 15 November 2026, about 09:40 IST (the commit time is the record), before any model has produced output on the Round 6 sealed test (`r6t_squad`, `r6t_hotpot`, `r6t_rgb`). The repository remains private at the team's direction. This note continues `research/2026-11-14-round6-preregistration.md`. It uses the sealed test that note built and hashed, and it lists every departure from that note.

## Question

Does adding a second look to the submitted system (main + fail-closed + V) raise macro success on fresh questions without raising the rate of answers on hidden evidence?

The second look works as follows. When the system refuses after it has already seen passages (the model abstained, or its turn was invalid), it is asked once more, from the same prompt, to answer anyway (`<answer>` is forced, greedy, 160 new tokens). That forced answer must pass the deterministic checks (it cites only passages that were shown, and each part of the answer appears in a cited passage). V, the plain self-check (main adapter, wording A), then scores it. The answer is shown only if V's probability of SUPPORTED is at least 0.90. Otherwise the refusal stands. An answer that the model gave in its first attempt is not touched by the second look.

Second-look threshold: 0.90

## What I knew when I wrote this

- **Round 6 never tried this with V.** Round 6 built the second look and gated it on its trained checkers (V2m, V2b). Both failed the checker gate on development data, so the second look stayed off (Amendment 3 of the Round 6 note). V was never the checker of a second look in any registered gate.
- **Development data, offline, no GPU.** On 15 November I scored the saved Round 6 development outputs (`runs/kaggle/ats-r6-system-dev/out`, r4dev and HotpotQA dev only) with V as the second-look checker. Main + FB + V is the baseline. Macro success and correct and cited are the numbers of `.cache/scratch/second_look_macro.py` (correct and cited on the r4dev part; the macro is identical when pooled by `report_round6_second_look.py`). The hidden-evidence answers are those of the 134 hidden-evidence r4dev episodes.

  | Second look at | Macro success | Correct and cited | Hidden-evidence answers (of 134) |
  | --- | ---: | ---: | ---: |
  | none (submitted) | 0.540 | 0.228 | 5 |
  | 0.90 | 0.554 | 0.270 | 5 |
  | 0.85 | 0.563 | 0.296 | 6 |
  | 0.80 | 0.568 | 0.323 | 8 |
  | 0.70 | 0.569 | 0.344 | 8 |

  What the forced answers themselves look like (main's 112 answerable and 198 hidden-evidence refusals with shown passages, r4dev and HotpotQA dev together):

  | V at least | Correct and cited added | Wrong answers shown | Hidden-evidence refusals turned into answers (of 198) |
  | ---: | ---: | ---: | ---: |
  | 0.70 | 25 | 14 | 6 |
  | 0.80 | 21 | 7 | 3 |
  | 0.85 | 15 | 5 | 1 |
  | 0.90 | 9 | 4 | 0 |
  | 0.95 | 3 | 3 | 0 |
- **The threshold was chosen after seeing those tables.** The Round 6 recovery rule picks the smallest threshold at which at most 5% of hidden-evidence refusals turn into answers and at least three correct answers are added per hidden one. Pooled over main and candF1 (the Round 6 rule), V does not pass it: its best row is 0.70 (53 correct added against 19 hidden, which needs 57). Over main alone, the same rule would choose 0.65 (27 correct added against 9 hidden). I chose 0.90 instead, because it is the lowest threshold in the grid that adds no hidden-evidence answer on development data (0.95 also adds none, but only 3 correct answers). This is a post hoc choice, made on development data, with one value. It is a stricter threshold than either version of the Round 6 rule would pick. No other threshold will be adopted from the test, and none will be tuned on it.
- **The submitted system has already been tested on the opened Round 5 test** (`reports/reference_policies.json`): main + FB + V macro 0.580, correct and cited 0.463, 42 of 375 hidden-evidence answers. candF1 + FB + V was also evaluated there and did not pass its safety rule (hidden-answer rate rise at most +0.02). The Round 5 test is spent. The second look was not run on it.
- **No model output exists on the Round 6 sealed test.** The only files in `runs/` that touch `r6t_*` are scripted-oracle replays (`runs/oracle_r6`), which use no model. The hosted comparison models were never run on it. The sealed inputs were hashed on 14 November (`reports/round6_test_manifest.json`, SHA256 `b578e6f10f00cf7bba79bf1ba8ee163b0bbf9a83031c6ef7784a22ed0de0e475`) and all 14 file hashes in it matched when I checked them at about 09:25 IST on 15 November, before this note.
- **The multilingual XQuAD results do not inform this choice.** They were produced with `--no-second-look`.

## System and run

- One generator: the `main` adapter (`runs/kaggle/ats-train-both/out/main/adapter`, the one in the published `main` repository), on the base Tiny Aya Global 3.35B. V uses the same adapter. Nothing is trained.
- Jobs: `main` on `bench_r6t_squad`, `bench_r6t_hotpot` and `bench_r6t_rgb`; the three-search budget, top-three passages, 800-character display, lenient parsing, `--extra-questions 0`, 160 new tokens and the default 4,096-token input limit, as in the Round 6 development run. The English path is unchanged from that run.
- Stage: `r6-system` with `--checkers V` and the second look on. The stage reads the bundle's three sealed folders. The full bundle (not the development-only one) carries them.
- Hardware: Kaggle account 1 (`bhaveshmeghwal`), two T4s, timeout 4 hours. About 2 GPU-hours are expected (679 episodes).
- The sealed test is opened **once**. A rerun is allowed only if the kernel crashes for an infrastructure reason, with the same code, the same bundle id and the same arguments. I will not read partial outputs to decide anything. The decision uses the one complete run.

## Systems compared

All come from the one run of the main adapter: `main` (raw), `main+FB` (an invalid turn ends as an abstention), `main+FB+V` (the submitted system; V accepts at 0.50), `main+FB+V+second look` (the candidate), and an always-refuse baseline. Scorer v3. Macro success is the mean of the success rates of four components: SQuAD single, SQuAD two-part, HotpotQA and RGB.

## Decision rule (registered before the run)

The candidate replaces main + FB + V as the submitted system only if both hold:

1. The macro-success difference (candidate minus main + FB + V) has a 95% interval whose lower end is above zero.
2. The difference in the rate of answers on hidden-evidence episodes (pooled over all hidden-evidence episodes) has a 95% interval whose upper end is at most +0.02.

Intervals: 3,000 paired cluster-bootstrap draws, seed 47, the same draws for both systems. SQuAD is clustered by article, HotpotQA and RGB by question.

This is the rule that the Round 6 note set for a new system, with one change. That note also asks that the hidden-answer-rate difference be below zero as a point estimate. A second look can only turn refusals into answers, so the rate cannot fall, and the clause cannot be met by construction. I replace it with the upper bound of at most +0.02, which is the user-approved rule ("the hidden-evidence answer rate may rise by at most 2 points").

If both conditions hold, the candidate becomes the submitted system. I then set the live system's `recover_threshold` default to 0.90, update the README, the Article, the abstract, HISTORY and the Hugging Face cards, and say the test was run once. If either fails, nothing about the submitted system changes. The result is reported as a failed pre-registered test, and the test's `main+FB+V` row is reported as a second fresh measurement of the submitted system.

## What is reported either way

Every system's macro success, correct and cited among answerable episodes, hidden-evidence answers, answers shown at all, and correct among answers shown. For the second look: the table of what it adds at each threshold from 0.60 to 0.95 (descriptive; only 0.90 counts for the decision), how many forced answers were invalid, and how many wrong answers it showed. Paired intervals for the other comparisons are descriptive. I also report the count of truncated prompts and any replay discrepancy; a run with truncated prompts or replay mismatches is reported as such and not hidden.

## Departures from the Round 6 note

- Only the main adapter runs. Base, candF1 and the hosted models are not run on this test.
- V is the only checker. V2m and V2b failed development selection and play no part.
- The recovery threshold is 0.90, chosen as described above, not by the gate.
- A new report script, `scripts/report_round6_second_look.py`, replaces `scripts/report_round6.py`, which refuses to run without an adopted trained checker. The new script checks that this note is committed at HEAD before it reads any sealed output. I tested it on a synthetic run built from the scripted-oracle replays and on the development outputs (it reproduces macro 0.540 and 0.554). I did not change it after the run.
- The decision rule drops the point-estimate clause, as explained above.
