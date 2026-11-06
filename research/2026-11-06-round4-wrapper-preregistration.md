# Round 4 addendum: two inference-time wrappers, pre-registered

Written 6 November 2026, 23:25 IST. The round-4 final-test kernel `ats-eval-jobs-r4test` was still running, and none of its traces had been downloaded or read. Code: `scripts/wrapper_round4.py`. It adds no training and no model calls. It rescores the main and candA traces that the final test is already producing.

## Why

Round 4 development selection found no eligible checkpoint (`reports/round4_selection.md`). A CPU-only look at traces that already existed suggested two cheap levers outside the weights:

- **Fail closed (FB).** In the round-3 fresh test, 59 of main's 739 episodes ended on an invalid or looping turn. Returning an abstention there instead raised main's macro from 0.489 to 0.543, with the same 66 of 326 unsupported answers and the same 179 of 413 supported answers. On test2 the rate went from 0.582 to 0.597.
- **Agreement gate (G).** If main and candA run independently and the answer is returned only when both agree, unsupported answers in the round-3 fresh test fell from 66 to 38 of 326 (0.202 to 0.117). Supported answers fell from 179 to 156 of 413, and macro was 0.534. On test2, unsupported answers went from 3 to 2 of 74 and macro from 0.582 to 0.553.
- On round-4 development data, several adapter pairs looked better still. They were picked from 30 pairs on that data, so they are not tested here.

These numbers come from data that informed earlier choices. They motivate the test; they are not evidence of it.

## Fixed rule

Systems, scored on `bench_r4test` with the round-4 metric (`report_round4.describe`: macro over single and compound, success needs every required evidence aspect cited):

- `FB(main)`: main, with invalid or looping endings replaced by an abstention. `FB(candA)` is reported for description only.
- `G(main, candA)`: return main's answer only if main and candA both answered and their normalised answers have token F1 of at least 0.5. Otherwise abstain. Normalisation removes the answer tag, chunk IDs, punctuation, case and the words a, an, the, and.

Claims, each with a 95% bootstrap interval grouped by question (`report_round4.paired_interval`, seed 47):

1. FB(main) raises the score: macro difference interval above zero, and no more unsupported answers than main.
2. G(main, candA) lowers the unsupported-answer rate: interval of the difference below zero.
3. G(main, candA) does not lower the score: macro difference interval not below zero. Supported success is reported with its interval either way.

No other pair, threshold or normalisation will be tried on this test. The result does not change the headline model automatically. Both are inference-time wrappers on the same Tiny Aya Global base, not new adapters, and must be described that way. G doubles inference cost.
