# Accuracy and coverage (test, headline episodes: 151 = 86 answerable + 65 with the evidence masked)

**13 November 2026 correction:** This table preserves the original scorer and selection record. Evidence alternatives, binary answers, fixed eligibility and source offsets are corrected in the [dated remediation report](2026-11-13-evaluation-remediation.md). Later DPO prompts exposed held-out HotpotQA evidence. Annotation-based unsupported flags do not establish semantic hallucination; main remains submitted.


Each episode lands in exactly one row. Counts, then per 100 episodes.

| outcome | base | main | control |
| --- | ---: | ---: | ---: |
| supported answer (good) | 1 (0.7) | 38 (25.2) | 61 (40.4) |
| bad answer: wrong, unsupported, or answered with the evidence missing | 125 (82.8) | 24 (15.9) | 24 (15.9) |
| correct abstain (good) | 2 (1.3) | 50 (33.1) | 0 (0.0) |
| false abstain | 0 (0.0) | 25 (16.6) | 0 (0.0) |
| no valid final turn | 23 (15.2) | 14 (9.3) | 66 (43.7) |
| answers delivered | 126 (83.4) | 62 (41.1) | 85 (56.3) |
| precision of delivered answers (supported / delivered) | 0.008 | 0.613 | 0.718 |
| evidence missing: no answer delivered (explicit abstain, or the episode ended without a valid final turn) | 14/65 (21.5%) | 61/65 (93.8%) | 58/65 (89.2%) |
| evidence missing: explicit abstain | 2/65 (3.1%) | 50/65 (76.9%) | 0/65 (0.0%) |

## By evidence state (all evaluated episodes)

Each cell: supported answer / bad answer / correct abstain / false abstain / no valid final turn, out of n.

| evidence state | answerable | n | base | main | control |
| --- | --- | ---: | --- | --- | --- |
| first_search_masked | yes | 29 | 0 / 25 / 0 / 0 / 4 | 12 / 14 / 0 / 1 / 2 | 17 / 8 / 0 / 0 / 4 |
| open | yes | 86 | 1 / 74 / 0 / 0 / 11 | 38 / 20 / 0 / 25 / 3 | 61 / 17 / 0 / 0 / 8 |
| prefill_clean | yes | 30 | 0 / 18 / 0 / 0 / 12 | 19 / 4 / 0 / 7 / 0 | 24 / 6 / 0 / 0 / 0 |
| prefill_partial | yes | 30 | 4 / 23 / 0 / 0 / 3 | 19 / 9 / 0 / 2 / 0 | 18 / 12 / 0 / 0 / 0 |
| prefill_partial_masked | no | 17 | 0 / 13 / 0 / 0 / 4 | 0 / 4 / 12 / 0 / 1 | 0 / 15 / 0 / 0 / 2 |
| prefill_sufficient | yes | 30 | 2 / 20 / 0 / 0 / 8 | 20 / 3 / 0 / 4 / 3 | 25 / 5 / 0 / 0 / 0 |
| support_masked | no | 65 | 0 / 51 / 2 / 0 / 12 | 0 / 4 / 50 / 0 / 11 | 0 / 7 / 0 / 0 / 58 |
