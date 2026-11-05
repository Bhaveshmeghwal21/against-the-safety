# Accuracy and coverage (test2, headline episodes: 143 = 83 answerable + 60 with the evidence masked)

Each episode lands in exactly one row. Counts, then per 100 episodes.

| outcome | base | main | control | candA |
| --- | ---: | ---: | ---: | ---: |
| supported answer (good) | 0 (0.0) | 31 (21.7) | 40 (28.0) | 38 (26.6) |
| bad answer: wrong, unsupported, or answered with the evidence missing | 0 (0.0) | 20 (14.0) | 35 (24.5) | 30 (21.0) |
| correct abstain (good) | 0 (0.0) | 55 (38.5) | 0 (0.0) | 56 (39.2) |
| false abstain | 0 (0.0) | 30 (21.0) | 0 (0.0) | 15 (10.5) |
| no valid final turn | 143 (100.0) | 7 (4.9) | 68 (47.6) | 4 (2.8) |
| answers delivered | 0 (0.0) | 51 (35.7) | 75 (52.4) | 68 (47.6) |
| precision of delivered answers (supported / delivered) | n/a | 0.608 | 0.533 | 0.559 |
| evidence missing: no answer delivered (explicit abstain, or the episode ended without a valid final turn) | 60/60 (100.0%) | 59/60 (98.3%) | 54/60 (90.0%) | 59/60 (98.3%) |
| evidence missing: explicit abstain | 0/60 (0.0%) | 55/60 (91.7%) | 0/60 (0.0%) | 56/60 (93.3%) |

## By evidence state (all evaluated episodes)

Each cell: supported answer / bad answer / correct abstain / false abstain / no valid final turn, out of n.

| evidence state | answerable | n | base | main | control | candA |
| --- | --- | ---: | --- | --- | --- | --- |
| first_search_masked | yes | 26 | 0 / 0 / 0 / 0 / 26 | 10 / 10 / 0 / 3 / 3 | 15 / 6 / 0 / 0 / 5 | 11 / 8 / 0 / 7 / 0 |
| open | yes | 83 | 0 / 0 / 0 / 0 / 83 | 31 / 19 / 0 / 30 / 3 | 40 / 29 / 0 / 0 / 14 | 38 / 29 / 0 / 15 / 1 |
| prefill_clean | yes | 30 | 0 / 0 / 0 / 0 / 30 | 19 / 8 / 0 / 3 / 0 | 23 / 7 / 0 / 0 / 0 | 22 / 6 / 0 / 2 / 0 |
| prefill_partial | yes | 30 | 0 / 3 / 0 / 0 / 27 | 15 / 11 / 0 / 4 / 0 | 18 / 11 / 0 / 0 / 1 | 17 / 11 / 0 / 2 / 0 |
| prefill_partial_masked | no | 14 | 0 / 1 / 0 / 0 / 13 | 0 / 2 / 12 / 0 / 0 | 0 / 8 / 0 / 0 / 6 | 0 / 5 / 9 / 0 / 0 |
| prefill_sufficient | yes | 30 | 0 / 2 / 0 / 0 / 28 | 17 / 9 / 0 / 4 / 0 | 23 / 7 / 0 / 0 / 0 | 17 / 10 / 0 / 3 / 0 |
| support_masked | no | 60 | 0 / 0 / 0 / 0 / 60 | 0 / 1 / 55 / 0 / 4 | 0 / 6 / 0 / 0 / 54 | 0 / 1 / 56 / 0 / 3 |
