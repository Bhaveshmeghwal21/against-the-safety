# Accuracy and coverage (dev, headline episodes: 159 = 86 answerable + 73 with the evidence masked)

Each episode lands in exactly one row. Counts, then per 100 episodes.

| outcome | b-3 | b-2 | b-1 | b0 | b+1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| supported answer (good) | 37 (23.3) | 37 (23.3) | 37 (23.3) | 37 (23.3) | 37 (23.3) |
| bad answer: wrong, unsupported, or answered with the evidence missing | 31 (19.5) | 30 (18.9) | 28 (17.6) | 28 (17.6) | 26 (16.4) |
| correct abstain (good) | 67 (42.1) | 68 (42.8) | 70 (44.0) | 70 (44.0) | 70 (44.0) |
| false abstain | 20 (12.6) | 20 (12.6) | 20 (12.6) | 20 (12.6) | 22 (13.8) |
| no valid final turn | 4 (2.5) | 4 (2.5) | 4 (2.5) | 4 (2.5) | 4 (2.5) |
| answers delivered | 68 (42.8) | 67 (42.1) | 65 (40.9) | 65 (40.9) | 63 (39.6) |
| precision of delivered answers (supported / delivered) | 0.544 | 0.552 | 0.569 | 0.569 | 0.587 |
| evidence missing: no answer delivered (explicit abstain, or the episode ended without a valid final turn) | 68/73 (93.2%) | 69/73 (94.5%) | 71/73 (97.3%) | 71/73 (97.3%) | 71/73 (97.3%) |
| evidence missing: explicit abstain | 67/73 (91.8%) | 68/73 (93.2%) | 70/73 (95.9%) | 70/73 (95.9%) | 70/73 (95.9%) |

## By evidence state (all evaluated episodes)

Each cell: supported answer / bad answer / correct abstain / false abstain / no valid final turn, out of n.

| evidence state | answerable | n | b-3 | b-2 | b-1 | b0 | b+1 |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| first_search_masked | yes | 29 | 8 / 18 / 0 / 3 / 0 | 8 / 18 / 0 / 3 / 0 | 8 / 18 / 0 / 3 / 0 | 8 / 18 / 0 / 3 / 0 | 8 / 18 / 0 / 3 / 0 |
| open | yes | 86 | 37 / 26 / 0 / 20 / 3 | 37 / 26 / 0 / 20 / 3 | 37 / 26 / 0 / 20 / 3 | 37 / 26 / 0 / 20 / 3 | 37 / 24 / 0 / 22 / 3 |
| prefill_clean | yes | 30 | 19 / 7 / 0 / 3 / 1 | 19 / 7 / 0 / 3 / 1 | 19 / 7 / 0 / 3 / 1 | 19 / 7 / 0 / 3 / 1 | 19 / 7 / 0 / 3 / 1 |
| prefill_partial | yes | 30 | 16 / 10 / 0 / 4 / 0 | 16 / 10 / 0 / 4 / 0 | 15 / 10 / 0 / 5 / 0 | 14 / 10 / 0 / 6 / 0 | 14 / 9 / 0 / 7 / 0 |
| prefill_partial_masked | no | 15 | 0 / 8 / 7 / 0 / 0 | 0 / 8 / 7 / 0 / 0 | 0 / 8 / 7 / 0 / 0 | 0 / 8 / 7 / 0 / 0 | 0 / 8 / 7 / 0 / 0 |
| prefill_sufficient | yes | 30 | 18 / 8 / 0 / 4 / 0 | 18 / 8 / 0 / 4 / 0 | 18 / 8 / 0 / 4 / 0 | 18 / 8 / 0 / 4 / 0 | 18 / 8 / 0 / 4 / 0 |
| support_masked | no | 73 | 0 / 5 / 67 / 0 / 1 | 0 / 4 / 68 / 0 / 1 | 0 / 2 / 70 / 0 / 1 | 0 / 2 / 70 / 0 / 1 | 0 / 2 / 70 / 0 / 1 |
