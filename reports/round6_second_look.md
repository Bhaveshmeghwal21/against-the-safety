# Round 6 sealed test: the second look checked by V

3000 paired cluster draws, seed 47; SQuAD by article, HotpotQA and RGB by question. Shared eligible episodes: 679.

| System | Macro success | Correct and cited, of answerable | Answers on hidden evidence | Answered at all | Correct among answered |
| --- | ---: | ---: | ---: | ---: | ---: |
| main | 44.6% | 22.7% | 61/291 | 47.3% | 27.4% |
| main+FB | 50.2% | 22.7% | 61/291 | 47.3% | 27.4% |
| main+FB+V | 54.1% | 21.1% | 29/291 | 34.8% | 34.7% |
| main+FB+V+second look | 55.0% | 23.5% | 31/291 | 37.4% | 35.8% |
| always refuse | 43.8% | 0.0% | 0/291 | 0.0% | - |

## Pre-registered decision

- main+FB+V+second look versus main+FB+V: DO NOT ADOPT; macro difference +0.009 (95% interval -0.003 to +0.021); hidden-answer-rate difference +0.007 (95% interval +0.000 to +0.018).

## Other paired comparisons (descriptive)

| Comparison | Macro | Correct and cited | Hidden-answer rate |
| --- | ---: | ---: | ---: |
| main+FB+V+second look/main+FB+V | +0.9 pp [-0.3, +2.1] | +2.3 pp [+0.7, +4.1] | +0.7 pp [+0.0, +1.8] |
| main+FB+V/main+FB | +3.9 pp [+1.9, +6.0] | -1.5 pp [-2.9, -0.5] | -11.0 pp [-15.2, -7.4] |
| main+FB+V+second look/main+FB | +4.8 pp [+2.5, +7.1] | +0.8 pp [-1.4, +2.9] | -10.3 pp [-14.6, -6.5] |
| main+FB+V/main | +9.5 pp [+7.1, +11.9] | -1.5 pp [-2.9, -0.5] | -11.0 pp [-15.2, -7.4] |

## What the second look adds at each threshold (descriptive; only 0.90 was registered)

| Threshold | Correct and cited added | Wrong answers shown | Hidden-evidence answers added |
| ---: | ---: | ---: | ---: |
| 0.60 | 27 | 36 | 25/176 |
| 0.65 | 24 | 32 | 19/176 |
| 0.70 | 22 | 30 | 13/176 |
| 0.75 | 16 | 25 | 9/176 |
| 0.80 | 16 | 19 | 6/176 |
| 0.85 | 13 | 13 | 5/176 |
| 0.90 | 9 | 7 | 2/176 |
| 0.95 | 4 | 3 | 1/176 |

Integrity (truncated prompts and input limit per part): {'r6t_squad': {'truncated_inputs': 0, 'max_input_tokens': 4096}, 'r6t_hotpot': {'truncated_inputs': 0, 'max_input_tokens': 4096}, 'r6t_rgb': {'truncated_inputs': 0, 'max_input_tokens': 4096}}
