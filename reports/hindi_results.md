# Hindi training of main and candF1: results

3000 paired cluster draws, seed 47. Scorer v3 (XQuAD normalization for Hindi and English).

## Decisions

- hiMain: A1_hindi_gain met, A2_hindi_safety not met, A3_english_not_worse not met, published_as_hindi_variant not met
- hiF: A1_hindi_gain met, A2_hindi_safety not met, A3_english_not_worse not met, published_as_hindi_variant not met

## Fresh Hindi test (xq2_hi, 300 XQuAD questions)

| system | episodes | success | correct and cited | answers on hidden evidence | false refusals | invalid raw turns | correct among answers shown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| main | 582 | 13.1% | 2.1% | 5/291 (1.7%) | 17.9% | 75.6% | 30.0% |
| main+FB | 582 | 50.2% | 2.1% | 5/291 (1.7%) | 94.8% | 0.0% | 30.0% |
| main+FB+V | 582 | 50.3% | 2.1% | 4/291 (1.4%) | 96.6% | 0.0% | 42.9% |
| candF1 | 582 | 11.5% | 3.8% | 12/291 (4.1%) | 19.6% | 72.3% | 22.9% |
| candF1+FB | 582 | 49.8% | 3.8% | 12/291 (4.1%) | 87.6% | 0.0% | 22.9% |
| candF1+FB+V | 582 | 51.4% | 3.8% | 3/291 (1.0%) | 91.4% | 0.0% | 39.3% |
| hiMain | 582 | 63.2% | 39.2% | 36/291 (12.4%) | 30.9% | 0.2% | 48.1% |
| hiMain+FB | 582 | 63.4% | 39.2% | 36/291 (12.4%) | 30.9% | 0.0% | 48.1% |
| hiMain+FB+V | 582 | 64.6% | 35.4% | 18/291 (6.2%) | 47.4% | 0.0% | 60.2% |
| hiF | 582 | 63.7% | 37.8% | 27/291 (9.3%) | 32.3% | 0.9% | 49.5% |
| hiF+FB | 582 | 64.3% | 37.8% | 27/291 (9.3%) | 33.0% | 0.0% | 49.5% |
| hiF+FB+V | 582 | 65.3% | 35.1% | 13/291 (4.5%) | 48.1% | 0.0% | 62.2% |
| always refuse | 582 | 50.0% | 0.0% | 0/291 (0.0%) | 100.0% | 0.0% | - |

| comparison | questions | correct and cited | answers on hidden evidence | success |
| --- | ---: | ---: | ---: | ---: |
| hiMain+FB minus main+FB | 291 | +37.1 pp [+32.1, +42.3] | +10.7 pp [+6.4, +15.6] | +13.2 pp [+9.6, +16.6] |
| hiMain+FB+V minus main+FB+V | 291 | +33.3 pp [+27.9, +38.9] | +4.8 pp [+2.1, +7.8] | +14.3 pp [+11.2, +17.3] |
| hiF+FB minus candF1+FB | 291 | +34.0 pp [+28.6, +39.8] | +5.2 pp [+1.0, +9.4] | +14.4 pp [+11.3, +17.8] |
| hiF+FB+V minus candF1+FB+V | 291 | +31.3 pp [+25.5, +37.3] | +3.4 pp [+0.7, +6.6] | +13.9 pp [+10.8, +17.2] |

## Same questions in English (xq2_en)

| system | episodes | success | correct and cited | answers on hidden evidence | false refusals | invalid raw turns | correct among answers shown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| main | 582 | 39.9% | 1.7% | 1/291 (0.3%) | 80.4% | 19.4% | 62.5% |
| main+FB | 582 | 50.7% | 1.7% | 1/291 (0.3%) | 97.6% | 0.0% | 62.5% |
| main+FB+V | 582 | 50.5% | 1.4% | 1/291 (0.3%) | 98.3% | 0.0% | 66.7% |
| candF1 | 582 | 45.5% | 10.7% | 3/291 (1.0%) | 69.8% | 17.4% | 70.5% |
| candF1+FB | 582 | 54.8% | 10.7% | 3/291 (1.0%) | 85.9% | 0.0% | 70.5% |
| candF1+FB+V | 582 | 55.0% | 10.3% | 1/291 (0.3%) | 86.6% | 0.0% | 75.0% |
| hiMain | 582 | 55.3% | 13.1% | 1/291 (0.3%) | 80.8% | 1.4% | 69.1% |
| hiMain+FB | 582 | 56.4% | 13.1% | 1/291 (0.3%) | 81.4% | 0.0% | 69.1% |
| hiMain+FB+V | 582 | 56.4% | 12.7% | 0/291 (0.0%) | 83.2% | 0.0% | 75.5% |
| hiF | 582 | 59.6% | 24.4% | 4/291 (1.4%) | 64.9% | 3.6% | 74.0% |
| hiF+FB | 582 | 61.5% | 24.4% | 4/291 (1.4%) | 68.4% | 0.0% | 74.0% |
| hiF+FB+V | 582 | 61.5% | 24.1% | 3/291 (1.0%) | 69.8% | 0.0% | 76.9% |
| always refuse | 582 | 50.0% | 0.0% | 0/291 (0.0%) | 100.0% | 0.0% | - |

| comparison | questions | correct and cited | answers on hidden evidence | success |
| --- | ---: | ---: | ---: | ---: |
| hiMain+FB minus main+FB | 291 | +11.3 pp [+7.6, +15.1] | +0.0 pp [-1.0, +1.0] | +5.7 pp [+3.7, +7.8] |
| hiMain+FB+V minus main+FB+V | 291 | +11.3 pp [+7.9, +14.9] | -0.3 pp [-1.1, +0.0] | +5.8 pp [+4.1, +7.7] |
| hiF+FB minus candF1+FB | 291 | +13.7 pp [+9.3, +18.5] | +0.3 pp [-1.1, +1.8] | +6.7 pp [+4.4, +9.0] |
| hiF+FB+V minus candF1+FB+V | 291 | +13.7 pp [+9.3, +18.5] | +0.7 pp [-0.7, +2.4] | +6.5 pp [+4.3, +8.7] |

## English regression (Round 6 test, 679 shared eligible episodes)

| system | macro success | correct and cited | answers on hidden evidence |
| --- | ---: | ---: | ---: |
| main+FB+V | 54.1% | 21.1% | 29/291 |
| candF1+FB+V | 62.6% | 38.7% | 28/291 |
| hiMain+FB+V | 55.1% | 30.7% | 53/291 |
| hiF+FB+V | 63.9% | 45.6% | 33/291 |
| always refuse | 43.8% | 0.0% | 0/291 |

| comparison | macro success | correct and cited | hidden-answer rate |
| --- | ---: | ---: | ---: |
| hiMain+FB+V/main+FB+V | +1.0 pp [-2.0, +4.1] | +9.5 pp [+5.3, +14.0] | +8.2 pp [+4.1, +12.2] |
| hiF+FB+V/candF1+FB+V | +1.2 pp [-1.5, +4.0] | +7.0 pp [+2.4, +11.5] | +1.7 pp [-0.7, +4.6] |

## Development set (hi_dev, 120 machine-translated questions; sanity only, no rule depends on it)

| system | episodes | success | correct and cited | answers on hidden evidence | false refusals | invalid raw turns | correct among answers shown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| main | 202 | 11.9% | 5.0% | 17/101 (16.8%) | 14.9% | 65.8% | 14.3% |
| main+FB | 202 | 44.1% | 5.0% | 17/101 (16.8%) | 82.2% | 0.0% | 14.3% |
| candF1 | 202 | 10.4% | 7.9% | 17/101 (16.8%) | 12.9% | 67.8% | 20.5% |
| candF1+FB | 202 | 45.5% | 7.9% | 17/101 (16.8%) | 78.2% | 0.0% | 20.5% |
| hiMain | 202 | 41.6% | 31.7% | 48/101 (47.5%) | 16.8% | 1.0% | 24.4% |
| hiMain+FB | 202 | 42.1% | 31.7% | 48/101 (47.5%) | 17.8% | 0.0% | 24.4% |
| hiF | 202 | 42.6% | 32.7% | 47/101 (46.5%) | 14.9% | 1.5% | 25.2% |
| hiF+FB | 202 | 43.1% | 32.7% | 47/101 (46.5%) | 16.8% | 0.0% | 25.2% |

Prompts truncated by the input limit: none. Input limits used: [4096, 8000]; main's r6t run is the Round 6 test run. No prompt was truncated in any run, so the differing limits do not matter.
