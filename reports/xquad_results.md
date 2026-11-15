# Multilingual check: Hindi, English and Spanish on XQuAD

3000 paired draws over Wikipedia articles, seed 47. Scorer v3 with the XQuAD normalization.

## Hindi

| system | episodes | success | correct and cited | answers on hidden evidence | refuses when evidence is there | correct among answers shown | invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 570 | 4.0% | 0.0% | 188/285 (66.0%) | 7.7% | 0.0% | 27.2% |
| base+FB | 570 | 17.0% | 0.0% | 188/285 (66.0%) | 36.1% | 0.0% | 0.0% |
| main | 570 | 13.5% | 2.5% | 6/285 (2.1%) | 21.8% | 31.8% | 73.0% |
| main+FB | 570 | 50.2% | 2.5% | 6/285 (2.1%) | 94.4% | 31.8% | 0.0% |
| main+FB+V | 570 | 50.5% | 2.1% | 3/285 (1.1%) | 96.1% | 42.9% | 0.0% |
| always refuse | 570 | 50.0% | 0.0% | 0/285 (0.0%) | 100.0% | - | 0.0% |

## English

| system | episodes | success | correct and cited | answers on hidden evidence | refuses when evidence is there | correct among answers shown | invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 590 | 2.4% | 0.0% | 205/295 (69.5%) | 0.0% | 0.0% | 24.1% |
| base+FB | 590 | 15.3% | 0.0% | 205/295 (69.5%) | 22.4% | 0.0% | 0.0% |
| main | 590 | 38.8% | 2.7% | 0/295 (0.0%) | 75.3% | 100.0% | 23.6% |
| main+FB | 590 | 51.4% | 2.7% | 0/295 (0.0%) | 97.3% | 100.0% | 0.0% |
| main+FB+V | 590 | 51.4% | 2.7% | 0/295 (0.0%) | 97.3% | 100.0% | 0.0% |
| always refuse | 590 | 50.0% | 0.0% | 0/295 (0.0%) | 100.0% | - | 0.0% |

## Spanish

| system | episodes | success | correct and cited | answers on hidden evidence | refuses when evidence is there | correct among answers shown | invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 570 | 1.2% | 0.0% | 237/285 (83.2%) | 1.1% | 0.0% | 14.4% |
| base+FB | 570 | 8.4% | 0.0% | 237/285 (83.2%) | 15.4% | 0.0% | 0.0% |
| main | 570 | 48.6% | 4.9% | 2/285 (0.7%) | 87.7% | 70.0% | 6.5% |
| main+FB | 570 | 52.1% | 4.9% | 2/285 (0.7%) | 93.7% | 70.0% | 0.0% |
| main+FB+V | 570 | 52.1% | 4.6% | 1/285 (0.4%) | 94.4% | 76.5% | 0.0% |
| always refuse | 570 | 50.0% | 0.0% | 0/285 (0.0%) | 100.0% | - | 0.0% |

## Paired comparisons (difference, 95% interval)

| comparison | questions | correct and cited | answers on hidden evidence | success |
| --- | ---: | ---: | ---: | ---: |
| main+FB+V: Hindi minus English | 282 | -0.7 pp [-3.2, +2.1] | +1.1 pp [+0.0, +2.2] | -0.9 pp [-2.2, +0.5] |
| main: Hindi minus English | 282 | -0.4 pp [-2.8, +2.2] | +2.1 pp [+0.7, +4.2] | -25.5 pp [-29.4, -21.8] |
| Hindi: main+FB+V minus base | 285 | +2.1 pp [+0.7, +3.9] | -64.9 pp [-72.1, -58.3] | +46.5 pp [+44.3, +48.6] |
| Hindi: main minus base | 285 | +2.5 pp [+1.0, +4.2] | -63.9 pp [-71.6, -56.4] | +9.5 pp [+6.3, +12.7] |
| English: main+FB+V minus base | 295 | +2.7 pp [+1.0, +4.4] | -69.5 pp [-75.2, -64.0] | +49.0 pp [+47.6, +50.3] |
| English: main minus base | 295 | +2.7 pp [+1.0, +4.4] | -69.5 pp [-75.2, -64.0] | +36.4 pp [+33.4, +39.5] |
| main+FB+V: Spanish minus English | 285 | +1.8 pp [-0.4, +3.8] | +0.4 pp [+0.0, +1.1] | +0.7 pp [-0.4, +1.7] |
| main: Spanish minus English | 285 | +2.1 pp [+0.0, +4.1] | +0.7 pp [+0.0, +1.8] | +10.0 pp [+6.4, +13.4] |
| Spanish: main+FB+V minus base | 285 | +4.6 pp [+2.4, +6.9] | -82.8 pp [-87.5, -77.4] | +50.9 pp [+49.3, +52.5] |
| Spanish: main minus base | 285 | +4.9 pp [+2.5, +7.3] | -82.5 pp [-87.1, -77.2] | +47.4 pp [+45.3, +49.5] |

Prompts truncated by the input limit: {'base_hi': 0, 'main_hi': 0, 'base_en': 0, 'main_en': 0, 'base_es': 0, 'main_es': 0}
Stored scores that differ on replay: {'base_hi': 0, 'main_hi': 0, 'base_en': 0, 'main_en': 0, 'base_es': 0, 'main_es': 0}

Hindi answers with no Devanagari letter (names and numbers legitimately have none):

- base: 30 of 370 answers shown (8.1%); gold answers: 43 of 285 (15.1%)
- main: 4 of 22 answers shown (18.2%); gold answers: 43 of 285 (15.1%)
- main+FB+V: 4 of 14 answers shown (28.6%); gold answers: 43 of 285 (15.1%)
