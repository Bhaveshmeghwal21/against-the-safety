# CandF1 multilingual XQuAD comparison — verifier incomplete

Exploratory follow-up on already opened XQuAD questions; no new model was trained.
Candidate verifier incomplete; no candidate +V score is reported.
3,000 paired article-cluster draws, seed 47.

## Hindi

| System | Success | Correct and cited | Hidden answers | False refusals | Answer coverage | Correct among answered | Invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 4.0% | 0.0% | 188/285 | 7.7% | 64.9% | 0.0% | 27.2% |
| base+FB | 17.0% | 0.0% | 188/285 | 36.1% | 64.9% | 0.0% | 0.0% |
| main | 13.5% | 2.5% | 6/285 | 21.8% | 3.9% | 31.8% | 73.0% |
| main+FB | 50.2% | 2.5% | 6/285 | 94.4% | 3.9% | 31.8% | 0.0% |
| main+FB+V | 50.5% | 2.1% | 3/285 | 96.1% | 2.5% | 42.9% | 0.0% |
| candF1 | 14.6% | 7.7% | 8/285 | 18.9% | 8.6% | 44.9% | 71.2% |
| candF1+FB | 52.5% | 7.7% | 8/285 | 85.6% | 8.6% | 44.9% | 0.0% |
| always refuse | 50.0% | 0.0% | 0/285 | 100.0% | 0.0% | — | 0.0% |

candF1+FB minus main+FB, on 285 paired questions:
- correct_and_cited: +5.3 pp (95% interval +2.1 to +8.7).
- hidden_answer_rate: +0.7 pp (95% interval -2.1 to +3.5).
- success: +2.3 pp (95% interval +0.5 to +4.2).

## English

| System | Success | Correct and cited | Hidden answers | False refusals | Answer coverage | Correct among answered | Invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 2.4% | 0.0% | 205/295 | 0.0% | 73.6% | 0.0% | 24.1% |
| base+FB | 15.3% | 0.0% | 205/295 | 22.4% | 73.6% | 0.0% | 0.0% |
| main | 38.8% | 2.7% | 0/295 | 75.3% | 1.4% | 100.0% | 23.6% |
| main+FB | 51.4% | 2.7% | 0/295 | 97.3% | 1.4% | 100.0% | 0.0% |
| main+FB+V | 51.4% | 2.7% | 0/295 | 97.3% | 1.4% | 100.0% | 0.0% |
| candF1 | 43.1% | 7.1% | 3/295 | 71.2% | 4.7% | 75.0% | 20.2% |
| candF1+FB | 53.1% | 7.1% | 3/295 | 91.5% | 4.7% | 75.0% | 0.0% |
| always refuse | 50.0% | 0.0% | 0/295 | 100.0% | 0.0% | — | 0.0% |

candF1+FB minus main+FB, on 295 paired questions:
- correct_and_cited: +4.4 pp (95% interval +1.7 to +7.1).
- hidden_answer_rate: +1.0 pp (95% interval +0.0 to +2.3).
- success: +1.7 pp (95% interval +0.2 to +3.3).

## Spanish

| System | Success | Correct and cited | Hidden answers | False refusals | Answer coverage | Correct among answered | Invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 1.2% | 0.0% | 237/285 | 1.1% | 83.9% | 0.0% | 14.4% |
| base+FB | 8.4% | 0.0% | 237/285 | 15.4% | 83.9% | 0.0% | 0.0% |
| main | 48.6% | 4.9% | 2/285 | 87.7% | 3.5% | 70.0% | 6.5% |
| main+FB | 52.1% | 4.9% | 2/285 | 93.7% | 3.5% | 70.0% | 0.0% |
| main+FB+V | 52.1% | 4.6% | 1/285 | 94.4% | 3.0% | 76.5% | 0.0% |
| candF1 | 35.1% | 7.0% | 1/285 | 65.6% | 4.7% | 74.1% | 30.9% |
| candF1+FB | 53.3% | 7.0% | 1/285 | 90.9% | 4.7% | 74.1% | 0.0% |
| always refuse | 50.0% | 0.0% | 0/285 | 100.0% | 0.0% | — | 0.0% |

candF1+FB minus main+FB, on 285 paired questions:
- correct_and_cited: +2.1 pp (95% interval -1.1 to +5.5).
- hidden_answer_rate: -0.4 pp (95% interval -1.4 to +0.7).
- success: +1.2 pp (95% interval -0.5 to +3.0).

Hidden-evidence outcomes are annotation-based flags, not human-adjudicated hallucinations.
The XQuAD results were inspected before this follow-up; do not use this as a fresh blind selection test.
