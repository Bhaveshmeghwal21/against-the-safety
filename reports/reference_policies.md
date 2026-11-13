# Reference policies

Each row uses its own available, scorer-v3 eligible episodes.

| System | Macro success | Correct and cited, of answerable | Answers given, of hidden-evidence | Answered at all | Correct among answered |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 3.1% | 5.7% | 281/375 | 77.6% | 4.4% |
| main | 47.9% | 48.8% | 115/375 | 56.3% | 51.9% |
| main+FB+V | 58.0% | 46.3% | 42/375 | 41.2% | 67.3% |
| candF1+FB+V | 64.0% | 53.3% | 40/375 | 41.8% | 76.4% |
| azure_native+FB | 60.4% | 55.3% | 91/374 | 68.0% | 48.7% |
| azure_deepseek_native+FB | 64.9% | 63.7% | 60/373 | 61.1% | 62.3% |
| always_refuse | 42.0% | 0.0% | 0/375 | 0.0% | — |

## Earlier benchmark suites

| Suite | System | Success | Correct and cited, of answerable | Answers given, of hidden-evidence | Answered at all | Correct among answered |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2wiki | main | 51.8% | 21.8% | 3/79 | 42.8% | 26.8% |
| 2wiki | always_refuse | 47.6% | 0.0% | 0/79 | 0.0% | — |
| musique | main | 50.0% | 18.4% | 1/79 | 21.7% | 44.4% |
| musique | always_refuse | 47.6% | 0.0% | 0/79 | 0.0% | — |
| squad2 | main | 48.0% | 6.8% | 4/128 | 5.9% | 41.7% |
| squad2 | always_refuse | 63.4% | 0.0% | 0/128 | 0.0% | — |
| rgb | main | 58.1% | 70.6% | 89/167 | 76.4% | 58.4% |
| rgb | always_refuse | 36.9% | 0.0% | 0/167 | 0.0% | — |
| rgb_int | main | 2.8% | 0.8% | 4/10 | 92.3% | 0.8% |
| rgb_int | always_refuse | 7.0% | 0.0% | 0/10 | 0.0% | — |
