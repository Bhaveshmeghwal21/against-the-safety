# Azure GPT-6 Luna vs the saved RAG systems

Same question/variant keys, index, masks, top 3, passage display and three-search budget. Scorer v2; every compared actor uses the same intersection of scorable episodes. Exclusion remains a heuristic, not a proof of missing evidence. Macro is the unweighted mean of success rates on SQuAD single, SQuAD compound, HotpotQA and RGB; success is a supported correct answer or a correct abstention.
Azure exact uses the original prompt/parser and default temperature/reasoning; native uses function tools, minimal output-format normalization and reasoning_effort=none, as required by this deployment. Tiny Aya uses greedy decoding and 160 generated tokens; Azure has a 512-token completion cap including any reasoning. Each Azure mode is a separate stochastic run.
FB maps invalid actions to abstention. V is the previously saved Tiny Aya verifier, available for the three round-5 adapter systems only. Azure has no V; no adapter was run for this comparison. Provider-blocked episodes are excluded, not scored as wrong answers or abstentions.

Round 5: 876 common episodes of 938; provider-unavailable 3; excluded by at least one actor under v2 59.

| system | macro | single | compound | HotpotQA | RGB | unsupported / hidden | supported correct | false refusals | invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| main | 0.413 | 0.540 | 0.225 | 0.526 | 0.363 | 86/315 | 0.291 | 0.223 | 0.082 |
| main+FB+V | 0.507 | 0.583 | 0.388 | 0.551 | 0.508 | 25/315 | 0.276 | 0.390 | 0.000 |
| candA+FB+V | 0.530 | 0.607 | 0.388 | 0.615 | 0.511 | 33/315 | 0.310 | 0.321 | 0.000 |
| candF1+FB+V | 0.557 | 0.607 | 0.519 | 0.603 | 0.499 | 26/315 | 0.324 | 0.376 | 0.000 |
| azure_exact | 0.212 | 0.308 | 0.156 | 0.167 | 0.215 | 36/315 | 0.137 | 0.059 | 0.382 |
| azure_exact+FB | 0.428 | 0.540 | 0.381 | 0.449 | 0.342 | 36/315 | 0.137 | 0.369 | 0.000 |
| azure_native | 0.287 | 0.564 | 0.056 | 0.231 | 0.295 | 62/315 | 0.392 | 0.007 | 0.242 |
| azure_native+FB | 0.548 | 0.787 | 0.375 | 0.551 | 0.478 | 62/315 | 0.392 | 0.027 | 0.000 |

## Paired differences

95% bootstrap intervals resample complete question IDs within each dataset, seed 47; they do not remove article-level correlation.

- azure_exact/main: macro -0.202 [-0.243, -0.160]; unsupported rate -0.159 [-0.218, -0.097].
- azure_exact/main+FB+V: macro -0.296 [-0.331, -0.260]; unsupported rate +0.035 [-0.010, +0.081].
- azure_exact/candF1+FB+V: macro -0.345 [-0.385, -0.306]; unsupported rate +0.032 [-0.013, +0.075].
- azure_exact+FB/main: macro +0.015 [-0.026, +0.053]; unsupported rate -0.159 [-0.218, -0.097].
- azure_exact+FB/main+FB+V: macro -0.079 [-0.111, -0.048]; unsupported rate +0.035 [-0.010, +0.081].
- azure_exact+FB/candF1+FB+V: macro -0.129 [-0.168, -0.091]; unsupported rate +0.032 [-0.013, +0.075].
- azure_native/main: macro -0.127 [-0.168, -0.085]; unsupported rate -0.076 [-0.150, -0.003].
- azure_native/main+FB+V: macro -0.221 [-0.259, -0.178]; unsupported rate +0.117 [+0.059, +0.176].
- azure_native/candF1+FB+V: macro -0.270 [-0.319, -0.221]; unsupported rate +0.114 [+0.057, +0.173].
- azure_native+FB/main: macro +0.134 [+0.090, +0.176]; unsupported rate -0.076 [-0.150, -0.003].
- azure_native+FB/main+FB+V: macro +0.040 [-0.000, +0.082]; unsupported rate +0.117 [+0.059, +0.176].
- azure_native+FB/candF1+FB+V: macro -0.009 [-0.056, +0.038]; unsupported rate +0.114 [+0.057, +0.173].

## Earlier external suites

These use scorer v2 on the common intersection. Only raw main and candA (and main plus fail-closed) were previously evaluated on these suites; the final round-5 verifier and candF1 were not run here. These are internal samples, not official leaderboard scores.

| benchmark | system | common / expected | success | unsupported / hidden | supported correct | false refusals | invalid |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2wiki | main | 165/166 | 0.521 | 3/78 | 0.218 | 0.207 | 0.055 |
| 2wiki | main+FB | 165/166 | 0.570 | 3/78 | 0.218 | 0.218 | 0.000 |
| 2wiki | candA | 165/166 | 0.545 | 5/78 | 0.230 | 0.184 | 0.018 |
| 2wiki | azure_exact | 165/166 | 0.152 | 0/78 | 0.023 | 0.115 | 0.630 |
| 2wiki | azure_exact+FB | 165/166 | 0.485 | 0/78 | 0.023 | 0.678 | 0.000 |
| 2wiki | azure_native | 165/166 | 0.212 | 1/78 | 0.207 | 0.057 | 0.491 |
| 2wiki | azure_native+FB | 165/166 | 0.576 | 1/78 | 0.207 | 0.299 | 0.000 |
| *2wiki exclusions* | missing 0; v2 1 | | | | | | |
| musique | main | 165/168 | 0.497 | 1/77 | 0.182 | 0.523 | 0.097 |
| musique | main+FB | 165/168 | 0.558 | 1/77 | 0.182 | 0.591 | 0.000 |
| musique | candA | 165/168 | 0.564 | 2/77 | 0.273 | 0.466 | 0.061 |
| musique | azure_exact | 165/168 | 0.182 | 1/77 | 0.057 | 0.091 | 0.642 |
| musique | azure_exact+FB | 165/168 | 0.491 | 1/77 | 0.057 | 0.716 | 0.000 |
| musique | azure_native | 165/168 | 0.188 | 2/77 | 0.250 | 0.114 | 0.515 |
| musique | azure_native+FB | 165/168 | 0.588 | 2/77 | 0.250 | 0.330 | 0.000 |
| *musique exclusions* | missing 0; v2 3 | | | | | | |
| squad2 | main | 197/202 | 0.467 | 5/124 | 0.055 | 0.740 | 0.218 |
| squad2 | main+FB | 197/202 | 0.624 | 5/124 | 0.055 | 0.904 | 0.000 |
| squad2 | candA | 197/202 | 0.563 | 6/124 | 0.082 | 0.753 | 0.096 |
| squad2 | azure_exact | 197/202 | 0.168 | 19/124 | 0.096 | 0.041 | 0.604 |
| squad2 | azure_exact+FB | 197/202 | 0.569 | 19/124 | 0.096 | 0.589 | 0.000 |
| squad2 | azure_native | 197/202 | 0.335 | 51/124 | 0.466 | 0.014 | 0.213 |
| squad2 | azure_native+FB | 197/202 | 0.543 | 51/124 | 0.466 | 0.027 | 0.000 |
| *squad2 exclusions* | missing 0; v2 5 | | | | | | |
| rgb | main | 405/453 | 0.356 | 57/119 | 0.336 | 0.073 | 0.054 |
| rgb | main+FB | 405/453 | 0.390 | 57/119 | 0.336 | 0.101 | 0.000 |
| rgb | candA | 405/453 | 0.363 | 73/119 | 0.360 | 0.024 | 0.010 |
| rgb | azure_exact | 405/453 | 0.198 | 13/119 | 0.087 | 0.014 | 0.230 |
| rgb | azure_exact+FB | 405/453 | 0.323 | 13/119 | 0.087 | 0.161 | 0.000 |
| rgb | azure_native | 405/453 | 0.286 | 34/119 | 0.318 | 0.003 | 0.158 |
| rgb | azure_native+FB | 405/453 | 0.435 | 34/119 | 0.318 | 0.017 | 0.000 |
| *rgb exclusions* | missing 0; v2 48 | | | | | | |
| rgb_int | main | 141/142 | 0.014 | 4/9 | 0.000 | 0.030 | 0.028 |
| rgb_int | main+FB | 141/142 | 0.035 | 4/9 | 0.000 | 0.038 | 0.000 |
| rgb_int | candA | 141/142 | 0.035 | 4/9 | 0.000 | 0.008 | 0.000 |
| rgb_int | azure_exact | 141/142 | 0.021 | 1/9 | 0.008 | 0.121 | 0.277 |
| rgb_int | azure_exact+FB | 141/142 | 0.064 | 1/9 | 0.008 | 0.371 | 0.000 |
| rgb_int | azure_native | 141/142 | 0.028 | 2/9 | 0.030 | 0.015 | 0.128 |
| rgb_int | azure_native+FB | 141/142 | 0.078 | 2/9 | 0.030 | 0.098 | 0.000 |
| *rgb_int exclusions* | missing 0; v2 1 | | | | | | |

The long-form data records exclusions by model. Models may retrieve different alternate support; applying one shared intersection avoids comparing unequal denominators, but incomplete alias and semantic checks can still mislabel cases.
