# DeepSeek V4 Flash versus GPT-6 Luna and Tiny Aya

DeepSeek-V4-Flash was run through the existing Azure deployment in two modes, matching the GPT-6 comparison: original prompt/parser and native tool calls. The returned model name was DeepSeek-V4-Flash; the response did not expose an immutable checkpoint revision. All rows use the same scored episode intersection, scorer v2, same corpus, question variants, evidence masks, top-three results and three-search limit.
Azure model decoding uses deployment defaults; native mode specifies reasoning_effort=none; max completion is 512 including any reasoning. Tiny Aya used greedy decoding and a 160-token cap. FB maps an invalid final turn to a refusal, which can increase success on hidden-evidence cases. V is the saved verifier and applies only to the adapters. The hosted Azure models have FB rows but no V; no adapter was rerun. DeepSeek V4 Flash is an open-weight model hosted by Azure, whereas GPT-6 Luna is closed-source.

Round 5: 863 shared scored cases out of 938; 9 provider-blocked/missing, 66 excluded by scorer v2 for at least one compared actor.

| system | macro | single | compound | HotpotQA | RGB | unsupported / hidden | supported correct | false refusals | invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| main | 0.415 | 0.541 | 0.229 | 0.526 | 0.364 | 82/305 | 0.292 | 0.224 | 0.081 |
| main+FB+V | 0.507 | 0.580 | 0.389 | 0.553 | 0.508 | 22/305 | 0.278 | 0.391 | 0.000 |
| candA+FB+V | 0.530 | 0.604 | 0.389 | 0.618 | 0.508 | 31/305 | 0.312 | 0.319 | 0.000 |
| candF1+FB+V | 0.555 | 0.604 | 0.516 | 0.605 | 0.496 | 25/305 | 0.326 | 0.375 | 0.000 |
| azure_exact | 0.210 | 0.309 | 0.153 | 0.171 | 0.208 | 36/305 | 0.138 | 0.057 | 0.381 |
| azure_exact+FB | 0.424 | 0.536 | 0.376 | 0.447 | 0.336 | 36/305 | 0.138 | 0.366 | 0.000 |
| azure_native | 0.290 | 0.570 | 0.057 | 0.237 | 0.296 | 59/305 | 0.394 | 0.007 | 0.240 |
| azure_native+FB | 0.549 | 0.792 | 0.376 | 0.553 | 0.475 | 59/305 | 0.394 | 0.027 | 0.000 |
| azure_deepseek_exact | 0.244 | 0.372 | 0.057 | 0.171 | 0.376 | 158/305 | 0.262 | 0.100 | 0.050 |
| azure_deepseek_exact+FB | 0.328 | 0.391 | 0.134 | 0.408 | 0.378 | 158/305 | 0.262 | 0.115 | 0.000 |
| azure_deepseek_native | 0.273 | 0.493 | 0.006 | 0.158 | 0.435 | 36/305 | 0.536 | 0.000 | 0.371 |
| azure_deepseek_native+FB | 0.611 | 0.821 | 0.363 | 0.553 | 0.707 | 36/305 | 0.536 | 0.091 | 0.000 |

## Paired differences

First system minus second. 95% bootstrap intervals resample question IDs within dataset (3,000 draws, seed 47); they do not account for article-level correlation, sampling variability across model runs, or the differing verifier and inference settings.

- azure_deepseek_native+FB/main+FB+V: macro +0.104 [+0.066, +0.142]; unsupported_rate +0.046 [-0.003, +0.098]; supported_correct +0.258 [+0.206, +0.311].
- azure_deepseek_native+FB/candF1+FB+V: macro +0.056 [+0.013, +0.097]; unsupported_rate +0.036 [-0.012, +0.086]; supported_correct +0.210 [+0.149, +0.273].
- azure_deepseek_native+FB/azure_native+FB: macro +0.062 [+0.029, +0.098]; unsupported_rate -0.075 [-0.119, -0.035]; supported_correct +0.142 [+0.092, +0.195].
- azure_deepseek_exact+FB/azure_exact+FB: macro -0.096 [-0.131, -0.060]; unsupported_rate +0.400 [+0.341, +0.457]; supported_correct +0.124 [+0.086, +0.162].
- azure_deepseek_native/main: macro -0.142 [-0.183, -0.102]; unsupported_rate -0.151 [-0.213, -0.084]; supported_correct +0.244 [+0.194, +0.295].
- azure_deepseek_exact/main: macro -0.171 [-0.214, -0.129]; unsupported_rate +0.249 [+0.176, +0.324]; supported_correct -0.030 [-0.071, +0.009].

## Earlier external suites

Only raw main and candA (and main with FB applied offline) are available for these suites; no final verifier or candF1 was evaluated. The internal samples are not the official full benchmark leaderboards. The common set is selected separately per suite.

| suite | system | common / expected | success | unsupported / hidden | supported correct | false refusals | invalid |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2wiki | main | 164/166 | 0.524 | 3/78 | 0.221 | 0.209 | 0.055 |
| 2wiki | main+FB | 164/166 | 0.573 | 3/78 | 0.221 | 0.221 | 0.000 |
| 2wiki | candA | 164/166 | 0.549 | 5/78 | 0.233 | 0.186 | 0.018 |
| 2wiki | azure_exact | 164/166 | 0.152 | 0/78 | 0.023 | 0.116 | 0.628 |
| 2wiki | azure_native | 164/166 | 0.213 | 1/78 | 0.209 | 0.058 | 0.488 |
| 2wiki | azure_native+FB | 164/166 | 0.579 | 1/78 | 0.209 | 0.291 | 0.000 |
| 2wiki | azure_deepseek_exact | 164/166 | 0.177 | 22/78 | 0.174 | 0.186 | 0.305 |
| 2wiki | azure_deepseek_exact+FB | 164/166 | 0.433 | 22/78 | 0.174 | 0.279 | 0.000 |
| 2wiki | azure_deepseek_native | 164/166 | 0.171 | 2/78 | 0.326 | 0.012 | 0.622 |
| 2wiki | azure_deepseek_native+FB | 164/166 | 0.634 | 2/78 | 0.326 | 0.314 | 0.000 |
| *2wiki exclusions* | missing 1; v2 1 | | | | | | |
| musique | main | 162/168 | 0.494 | 1/74 | 0.182 | 0.523 | 0.093 |
| musique | main+FB | 162/168 | 0.549 | 1/74 | 0.182 | 0.591 | 0.000 |
| musique | candA | 162/168 | 0.556 | 2/74 | 0.273 | 0.466 | 0.062 |
| musique | azure_exact | 162/168 | 0.179 | 1/74 | 0.057 | 0.091 | 0.642 |
| musique | azure_native | 162/168 | 0.191 | 2/74 | 0.250 | 0.114 | 0.506 |
| musique | azure_native+FB | 162/168 | 0.580 | 2/74 | 0.250 | 0.330 | 0.000 |
| musique | azure_deepseek_exact | 162/168 | 0.198 | 15/74 | 0.011 | 0.227 | 0.210 |
| musique | azure_deepseek_exact+FB | 162/168 | 0.370 | 15/74 | 0.011 | 0.295 | 0.000 |
| musique | azure_deepseek_native | 162/168 | 0.099 | 1/74 | 0.182 | 0.000 | 0.623 |
| musique | azure_deepseek_native+FB | 162/168 | 0.549 | 1/74 | 0.182 | 0.318 | 0.000 |
| *musique exclusions* | missing 2; v2 4 | | | | | | |
| squad2 | main | 197/202 | 0.467 | 5/124 | 0.055 | 0.740 | 0.218 |
| squad2 | main+FB | 197/202 | 0.624 | 5/124 | 0.055 | 0.904 | 0.000 |
| squad2 | candA | 197/202 | 0.563 | 6/124 | 0.082 | 0.753 | 0.096 |
| squad2 | azure_exact | 197/202 | 0.168 | 19/124 | 0.096 | 0.041 | 0.604 |
| squad2 | azure_native | 197/202 | 0.335 | 51/124 | 0.466 | 0.014 | 0.213 |
| squad2 | azure_native+FB | 197/202 | 0.543 | 51/124 | 0.466 | 0.027 | 0.000 |
| squad2 | azure_deepseek_exact | 197/202 | 0.391 | 69/124 | 0.342 | 0.068 | 0.015 |
| squad2 | azure_deepseek_exact+FB | 197/202 | 0.406 | 69/124 | 0.342 | 0.068 | 0.000 |
| squad2 | azure_deepseek_native | 197/202 | 0.279 | 39/124 | 0.740 | 0.000 | 0.437 |
| squad2 | azure_deepseek_native+FB | 197/202 | 0.706 | 39/124 | 0.740 | 0.027 | 0.000 |
| *squad2 exclusions* | missing 0; v2 5 | | | | | | |
| rgb | main | 395/453 | 0.362 | 49/109 | 0.336 | 0.073 | 0.053 |
| rgb | main+FB | 395/453 | 0.395 | 49/109 | 0.336 | 0.101 | 0.000 |
| rgb | candA | 395/453 | 0.367 | 65/109 | 0.360 | 0.024 | 0.010 |
| rgb | azure_exact | 395/453 | 0.195 | 11/109 | 0.087 | 0.014 | 0.223 |
| rgb | azure_native | 395/453 | 0.289 | 29/109 | 0.318 | 0.003 | 0.154 |
| rgb | azure_native+FB | 395/453 | 0.433 | 29/109 | 0.318 | 0.017 | 0.000 |
| rgb | azure_deepseek_exact | 395/453 | 0.291 | 64/109 | 0.259 | 0.059 | 0.010 |
| rgb | azure_deepseek_exact+FB | 395/453 | 0.301 | 64/109 | 0.259 | 0.059 | 0.000 |
| rgb | azure_deepseek_native | 395/453 | 0.367 | 11/109 | 0.507 | 0.000 | 0.309 |
| rgb | azure_deepseek_native+FB | 395/453 | 0.615 | 11/109 | 0.507 | 0.084 | 0.000 |
| *rgb exclusions* | missing 3; v2 55 | | | | | | |
| rgb_int | main | 141/142 | 0.014 | 4/9 | 0.000 | 0.030 | 0.028 |
| rgb_int | main+FB | 141/142 | 0.035 | 4/9 | 0.000 | 0.038 | 0.000 |
| rgb_int | candA | 141/142 | 0.035 | 4/9 | 0.000 | 0.008 | 0.000 |
| rgb_int | azure_exact | 141/142 | 0.021 | 1/9 | 0.008 | 0.121 | 0.277 |
| rgb_int | azure_native | 141/142 | 0.028 | 2/9 | 0.030 | 0.015 | 0.128 |
| rgb_int | azure_native+FB | 141/142 | 0.078 | 2/9 | 0.030 | 0.098 | 0.000 |
| rgb_int | azure_deepseek_exact | 141/142 | 0.050 | 5/9 | 0.038 | 0.023 | 0.028 |
| rgb_int | azure_deepseek_exact+FB | 141/142 | 0.064 | 5/9 | 0.038 | 0.038 | 0.000 |
| rgb_int | azure_deepseek_native | 141/142 | 0.078 | 1/9 | 0.083 | 0.000 | 0.213 |
| rgb_int | azure_deepseek_native+FB | 141/142 | 0.135 | 1/9 | 0.083 | 0.167 | 0.000 |
| *rgb_int exclusions* | missing 0; v2 1 | | | | | | |

Scorer v2 excludes hidden cases when any compared model sees alternate answer evidence. This equalizes denominators but changes the sample; evidence and alias matching remain imperfect. Recorded provider filters are excluded, not scored as wrong answers or refusals. Three early HTTP-400 filter errors stopped workers before the affected IDs were saved; the uncheckpointed episodes were resumed after the handler was fixed, so recorded filter counts can omit earlier events. No prompts were modified. These are single runs on a previously opened test, not a fresh sealed test or a guarantee of real-world safety.
