# Scorer version 2 against version 1 on saved transcripts

v2 = token precision >= 0.5 against the closest gold form, plus exclusion of hidden-evidence episodes where a shown passage holds the answer.

| run | scorer | n | single | compound | supported success | unsupported / hidden | false refusals | invalid |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| candA_bench_r4dev | v1 | 321 | 0.590 | 0.279 | 0.280 | 20/132 | 0.307 | 0.053 |
| candA_bench_r4dev | v2 | 319 | 0.587 | 0.279 | 0.280 | 20/130 | 0.307 | 0.053 |
| candA_dev | v1 | 159 | - | - | 0.349 | 9/73 | 0.163 | 0.025 |
| candA_dev | v2 | 159 | - | - | 0.326 | 9/73 | 0.163 | 0.025 |
| candE1_bench_r4dev | v1 | 321 | 0.429 | 0.351 | 0.317 | 17/132 | 0.270 | 0.249 |
| candE1_bench_r4dev | v2 | 320 | 0.421 | 0.333 | 0.302 | 17/131 | 0.270 | 0.250 |
| candE1_dev | v1 | 159 | - | - | 0.314 | 5/73 | 0.291 | 0.075 |
| candE1_dev | v2 | 159 | - | - | 0.267 | 5/73 | 0.291 | 0.075 |
| candE2_bench_r4dev | v1 | 321 | 0.538 | 0.324 | 0.370 | 21/132 | 0.243 | 0.150 |
| candE2_bench_r4dev | v2 | 320 | 0.526 | 0.306 | 0.349 | 21/131 | 0.243 | 0.150 |
| candE2_dev | v1 | 159 | - | - | 0.279 | 8/73 | 0.198 | 0.031 |
| candE2_dev | v2 | 159 | - | - | 0.244 | 8/73 | 0.198 | 0.031 |
| candF1_bench_r4dev | v1 | 321 | 0.567 | 0.486 | 0.392 | 13/132 | 0.280 | 0.112 |
| candF1_bench_r4dev | v2 | 320 | 0.560 | 0.477 | 0.381 | 13/131 | 0.280 | 0.113 |
| candF1_dev | v1 | 159 | - | - | 0.349 | 9/73 | 0.233 | 0.050 |
| candF1_dev | v2 | 159 | - | - | 0.314 | 9/73 | 0.233 | 0.050 |
| candF2_bench_r4dev | v1 | 321 | 0.595 | 0.396 | 0.402 | 22/132 | 0.238 | 0.090 |
| candF2_bench_r4dev | v2 | 320 | 0.589 | 0.387 | 0.392 | 22/131 | 0.238 | 0.091 |
| candF2_dev | v1 | 159 | - | - | 0.395 | 15/73 | 0.140 | 0.025 |
| candF2_dev | v2 | 159 | - | - | 0.360 | 15/73 | 0.140 | 0.025 |
| main_bench_r4dev | v1 | 321 | 0.486 | 0.297 | 0.243 | 18/132 | 0.376 | 0.134 |
| main_bench_r4dev | v2 | 320 | 0.483 | 0.297 | 0.243 | 18/131 | 0.376 | 0.134 |
| main_dev | v1 | 159 | - | - | 0.314 | 2/73 | 0.233 | 0.025 |
| main_dev | v2 | 159 | - | - | 0.256 | 2/73 | 0.233 | 0.025 |
| base_bench_r4test | v1 | 449 | 0.027 | 0.005 | 0.000 | 112/187 | 0.000 | 0.323 |
| base_bench_r4test | v2 | 449 | 0.027 | 0.005 | 0.000 | 112/187 | 0.000 | 0.323 |
| candA_bench_r4test | v1 | 449 | 0.554 | 0.196 | 0.252 | 55/187 | 0.225 | 0.042 |
| candA_bench_r4test | v2 | 444 | 0.541 | 0.196 | 0.244 | 54/182 | 0.225 | 0.043 |
| candE1_bench_r4test | v1 | 449 | 0.458 | 0.296 | 0.336 | 33/187 | 0.237 | 0.214 |
| candE1_bench_r4test | v2 | 446 | 0.447 | 0.265 | 0.302 | 32/184 | 0.237 | 0.213 |
| main_bench_r4test | v1 | 449 | 0.481 | 0.233 | 0.202 | 40/187 | 0.302 | 0.111 |
| main_bench_r4test | v2 | 443 | 0.471 | 0.229 | 0.195 | 39/181 | 0.302 | 0.111 |

Excluded under v2 (hidden-evidence episodes with a shown passage holding the answer):

- candA_bench_r4dev: 2 (5709b503200fba1400368289:r4_missing_0, 5709b503200fba1400368289:support_masked)
- candA_dev: 0
- candE1_bench_r4dev: 1 (5709b503200fba1400368289:support_masked)
- candE1_dev: 0
- candE2_bench_r4dev: 1 (5709b503200fba1400368289:support_masked)
- candE2_dev: 0
- candF1_bench_r4dev: 1 (5709b503200fba1400368289:support_masked)
- candF1_dev: 0
- candF2_bench_r4dev: 1 (5709b503200fba1400368289:support_masked)
- candF2_dev: 0
- main_bench_r4dev: 1 (5709b503200fba1400368289:support_masked)
- main_dev: 0
- base_bench_r4test: 0
- candA_bench_r4test: 5 (56f730303d8e2e1400e37412:r4_missing_0, 5729f3426aef051400155144:r4_missing_0, 5729f3426aef051400155144:support_masked, 572f73b704bcaa1900d7696d:r4_missing_0, 572fffe804bcaa1900d76ff9:support_masked)
- candE1_bench_r4test: 3 (5729f3426aef051400155144:r4_missing_0, 572fffe804bcaa1900d76ff9:r4_missing_0, 572fffe804bcaa1900d76ff9:support_masked)
- main_bench_r4test: 6 (5706f6a790286e26004fc777:support_masked, 5729f3426aef051400155144:r4_missing_0, 5729f3426aef051400155144:support_masked, 572fffe804bcaa1900d76ff9:r4_missing_0, 572fffe804bcaa1900d76ff9:support_masked, cmp-56e783a900c9c71400d77203-56e77ae437bdd419002c400e:r4_partial_1)
