# Round 4 frozen final test

`base` is the original Tiny Aya Global without our LoRA, not the separate Tiny Aya Base model.

| system | macro evidence success | single | compound | unsupported / hidden | false refusals | invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.016 | 0.027 | 0.005 | 112/187 | 0.000 | 0.323 |
| main | 0.357 | 0.481 | 0.233 | 40/187 | 0.302 | 0.111 |
| candA | 0.375 | 0.554 | 0.196 | 55/187 | 0.225 | 0.042 |
| candE1 | 0.377 | 0.458 | 0.296 | 33/187 | 0.237 | 0.214 |

Candidate versus main: +0.020, 95% interval [-0.02478856222311726, 0.06352194925493354].
Unsupported-answer difference: -0.037, 95% interval [-0.08108108108108109, 0.010416666666666657].
Primary-test recommendation, provisional until transfer checks and answer inspection: **main**.

A diagnostic checkpoint cannot be promoted from this test when development selection failed.

Post-evaluation trace audit, 7 November: some gold-mask unsupported labels are confirmed false positives because alternative shown passages still contain the answer. Source-question inconsistencies and answer-matching limitations also remain. The frozen numbers above are unchanged protocol scores, not manually verified hallucination or factual-accuracy rates. See the [dated result and label audit](../research/2026-11-06-round4-results.md) for exact cases and interpretation. The decision remains to retain main.
