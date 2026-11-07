# Round 4 Bright-Pro pilot

Fixed custom robotics sample; document-ID weighted aspect coverage only. Not the official 175-query benchmark, a claim-entailment measure, or a checkpoint-selection set. Supporting facts can lie beyond the 800-character display.

| system | weighted aspect recall | cited aspect recall | invalid / 10 | input truncations |
| --- | ---: | ---: | ---: | ---: |
| main | 0.1703 | 0.0731 | 3 | 0 |
| candE1 | 0.0703 | 0.0231 | 5 | 0 |

| query ID | main recall | E1 recall | main final | E1 final |
| --- | ---: | ---: | --- | --- |
| 4 | 1.000 | 0.000 | answer | answer |
| 7 | 0.000 | 0.000 | answer | answer |
| 8 | 0.273 | 0.273 | abstain | abstain |
| 17 | 0.000 | 0.000 | invalid | invalid |
| 18 | 0.231 | 0.231 | answer | answer |
| 27 | 0.200 | 0.200 | invalid | invalid |
| 29 | 0.000 | 0.000 | invalid | invalid |
| 31 | 0.000 | 0.000 | abstain | invalid |
| 33 | 0.000 | 0.000 | abstain | invalid |
| 34 | 0.000 | 0.000 | abstain | abstain |

E1 has lower coverage and more invalid outputs in this pilot. Ten fixed sample queries do not establish general performance or winning chances. No Bright-Pro examples were used for training. Main remains the recommendation.
