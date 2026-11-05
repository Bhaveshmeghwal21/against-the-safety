# External benchmarks (lenient reading)

Evaluation only: nothing was trained on or chosen with these sets (plan section 22). Same harness as HotpotQA: shared index per benchmark, three searches, top 3, 800 characters shown. Not comparable with official leaderboard numbers.

## Summary: ends well (supported answer when answerable, abstain when the documents cannot answer)

| benchmark | Tiny Aya base | main | candA | control | main minus base, 95% CI | main minus candA, 95% CI | closed book |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| 2WikiMultihopQA | 0.018 (3/166) | 0.566 (94/166) | 0.620 (103/166) | 0.241 (40/166) | +0.548 [+0.479, +0.619] | -0.054 [-0.113, +0.000] | 0.276 |
| MuSiQue (2-hop) | 0.018 (3/168) | 0.536 (90/168) | 0.577 (97/168) | 0.250 (42/168) | +0.518 [+0.455, +0.581] | -0.042 [-0.098, +0.012] | 0.045 |
| SQuAD 2.0 | 0.010 (2/202) | 0.475 (96/202) | 0.559 (113/202) | 0.040 (8/202) | +0.465 [+0.404, +0.528] | -0.084 [-0.150, -0.015] | 0.137 |
| RGB | 0.157 (71/453) | 0.618 (280/453) | 0.660 (299/453) | 0.530 (240/453) | +0.461 [+0.406, +0.518] | -0.042 [-0.081, -0.004] | 0.180 |
| RGB integration | 0.070 (10/142) | 0.028 (4/142) | 0.049 (7/142) | 0.007 (1/142) | -0.042 [-0.097, +0.008] | -0.021 [-0.056, +0.014] | 0.100 |
| **all five, pooled** | 0.079 (89/1131) | 0.499 (564/1131) | 0.547 (619/1131) | 0.293 (331/1131) | +0.420 [+0.386, +0.453] | -0.049 [-0.072, -0.026] | |

## 2WikiMultihopQA

| metric | Tiny Aya base | main | candA | control |
| --- | ---: | ---: | ---: | ---: |
| ends well (all scored episodes) | 0.018 (3/166) | 0.566 (94/166) | 0.620 (103/166) | 0.241 (40/166) |
| supported correct, open search (answerable) | 0.000 (0/87) | 0.310 (27/87) | 0.368 (32/87) | 0.460 (40/87) |
| false abstain, open search (lower is better) | 0.011 (1/87) | 0.207 (18/87) | 0.184 (16/87) | 0.000 (0/87) |
| false abstain with the full evidence already shown, open search (post hoc; lower is better) | 0.000 (0/87) | 0.046 (4/87) | 0.046 (4/87) | 0.000 (0/87) |
| answer correct, citation not required, open search (post hoc) | 0.161 (14/87) | 0.310 (27/87) | 0.368 (32/87) | 0.460 (40/87) |
| correct abstain, support masked | 0.038 (3/79) | 0.848 (67/79) | 0.899 (71/79) | 0.000 (0/79) |
| answers when the documents cannot support it (lower is better) | 0.886 (70/79) | 0.038 (3/79) | 0.063 (5/79) | 0.165 (13/79) |
| invalid or rule-breaking turns (lower is better) | 0.048 (8/166) | 0.060 (10/166) | 0.018 (3/166) | 0.428 (71/166) |
| mean searches per episode | 1.0 | 2.36 | 2.11 | 1.99 |

| ends well, paired | before | after | difference | 95% CI | episodes / questions |
| --- | ---: | ---: | ---: | --- | ---: |
| main vs base | 0.018 | 0.566 | +0.548 | [+0.479, +0.619] | 166 / 87 |
| main vs candA | 0.620 | 0.566 | -0.054 | [-0.113, +0.000] | 166 / 87 |
| main vs control | 0.241 | 0.566 | +0.325 | [+0.259, +0.391] | 166 / 87 |

| AuditKIT (all episodes) | Tiny Aya base | main | candA | control |
| --- | ---: | ---: | ---: | ---: |
| evidence_set_recall | 0.322 | 0.552 | 0.563 | 0.667 |
| citation_support | 0.207 | 0.782 | 0.816 | 0.943 |
| abstention | 0.536 | 0.819 | 0.855 | 0.524 |
| tool_call_validity | 1.000 | 1.000 | 1.000 | 1.000 |
| redundant_tool_calls (lower is better) | 0.000 | 0.020 | 0.006 | 0.112 |
| agent_loop_detection (1.0 = loop-free) | 1.000 | 0.922 | 0.982 | 0.620 |

Closed book (base model, question only): 0.276 of 87 answerable questions.

## MuSiQue (2-hop)

| metric | Tiny Aya base | main | candA | control |
| --- | ---: | ---: | ---: | ---: |
| ends well (all scored episodes) | 0.018 (3/168) | 0.536 (90/168) | 0.577 (97/168) | 0.250 (42/168) |
| supported correct, open search (answerable) | 0.034 (3/88) | 0.250 (22/88) | 0.284 (25/88) | 0.477 (42/88) |
| false abstain, open search (lower is better) | 0.000 (0/88) | 0.523 (46/88) | 0.466 (41/88) | 0.000 (0/88) |
| false abstain with the full evidence already shown, open search (post hoc; lower is better) | 0.000 (0/88) | 0.170 (15/88) | 0.159 (14/88) | 0.000 (0/88) |
| answer correct, citation not required, open search (post hoc) | 0.114 (10/88) | 0.250 (22/88) | 0.284 (25/88) | 0.477 (42/88) |
| correct abstain, support masked | 0.000 (0/80) | 0.850 (68/80) | 0.900 (72/80) | 0.000 (0/80) |
| answers when the documents cannot support it (lower is better) | 0.912 (73/80) | 0.013 (1/80) | 0.025 (2/80) | 0.075 (6/80) |
| invalid or rule-breaking turns (lower is better) | 0.089 (15/168) | 0.101 (17/168) | 0.060 (10/168) | 0.589 (99/168) |
| mean searches per episode | 0.97 | 2.49 | 2.14 | 2.22 |

| ends well, paired | before | after | difference | 95% CI | episodes / questions |
| --- | ---: | ---: | ---: | --- | ---: |
| main vs base | 0.018 | 0.536 | +0.518 | [+0.455, +0.581] | 168 / 88 |
| main vs candA | 0.577 | 0.536 | -0.042 | [-0.098, +0.012] | 168 / 88 |
| main vs control | 0.250 | 0.536 | +0.286 | [+0.222, +0.349] | 168 / 88 |

| AuditKIT (all episodes) | Tiny Aya base | main | candA | control |
| --- | ---: | ---: | ---: | ---: |
| evidence_set_recall | 0.227 | 0.511 | 0.523 | 0.716 |
| citation_support | 0.227 | 0.409 | 0.489 | 0.716 |
| abstention | 0.524 | 0.655 | 0.708 | 0.524 |
| tool_call_validity | 1.000 | 1.000 | 1.000 | 1.000 |
| redundant_tool_calls (lower is better) | 0.000 | 0.035 | 0.021 | 0.168 |
| agent_loop_detection (1.0 = loop-free) | 1.000 | 0.893 | 0.935 | 0.476 |

Closed book (base model, question only): 0.045 of 88 answerable questions.

## SQuAD 2.0

| metric | Tiny Aya base | main | candA | control |
| --- | ---: | ---: | ---: | ---: |
| ends well (all scored episodes) | 0.010 (2/202) | 0.475 (96/202) | 0.559 (113/202) | 0.040 (8/202) |
| supported correct, open search (answerable) | 0.000 (0/73) | 0.055 (4/73) | 0.082 (6/73) | 0.110 (8/73) |
| false abstain, open search (lower is better) | 0.000 (0/73) | 0.740 (54/73) | 0.753 (55/73) | 0.000 (0/73) |
| false abstain with the full evidence already shown, open search (post hoc; lower is better) | 0.000 (0/73) | 0.740 (54/73) | 0.753 (55/73) | 0.000 (0/73) |
| answer correct, citation not required, open search (post hoc) | 0.178 (13/73) | 0.068 (5/73) | 0.110 (8/73) | 0.123 (9/73) |
| correct abstain, support masked | 0.041 (2/49) | 0.714 (35/49) | 0.776 (38/49) | 0.000 (0/49) |
| correct abstain, natural unanswerable (SQuAD 2.0) | 0.000 (0/80) | 0.713 (57/80) | 0.863 (69/80) | 0.000 (0/80) |
| answers when the documents cannot support it (lower is better) | 0.798 (103/129) | 0.039 (5/129) | 0.047 (6/129) | 0.140 (18/129) |
| invalid or rule-breaking turns (lower is better) | 0.203 (41/202) | 0.218 (44/202) | 0.109 (22/202) | 0.832 (168/202) |
| mean searches per episode | 0.99 | 2.52 | 2.12 | 2.33 |

| ends well, paired | before | after | difference | 95% CI | episodes / questions |
| --- | ---: | ---: | ---: | --- | ---: |
| main vs base | 0.010 | 0.475 | +0.465 | [+0.404, +0.528] | 202 / 153 |
| main vs candA | 0.559 | 0.475 | -0.084 | [-0.150, -0.015] | 202 / 153 |
| main vs control | 0.040 | 0.475 | +0.436 | [+0.366, +0.507] | 202 / 153 |

| AuditKIT (all episodes) | Tiny Aya base | main | candA | control |
| --- | ---: | ---: | ---: | ---: |
| evidence_set_recall | 0.918 | 1.000 | 1.000 | 1.000 |
| citation_support | 0.055 | 0.096 | 0.164 | 0.219 |
| abstention | 0.371 | 0.550 | 0.619 | 0.361 |
| tool_call_validity | 1.000 | 1.000 | 1.000 | 1.000 |
| redundant_tool_calls (lower is better) | 0.002 | 0.071 | 0.036 | 0.232 |
| agent_loop_detection (1.0 = loop-free) | 0.995 | 0.782 | 0.891 | 0.272 |

Closed book (base model, question only): 0.137 of 73 answerable questions.

## RGB

| metric | Tiny Aya base | main | candA | control |
| --- | ---: | ---: | ---: | ---: |
| ends well (all scored episodes) | 0.157 (71/453) | 0.618 (280/453) | 0.660 (299/453) | 0.530 (240/453) |
| supported correct, open search (answerable) | 0.186 (16/86) | 0.605 (52/86) | 0.802 (69/86) | 0.826 (71/86) |
| false abstain, open search (lower is better) | 0.000 (0/86) | 0.209 (18/86) | 0.081 (7/86) | 0.000 (0/86) |
| false abstain with the full evidence already shown, open search (post hoc; lower is better) | 0.000 (0/86) | 0.209 (18/86) | 0.081 (7/86) | 0.000 (0/86) |
| answer correct, citation not required, open search (post hoc) | 0.767 (66/86) | 0.616 (53/86) | 0.826 (71/86) | 0.826 (71/86) |
| correct abstain, support masked | 0.000 (0/67) | 0.522 (35/67) | 0.582 (39/67) | 0.000 (0/67) |
| answers when the documents cannot support it (lower is better) | 0.892 (149/167) | 0.533 (89/167) | 0.665 (111/167) | 0.749 (125/167) |
| RGB accuracy, noise 0 (answer correct, citation not required) | 0.770 (77/100) | 0.910 (91/100) | 0.930 (93/100) | 0.930 (93/100) |
| RGB accuracy, noise 0.8 | 0.850 (85/100) | 0.800 (80/100) | 0.840 (84/100) | 0.820 (82/100) |
| RGB rejection rate, noise 1 (abstains) | 0.010 (1/100) | 0.260 (26/100) | 0.150 (15/100) | 0.000 (0/100) |
| invalid or rule-breaking turns (lower is better) | 0.108 (49/453) | 0.055 (25/453) | 0.009 (4/453) | 0.117 (53/453) |
| mean searches per episode | 0.82 | 1.04 | 0.82 | 0.95 |

| ends well, paired | before | after | difference | 95% CI | episodes / questions |
| --- | ---: | ---: | ---: | --- | ---: |
| main vs base | 0.157 | 0.618 | +0.461 | [+0.406, +0.518] | 453 / 100 |
| main vs candA | 0.660 | 0.618 | -0.042 | [-0.081, -0.004] | 453 / 100 |
| main vs control | 0.530 | 0.618 | +0.088 | [+0.049, +0.126] | 453 / 100 |

| AuditKIT (all episodes) | Tiny Aya base | main | candA | control |
| --- | ---: | ---: | ---: | ---: |
| evidence_set_recall | 0.968 | 0.960 | 0.960 | 0.960 |
| citation_support | 0.558 | 0.899 | 0.969 | 0.962 |
| abstention | 0.634 | 0.720 | 0.735 | 0.631 |
| tool_call_validity | 1.000 | 1.000 | 1.000 | 1.000 |
| redundant_tool_calls (lower is better) | 0.003 | 0.030 | 0.006 | 0.049 |
| agent_loop_detection (1.0 = loop-free) | 0.993 | 0.945 | 0.991 | 0.903 |

Closed book (base model, question only): 0.180 of 100 answerable questions.

## RGB integration

| metric | Tiny Aya base | main | candA | control |
| --- | ---: | ---: | ---: | ---: |
| ends well (all scored episodes) | 0.070 (10/142) | 0.028 (4/142) | 0.049 (7/142) | 0.007 (1/142) |
| supported correct, open search (answerable) | 0.083 (1/12) | 0.000 (0/12) | 0.083 (1/12) | 0.000 (0/12) |
| false abstain, open search (lower is better) | 0.000 (0/12) | 0.083 (1/12) | 0.000 (0/12) | 0.000 (0/12) |
| false abstain with the full evidence already shown, open search (post hoc; lower is better) | 0.000 (0/12) | 0.000 (0/12) | 0.000 (0/12) | 0.000 (0/12) |
| answer correct, citation not required, open search (post hoc) | 0.333 (4/12) | 0.000 (0/12) | 0.083 (1/12) | 0.000 (0/12) |
| correct abstain, support masked | 0.000 (0/10) | 0.300 (3/10) | 0.600 (6/10) | 0.000 (0/10) |
| answers when the documents cannot support it (lower is better) | 0.900 (9/10) | 0.400 (4/10) | 0.400 (4/10) | 0.300 (3/10) |
| RGB integration accuracy, noise 0 | 0.383 (23/60) | 0.017 (1/60) | 0.000 (0/60) | 0.017 (1/60) |
| RGB integration accuracy, noise 0.4 | 0.300 (18/60) | 0.000 (0/60) | 0.000 (0/60) | 0.000 (0/60) |
| invalid or rule-breaking turns (lower is better) | 0.169 (24/142) | 0.028 (4/142) | 0.000 (0/142) | 0.070 (10/142) |
| mean searches per episode | 0.89 | 0.51 | 0.37 | 0.6 |

| ends well, paired | before | after | difference | 95% CI | episodes / questions |
| --- | ---: | ---: | ---: | --- | ---: |
| main vs base | 0.070 | 0.028 | -0.042 | [-0.097, +0.008] | 142 / 60 |
| main vs candA | 0.049 | 0.028 | -0.021 | [-0.056, +0.014] | 142 / 60 |
| main vs control | 0.007 | 0.028 | +0.021 | [-0.007, +0.051] | 142 / 60 |

| AuditKIT (all episodes) | Tiny Aya base | main | candA | control |
| --- | ---: | ---: | ---: | ---: |
| evidence_set_recall | 0.889 | 0.889 | 0.889 | 0.889 |
| citation_support | 0.527 | 0.962 | 0.992 | 0.977 |
| abstention | 0.930 | 0.923 | 0.965 | 0.930 |
| tool_call_validity | 1.000 | 1.000 | 1.000 | 1.000 |
| redundant_tool_calls (lower is better) | 0.000 | 0.028 | 0.000 | 0.048 |
| agent_loop_detection (1.0 = loop-free) | 1.000 | 0.972 | 1.000 | 0.944 |

Closed book (base model, question only): 0.100 of 60 answerable questions.

