# Third round, fresh test set (lenient reading)

Plan section 24. Opened once; no choice depends on these numbers.

| set (episodes) | main | candA | candA minus main |
| --- | ---: | ---: | --- |
| HotpotQA (fresh) (88) | 0.602 (53/88) | 0.727 (64/88) | +0.125 [+0.033, +0.224] |
| 2WikiMultihopQA (fresh) (86) | 0.651 (56/86) | 0.640 (55/86) | -0.012 [-0.106, +0.079] |
| MuSiQue 2-hop (fresh) (67) | 0.478 (32/67) | 0.627 (42/67) | +0.149 [+0.031, +0.265] |
| SQuAD 2.0 (fresh questions) (166) | 0.512 (85/166) | 0.614 (102/166) | +0.102 [+0.030, +0.176] |
| RGB (fresh) (266) | 0.628 (167/266) | 0.658 (175/266) | +0.030 [-0.015, +0.076] |
| RGB integration (fresh) (66) | 0.061 (4/66) | 0.030 (2/66) | -0.030 [-0.071, +0.000] |
| **macro average of 6 sets** | 0.489 | 0.549 | **+0.061 [+0.028, +0.093]** |

## Safety and refusals

| pooled over all sets | main | candA | difference [95% CI] | episodes |
| --- | ---: | ---: | --- | ---: |
| unsupported answers, hidden-evidence episodes (lower is better) | 0.203 | 0.233 | +0.031 [-0.003, +0.064] | 326 |
| false abstentions, open search (lower is better) | 0.406 | 0.368 | -0.038 [-0.096, +0.025] | 239 |

Plan section 24 criteria, applied post hoc (candA was not a pre-registered candidate; its selection status is unchanged): primary difference with its 95% interval above zero: **met**; pooled unsupported-answer rate at most 0.05 above the reference's: **met**. Would be recommended as the default model: **yes**.

## HotpotQA (fresh)

| metric | main | candA |
| --- | ---: | ---: |
| ends well (all scored episodes) | 0.602 (53/88) | 0.727 (64/88) |
| supported correct, open search (answerable) | 0.417 (20/48) | 0.604 (29/48) |
| false abstain, open search (lower is better) | 0.312 (15/48) | 0.167 (8/48) |
| false abstain with the full evidence already shown, open search (post hoc; lower is better) | 0.208 (10/48) | 0.104 (5/48) |
| answer correct, citation not required, open search (post hoc) | 0.417 (20/48) | 0.604 (29/48) |
| correct abstain, support masked | 0.825 (33/40) | 0.875 (35/40) |
| answers when the documents cannot support it (lower is better) | 0.150 (6/40) | 0.050 (2/40) |
| invalid or rule-breaking turns (lower is better) | 0.023 (2/88) | 0.034 (3/88) |
| mean searches per episode | 2.28 | 1.97 |

| AuditKIT (all episodes) | main | candA |
| --- | ---: | ---: |
| evidence_set_recall | 0.812 | 0.812 |
| citation_support | 0.667 | 0.833 |
| abstention | 0.750 | 0.852 |
| tool_call_validity | 1.000 | 1.000 |
| redundant_tool_calls (lower is better) | 0.008 | 0.004 |
| agent_loop_detection (1.0 = loop-free) | 0.977 | 0.989 |

## 2WikiMultihopQA (fresh)

| metric | main | candA |
| --- | ---: | ---: |
| ends well (all scored episodes) | 0.651 (56/86) | 0.640 (55/86) |
| supported correct, open search (answerable) | 0.444 (20/45) | 0.400 (18/45) |
| false abstain, open search (lower is better) | 0.178 (8/45) | 0.267 (12/45) |
| false abstain with the full evidence already shown, open search (post hoc; lower is better) | 0.022 (1/45) | 0.022 (1/45) |
| answer correct, citation not required, open search (post hoc) | 0.444 (20/45) | 0.400 (18/45) |
| correct abstain, support masked | 0.878 (36/41) | 0.902 (37/41) |
| answers when the documents cannot support it (lower is better) | 0.024 (1/41) | 0.049 (2/41) |
| invalid or rule-breaking turns (lower is better) | 0.047 (4/86) | 0.023 (2/86) |
| mean searches per episode | 2.41 | 2.2 |

| AuditKIT (all episodes) | main | candA |
| --- | ---: | ---: |
| evidence_set_recall | 0.556 | 0.489 |
| citation_support | 0.822 | 0.733 |
| abstention | 0.849 | 0.814 |
| tool_call_validity | 1.000 | 1.000 |
| redundant_tool_calls (lower is better) | 0.017 | 0.007 |
| agent_loop_detection (1.0 = loop-free) | 0.953 | 0.977 |

## MuSiQue 2-hop (fresh)

| metric | main | candA |
| --- | ---: | ---: |
| ends well (all scored episodes) | 0.478 (32/67) | 0.627 (42/67) |
| supported correct, open search (answerable) | 0.176 (6/34) | 0.353 (12/34) |
| false abstain, open search (lower is better) | 0.588 (20/34) | 0.471 (16/34) |
| false abstain with the full evidence already shown, open search (post hoc; lower is better) | 0.176 (6/34) | 0.118 (4/34) |
| answer correct, citation not required, open search (post hoc) | 0.206 (7/34) | 0.353 (12/34) |
| correct abstain, support masked | 0.788 (26/33) | 0.909 (30/33) |
| answers when the documents cannot support it (lower is better) | 0.000 (0/33) | 0.000 (0/33) |
| invalid or rule-breaking turns (lower is better) | 0.119 (8/67) | 0.060 (4/67) |
| mean searches per episode | 2.48 | 2.18 |

| AuditKIT (all episodes) | main | candA |
| --- | ---: | ---: |
| evidence_set_recall | 0.500 | 0.471 |
| citation_support | 0.382 | 0.500 |
| abstention | 0.597 | 0.716 |
| tool_call_validity | 1.000 | 1.000 |
| redundant_tool_calls (lower is better) | 0.040 | 0.015 |
| agent_loop_detection (1.0 = loop-free) | 0.881 | 0.955 |

## SQuAD 2.0 (fresh questions)

| metric | main | candA |
| --- | ---: | ---: |
| ends well (all scored episodes) | 0.512 (85/166) | 0.614 (102/166) |
| supported correct, open search (answerable) | 0.070 (4/57) | 0.105 (6/57) |
| false abstain, open search (lower is better) | 0.737 (42/57) | 0.772 (44/57) |
| false abstain with the full evidence already shown, open search (post hoc; lower is better) | 0.737 (42/57) | 0.772 (44/57) |
| answer correct, citation not required, open search (post hoc) | 0.070 (4/57) | 0.123 (7/57) |
| correct abstain, support masked | 0.714 (35/49) | 0.898 (44/49) |
| correct abstain, natural unanswerable (SQuAD 2.0) | 0.767 (46/60) | 0.867 (52/60) |
| answers when the documents cannot support it (lower is better) | 0.055 (6/109) | 0.055 (6/109) |
| invalid or rule-breaking turns (lower is better) | 0.181 (30/166) | 0.060 (10/166) |
| mean searches per episode | 2.49 | 2.16 |

| AuditKIT (all episodes) | main | candA |
| --- | ---: | ---: |
| evidence_set_recall | 1.000 | 1.000 |
| citation_support | 0.123 | 0.175 |
| abstention | 0.578 | 0.657 |
| tool_call_validity | 1.000 | 1.000 |
| redundant_tool_calls (lower is better) | 0.060 | 0.019 |
| agent_loop_detection (1.0 = loop-free) | 0.819 | 0.946 |

## RGB (fresh)

| metric | main | candA |
| --- | ---: | ---: |
| ends well (all scored episodes) | 0.628 (167/266) | 0.658 (175/266) |
| supported correct, open search (answerable) | 0.571 (28/49) | 0.776 (38/49) |
| false abstain, open search (lower is better) | 0.224 (11/49) | 0.143 (7/49) |
| false abstain with the full evidence already shown, open search (post hoc; lower is better) | 0.224 (11/49) | 0.122 (6/49) |
| answer correct, citation not required, open search (post hoc) | 0.612 (30/49) | 0.816 (40/49) |
| correct abstain, support masked | 0.649 (24/37) | 0.703 (26/37) |
| answers when the documents cannot support it (lower is better) | 0.526 (51/97) | 0.639 (62/97) |
| RGB accuracy, noise 0 (answer correct, citation not required) | 0.967 (58/60) | 0.967 (58/60) |
| RGB accuracy, noise 0.8 | 0.783 (47/60) | 0.800 (48/60) |
| RGB rejection rate, noise 1 (abstains) | 0.233 (14/60) | 0.117 (7/60) |
| invalid or rule-breaking turns (lower is better) | 0.056 (15/266) | 0.011 (3/266) |
| mean searches per episode | 1.0 | 0.77 |

| AuditKIT (all episodes) | main | candA |
| --- | ---: | ---: |
| evidence_set_recall | 0.938 | 1.000 |
| citation_support | 0.893 | 0.953 |
| abstention | 0.737 | 0.733 |
| tool_call_validity | 1.000 | 1.000 |
| redundant_tool_calls (lower is better) | 0.029 | 0.009 |
| agent_loop_detection (1.0 = loop-free) | 0.951 | 0.985 |

## RGB integration (fresh)

| metric | main | candA |
| --- | ---: | ---: |
| ends well (all scored episodes) | 0.061 (4/66) | 0.030 (2/66) |
| supported correct, open search (answerable) | 0.000 (0/6) | 0.000 (0/6) |
| false abstain, open search (lower is better) | 0.167 (1/6) | 0.167 (1/6) |
| false abstain with the full evidence already shown, open search (post hoc; lower is better) | 0.167 (1/6) | 0.167 (1/6) |
| answer correct, citation not required, open search (post hoc) | 0.000 (0/6) | 0.000 (0/6) |
| correct abstain, support masked | 0.667 (4/6) | 0.333 (2/6) |
| answers when the documents cannot support it (lower is better) | 0.333 (2/6) | 0.667 (4/6) |
| RGB integration accuracy, noise 0 | 0.000 (0/27) | 0.000 (0/27) |
| RGB integration accuracy, noise 0.4 | 0.000 (0/27) | 0.000 (0/27) |
| invalid or rule-breaking turns (lower is better) | 0.000 (0/66) | 0.000 (0/66) |
| mean searches per episode | 0.65 | 0.5 |

| AuditKIT (all episodes) | main | candA |
| --- | ---: | ---: |
| evidence_set_recall | n/a | n/a |
| citation_support | 0.967 | 0.967 |
| abstention | 0.939 | 0.909 |
| tool_call_validity | 1.000 | 1.000 |
| redundant_tool_calls (lower is better) | 0.000 | 0.000 |
| agent_loop_detection (1.0 = loop-free) | 1.000 | 1.000 |
