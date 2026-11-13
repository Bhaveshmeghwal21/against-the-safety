# Third round, fresh test set (lenient reading)

**13 November 2026 correction:** This table preserves the original scorer and selection record. Evidence alternatives, binary answers, fixed eligibility and source offsets are corrected in the [dated remediation report](2026-11-13-evaluation-remediation.md). Later DPO prompts exposed held-out HotpotQA evidence. Annotation-based unsupported flags do not establish semantic hallucination; main remains submitted.


Plan section 24. Opened once; no choice depends on these numbers.

| set (episodes) | main | candD | candA | candD minus main |
| --- | ---: | ---: | ---: | --- |
| HotpotQA (fresh) (88) | 0.602 (53/88) | 0.716 (63/88) | 0.727 (64/88) | +0.114 [+0.012, +0.209] |
| 2WikiMultihopQA (fresh) (86) | 0.651 (56/86) | 0.570 (49/86) | 0.640 (55/86) | -0.081 [-0.174, +0.000] |
| MuSiQue 2-hop (fresh) (67) | 0.478 (32/67) | 0.582 (39/67) | 0.627 (42/67) | +0.104 [+0.000, +0.200] |
| SQuAD 2.0 (fresh questions) (166) | 0.512 (85/166) | 0.645 (107/166) | 0.614 (102/166) | +0.133 [+0.031, +0.229] |
| RGB (fresh) (266) | 0.628 (167/266) | 0.665 (177/266) | 0.658 (175/266) | +0.038 [-0.007, +0.086] |
| RGB integration (fresh) (66) | 0.061 (4/66) | 0.152 (10/66) | 0.030 (2/66) | +0.091 [+0.000, +0.197] |
| **macro average of 6 sets** | 0.489 | 0.555 | 0.549 | **+0.066 [+0.031, +0.103]** |

## Safety and refusals

| pooled over all sets | main | candD | difference [95% CI] | episodes |
| --- | ---: | ---: | --- | ---: |
| unsupported answers, hidden-evidence episodes (lower is better) | 0.203 | 0.322 | +0.120 [+0.076, +0.163] | 326 |
| false abstentions, open search (lower is better) | 0.406 | 0.264 | -0.142 [-0.209, -0.075] | 239 |

Pre-registered criteria: primary difference with its 95% interval above zero: **met**; pooled unsupported-answer rate at most 0.05 above the reference's: **not met**. Recommended as the default model: **no**.

## HotpotQA (fresh)

| metric | main | candD | candA |
| --- | ---: | ---: | ---: |
| ends well (all scored episodes) | 0.602 (53/88) | 0.716 (63/88) | 0.727 (64/88) |
| supported correct, open search (answerable) | 0.417 (20/48) | 0.583 (28/48) | 0.604 (29/48) |
| false abstain, open search (lower is better) | 0.312 (15/48) | 0.188 (9/48) | 0.167 (8/48) |
| false abstain with the full evidence already shown, open search (post hoc; lower is better) | 0.208 (10/48) | 0.125 (6/48) | 0.104 (5/48) |
| answer correct, citation not required, open search (post hoc) | 0.417 (20/48) | 0.583 (28/48) | 0.604 (29/48) |
| correct abstain, support masked | 0.825 (33/40) | 0.875 (35/40) | 0.875 (35/40) |
| answers when the documents cannot support it (lower is better) | 0.150 (6/40) | 0.075 (3/40) | 0.050 (2/40) |
| invalid or rule-breaking turns (lower is better) | 0.023 (2/88) | 0.023 (2/88) | 0.034 (3/88) |
| mean searches per episode | 2.28 | 1.76 | 1.97 |

| AuditKIT (all episodes) | main | candD | candA |
| --- | ---: | ---: | ---: |
| evidence_set_recall | 0.812 | 0.688 | 0.812 |
| citation_support | 0.667 | 0.812 | 0.833 |
| abstention | 0.750 | 0.841 | 0.852 |
| tool_call_validity | 1.000 | 1.000 | 1.000 |
| redundant_tool_calls (lower is better) | 0.008 | 0.008 | 0.004 |
| agent_loop_detection (1.0 = loop-free) | 0.977 | 0.977 | 0.989 |

## 2WikiMultihopQA (fresh)

| metric | main | candD | candA |
| --- | ---: | ---: | ---: |
| ends well (all scored episodes) | 0.651 (56/86) | 0.570 (49/86) | 0.640 (55/86) |
| supported correct, open search (answerable) | 0.444 (20/45) | 0.289 (13/45) | 0.400 (18/45) |
| false abstain, open search (lower is better) | 0.178 (8/45) | 0.289 (13/45) | 0.267 (12/45) |
| false abstain with the full evidence already shown, open search (post hoc; lower is better) | 0.022 (1/45) | 0.044 (2/45) | 0.022 (1/45) |
| answer correct, citation not required, open search (post hoc) | 0.444 (20/45) | 0.289 (13/45) | 0.400 (18/45) |
| correct abstain, support masked | 0.878 (36/41) | 0.878 (36/41) | 0.902 (37/41) |
| answers when the documents cannot support it (lower is better) | 0.024 (1/41) | 0.098 (4/41) | 0.049 (2/41) |
| invalid or rule-breaking turns (lower is better) | 0.047 (4/86) | 0.012 (1/86) | 0.023 (2/86) |
| mean searches per episode | 2.41 | 2.05 | 2.2 |

| AuditKIT (all episodes) | main | candD | candA |
| --- | ---: | ---: | ---: |
| evidence_set_recall | 0.556 | 0.400 | 0.489 |
| citation_support | 0.822 | 0.711 | 0.733 |
| abstention | 0.849 | 0.791 | 0.814 |
| tool_call_validity | 1.000 | 1.000 | 1.000 |
| redundant_tool_calls (lower is better) | 0.017 | 0.004 | 0.007 |
| agent_loop_detection (1.0 = loop-free) | 0.953 | 0.988 | 0.977 |

## MuSiQue 2-hop (fresh)

| metric | main | candD | candA |
| --- | ---: | ---: | ---: |
| ends well (all scored episodes) | 0.478 (32/67) | 0.582 (39/67) | 0.627 (42/67) |
| supported correct, open search (answerable) | 0.176 (6/34) | 0.235 (8/34) | 0.353 (12/34) |
| false abstain, open search (lower is better) | 0.588 (20/34) | 0.500 (17/34) | 0.471 (16/34) |
| false abstain with the full evidence already shown, open search (post hoc; lower is better) | 0.176 (6/34) | 0.147 (5/34) | 0.118 (4/34) |
| answer correct, citation not required, open search (post hoc) | 0.206 (7/34) | 0.235 (8/34) | 0.353 (12/34) |
| correct abstain, support masked | 0.788 (26/33) | 0.939 (31/33) | 0.909 (30/33) |
| answers when the documents cannot support it (lower is better) | 0.000 (0/33) | 0.030 (1/33) | 0.000 (0/33) |
| invalid or rule-breaking turns (lower is better) | 0.119 (8/67) | 0.015 (1/67) | 0.060 (4/67) |
| mean searches per episode | 2.48 | 1.96 | 2.18 |

| AuditKIT (all episodes) | main | candD | candA |
| --- | ---: | ---: | ---: |
| evidence_set_recall | 0.500 | 0.353 | 0.471 |
| citation_support | 0.382 | 0.500 | 0.500 |
| abstention | 0.597 | 0.716 | 0.716 |
| tool_call_validity | 1.000 | 1.000 | 1.000 |
| redundant_tool_calls (lower is better) | 0.040 | 0.005 | 0.015 |
| agent_loop_detection (1.0 = loop-free) | 0.881 | 0.985 | 0.955 |

## SQuAD 2.0 (fresh questions)

| metric | main | candD | candA |
| --- | ---: | ---: | ---: |
| ends well (all scored episodes) | 0.512 (85/166) | 0.645 (107/166) | 0.614 (102/166) |
| supported correct, open search (answerable) | 0.070 (4/57) | 0.561 (32/57) | 0.105 (6/57) |
| false abstain, open search (lower is better) | 0.737 (42/57) | 0.316 (18/57) | 0.772 (44/57) |
| false abstain with the full evidence already shown, open search (post hoc; lower is better) | 0.737 (42/57) | 0.316 (18/57) | 0.772 (44/57) |
| answer correct, citation not required, open search (post hoc) | 0.070 (4/57) | 0.561 (32/57) | 0.123 (7/57) |
| correct abstain, support masked | 0.714 (35/49) | 0.776 (38/49) | 0.898 (44/49) |
| correct abstain, natural unanswerable (SQuAD 2.0) | 0.767 (46/60) | 0.617 (37/60) | 0.867 (52/60) |
| answers when the documents cannot support it (lower is better) | 0.055 (6/109) | 0.284 (31/109) | 0.055 (6/109) |
| invalid or rule-breaking turns (lower is better) | 0.181 (30/166) | 0.018 (3/166) | 0.060 (10/166) |
| mean searches per episode | 2.49 | 1.61 | 2.16 |

| AuditKIT (all episodes) | main | candD | candA |
| --- | ---: | ---: | ---: |
| evidence_set_recall | 1.000 | 1.000 | 1.000 |
| citation_support | 0.123 | 0.684 | 0.175 |
| abstention | 0.578 | 0.687 | 0.657 |
| tool_call_validity | 1.000 | 1.000 | 1.000 |
| redundant_tool_calls (lower is better) | 0.060 | 0.008 | 0.019 |
| agent_loop_detection (1.0 = loop-free) | 0.819 | 0.982 | 0.946 |

## RGB (fresh)

| metric | main | candD | candA |
| --- | ---: | ---: | ---: |
| ends well (all scored episodes) | 0.628 (167/266) | 0.665 (177/266) | 0.658 (175/266) |
| supported correct, open search (answerable) | 0.571 (28/49) | 0.776 (38/49) | 0.776 (38/49) |
| false abstain, open search (lower is better) | 0.224 (11/49) | 0.082 (4/49) | 0.143 (7/49) |
| false abstain with the full evidence already shown, open search (post hoc; lower is better) | 0.224 (11/49) | 0.082 (4/49) | 0.122 (6/49) |
| answer correct, citation not required, open search (post hoc) | 0.612 (30/49) | 0.816 (40/49) | 0.816 (40/49) |
| correct abstain, support masked | 0.649 (24/37) | 0.730 (27/37) | 0.703 (26/37) |
| answers when the documents cannot support it (lower is better) | 0.526 (51/97) | 0.660 (64/97) | 0.639 (62/97) |
| RGB accuracy, noise 0 (answer correct, citation not required) | 0.967 (58/60) | 0.983 (59/60) | 0.967 (58/60) |
| RGB accuracy, noise 0.8 | 0.783 (47/60) | 0.800 (48/60) | 0.800 (48/60) |
| RGB rejection rate, noise 1 (abstains) | 0.233 (14/60) | 0.100 (6/60) | 0.117 (7/60) |
| invalid or rule-breaking turns (lower is better) | 0.056 (15/266) | 0.000 (0/266) | 0.011 (3/266) |
| mean searches per episode | 1.0 | 0.55 | 0.77 |

| AuditKIT (all episodes) | main | candD | candA |
| --- | ---: | ---: | ---: |
| evidence_set_recall | 0.938 | 1.000 | 1.000 |
| citation_support | 0.893 | 0.976 | 0.953 |
| abstention | 0.737 | 0.744 | 0.733 |
| tool_call_validity | 1.000 | 1.000 | 1.000 |
| redundant_tool_calls (lower is better) | 0.029 | 0.000 | 0.009 |
| agent_loop_detection (1.0 = loop-free) | 0.951 | 1.000 | 0.985 |

## RGB integration (fresh)

| metric | main | candD | candA |
| --- | ---: | ---: | ---: |
| ends well (all scored episodes) | 0.061 (4/66) | 0.152 (10/66) | 0.030 (2/66) |
| supported correct, open search (answerable) | 0.000 (0/6) | 0.000 (0/6) | 0.000 (0/6) |
| false abstain, open search (lower is better) | 0.167 (1/6) | 0.333 (2/6) | 0.167 (1/6) |
| false abstain with the full evidence already shown, open search (post hoc; lower is better) | 0.167 (1/6) | 0.333 (2/6) | 0.167 (1/6) |
| answer correct, citation not required, open search (post hoc) | 0.000 (0/6) | 0.000 (0/6) | 0.000 (0/6) |
| correct abstain, support masked | 0.667 (4/6) | 0.667 (4/6) | 0.333 (2/6) |
| answers when the documents cannot support it (lower is better) | 0.333 (2/6) | 0.333 (2/6) | 0.667 (4/6) |
| RGB integration accuracy, noise 0 | 0.000 (0/27) | 0.148 (4/27) | 0.000 (0/27) |
| RGB integration accuracy, noise 0.4 | 0.000 (0/27) | 0.074 (2/27) | 0.000 (0/27) |
| invalid or rule-breaking turns (lower is better) | 0.000 (0/66) | 0.000 (0/66) | 0.000 (0/66) |
| mean searches per episode | 0.65 | 0.32 | 0.5 |

| AuditKIT (all episodes) | main | candD | candA |
| --- | ---: | ---: | ---: |
| evidence_set_recall | n/a | n/a | n/a |
| citation_support | 0.967 | 0.967 | 0.967 |
| abstention | 0.939 | 0.939 | 0.909 |
| tool_call_validity | 1.000 | 1.000 | 1.000 |
| redundant_tool_calls (lower is better) | 0.000 | 0.000 | 0.000 |
| agent_loop_detection (1.0 = loop-free) | 1.000 | 1.000 | 1.000 |
