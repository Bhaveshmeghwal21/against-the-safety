# Scorecard (test2): main against base

Each number is reported separately with its denominator. There is no composite score.

## when_to_retrieve

| metric | base | main | control | candA |
| --- | ---: | ---: | ---: | ---: |
| missed_search_rate (no evidence supplied; lower is better) | 0.947 (160/169) | 0.000 (0/169) | 0.000 (0/169) | 0.000 (0/169) |
| unnecessary_search_rate (sufficient evidence supplied; lower is better) | 0.067 (4/60) | 0.167 (10/60) | 0.050 (3/60) | 0.150 (9/60) |
| search_when_evidence_partial (higher is better) | 0.267 (8/30) | 0.667 (20/30) | 0.833 (25/30) | 0.667 (20/30) |

## what_to_retrieve

| metric | base | main | control | candA |
| --- | ---: | ---: | ---: | ---: |
| full_support_found_by_own_queries (open questions) | 0.012 (1/83) | 0.711 (59/83) | 0.747 (62/83) | 0.675 (56/83) |
| second_hop_recovery (partial or weak first result) | 0.089 (5/56) | 0.500 (28/56) | 0.643 (36/56) | 0.571 (32/56) |
| auditkit_evidence_set_recall | 0.332 | 0.739 | 0.794 | 0.744 |

## grounding

| metric | base | main | control | candA |
| --- | ---: | ---: | ---: | ---: |
| supported_correct (answerable episodes) | 0.000 (0/199) | 0.462 (92/199) | 0.598 (119/199) | 0.528 (105/199) |
| supported_correct (open questions) | 0.000 (0/83) | 0.373 (31/83) | 0.482 (40/83) | 0.458 (38/83) |
| answer_correct_ignoring_citations (answerable; F1 >= 0.5) | 0.000 (0/199) | 0.472 (94/199) | 0.598 (119/199) | 0.528 (105/199) |
| supported_exact_match (answerable; exact match instead of F1 >= 0.5) | 0.000 (0/199) | 0.372 (74/199) | 0.497 (99/199) | 0.437 (87/199) |
| cites_a_gold_passage_when_answering (answerable) | 0.400 (2/5) | 0.966 (144/149) | 0.983 (176/179) | 0.994 (168/169) |
| answer_f1_when_answered | 0.000 | 0.603 | 0.615 | 0.589 |
| auditkit_citation_support | 0.025 | 0.749 | 0.899 | 0.849 |

## ignores_distractors

| metric | base | main | control | candA |
| --- | ---: | ---: | ---: | ---: |
| supported_correct with distractor passage | 0.000 (0/30) | 0.567 (17/30) | 0.767 (23/30) | 0.567 (17/30) |
| supported_correct, same evidence without distractor | 0.000 (0/30) | 0.633 (19/30) | 0.767 (23/30) | 0.733 (22/30) |

## abstention (documents-only requirement)

| metric | base | main | control | candA |
| --- | ---: | ---: | ---: | ---: |
| correct_abstain (unanswerable episodes) | 0.000 (0/74) | 0.905 (67/74) | 0.000 (0/74) | 0.878 (65/74) |
| unsupported_answer_rate (unanswerable; lower is better) | 0.014 (1/74) | 0.041 (3/74) | 0.189 (14/74) | 0.081 (6/74) |
| false_abstain (answerable; lower is better) | 0.000 (0/199) | 0.221 (44/199) | 0.000 (0/199) | 0.146 (29/199) |
| auditkit_abstention | 0.729 | 0.813 | 0.729 | 0.861 |

## tool_call_reliability (AuditKIT, Track 3)

| metric | base | main | control | candA |
| --- | ---: | ---: | ---: | ---: |
| tool_call_validity | 0.933 | 1.000 | 1.000 | 1.000 |
| redundant_tool_calls (lower is better) | 0.000 | 0.018 | 0.118 | 0.005 |
| agent_loop_detection (1.0 = loop-free) | 1.000 | 0.963 | 0.729 | 0.985 |
| invalid_turn_rate (unparseable or rule-breaking; lower is better) | 0.978 (267/273) | 0.037 (10/273) | 0.293 (80/273) | 0.015 (4/273) |

## efficiency

| metric | base | main | control | candA |
| --- | ---: | ---: | ---: | ---: |
| mean_searches_per_episode | 0.100 | 1.630 | 1.400 | 1.450 |
| mean_input_tokens_per_episode | 1020.800 | 3200.400 | 2874.800 | 2945.900 |
| mean_seconds_per_episode | 1.695 | 3.999 | 3.596 | 3.777 |

## Episodes

| system | episodes |
| --- | ---: |
| base | 273 |
| main | 273 |
| control | 273 |
| candA | 273 |

## Paired difference: main vs base

95% interval from resampling whole questions; both systems ran the same episodes.

| metric | before | after | difference | 95% CI | episodes / questions | only-after / only-before good |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| end-to-end good (headline: supported answer when answerable, abstain when not) | 0.000 | 0.601 | +0.601 | [+0.536, +0.667] | 143 / 83 | 86 / 0 |
| end-to-end good, exact match instead of F1 >= 0.5 (headline) | 0.000 | 0.559 | +0.559 | [+0.496, +0.628] | 143 / 83 | 80 / 0 |
| supported_correct (answerable episodes) | 0.000 | 0.462 | +0.462 | [+0.362, +0.561] | 199 / 83 | 92 / 0 |
| answer correct, ignoring citations (answerable; F1 >= 0.5) | 0.000 | 0.472 | +0.472 | [+0.369, +0.575] | 199 / 83 | 94 / 0 |
| correct_abstain (unanswerable episodes) | 0.000 | 0.905 | +0.905 | [+0.838, +0.962] | 74 / 60 | 67 / 0 |
| unsupported answer (unanswerable; lower is better) | 0.014 | 0.041 | +0.027 | [+0.000, +0.068] | 74 / 60 | 2 / 0 |
| supported_correct with distractor passage | 0.000 | 0.567 | +0.567 | [+0.400, +0.733] | 30 / 30 | 17 / 0 |

## Paired difference: main vs control

95% interval from resampling whole questions; both systems ran the same episodes.

| metric | before | after | difference | 95% CI | episodes / questions | only-after / only-before good |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| end-to-end good (headline: supported answer when answerable, abstain when not) | 0.280 | 0.601 | +0.322 | [+0.246, +0.392] | 143 / 83 | 61 / 15 |
| end-to-end good, exact match instead of F1 >= 0.5 (headline) | 0.238 | 0.559 | +0.322 | [+0.250, +0.388] | 143 / 83 | 60 / 14 |
| supported_correct (answerable episodes) | 0.598 | 0.462 | -0.136 | [-0.207, -0.066] | 199 / 83 | 10 / 37 |
| answer correct, ignoring citations (answerable; F1 >= 0.5) | 0.598 | 0.472 | -0.126 | [-0.198, -0.054] | 199 / 83 | 10 / 35 |
| correct_abstain (unanswerable episodes) | 0.000 | 0.905 | +0.905 | [+0.838, +0.962] | 74 / 60 | 67 / 0 |
| unsupported answer (unanswerable; lower is better) | 0.189 | 0.041 | -0.149 | [-0.222, -0.074] | 74 / 60 | 0 / 11 |
| supported_correct with distractor passage | 0.767 | 0.567 | -0.200 | [-0.367, -0.067] | 30 / 30 | 0 / 6 |

## Paired difference: main vs candA

95% interval from resampling whole questions; both systems ran the same episodes.

| metric | before | after | difference | 95% CI | episodes / questions | only-after / only-before good |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| end-to-end good (headline: supported answer when answerable, abstain when not) | 0.657 | 0.601 | -0.056 | [-0.122, +0.007] | 143 / 83 | 10 / 18 |
| end-to-end good, exact match instead of F1 >= 0.5 (headline) | 0.615 | 0.559 | -0.056 | [-0.117, +0.000] | 143 / 83 | 8 / 16 |
| supported_correct (answerable episodes) | 0.528 | 0.462 | -0.065 | [-0.142, +0.015] | 199 / 83 | 15 / 28 |
| answer correct, ignoring citations (answerable; F1 >= 0.5) | 0.528 | 0.472 | -0.055 | [-0.134, +0.026] | 199 / 83 | 16 / 27 |
| correct_abstain (unanswerable episodes) | 0.878 | 0.905 | +0.027 | [-0.056, +0.110] | 74 / 60 | 7 / 5 |
| unsupported answer (unanswerable; lower is better) | 0.081 | 0.041 | -0.041 | [-0.100, +0.014] | 74 / 60 | 1 / 4 |
| supported_correct with distractor passage | 0.567 | 0.567 | +0.000 | [-0.200, +0.200] | 30 / 30 | 4 / 4 |
