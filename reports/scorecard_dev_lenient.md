# Scorecard (dev): main against base

Each number is reported separately with its denominator. There is no composite score.

## when_to_retrieve

| metric | base | main | control |
| --- | ---: | ---: | ---: |
| missed_search_rate (no evidence supplied; lower is better) | 0.043 (8/188) | 0.000 (0/188) | 0.000 (0/188) |
| unnecessary_search_rate (sufficient evidence supplied; lower is better) | 0.983 (59/60) | 0.200 (12/60) | 0.067 (4/60) |
| search_when_evidence_partial (higher is better) | 0.967 (29/30) | 0.667 (20/30) | 0.633 (19/30) |

## what_to_retrieve

| metric | base | main | control |
| --- | ---: | ---: | ---: |
| full_support_found_by_own_queries (open questions) | 0.267 (23/86) | 0.651 (56/86) | 0.709 (61/86) |
| second_hop_recovery (partial or weak first result) | 0.186 (11/59) | 0.339 (20/59) | 0.424 (25/59) |
| auditkit_evidence_set_recall | 0.459 | 0.663 | 0.712 |

## grounding

| metric | base | main | control |
| --- | ---: | ---: | ---: |
| supported_correct (answerable episodes) | 0.044 (9/205) | 0.468 (96/205) | 0.532 (109/205) |
| supported_correct (open questions) | 0.023 (2/86) | 0.430 (37/86) | 0.512 (44/86) |
| answer_correct_ignoring_citations (answerable; F1 >= 0.5) | 0.098 (20/205) | 0.468 (96/205) | 0.532 (109/205) |
| cites_a_gold_passage_when_answering (answerable) | 0.203 (36/177) | 0.994 (164/165) | 0.990 (189/191) |
| answer_f1_when_answered | 0.132 | 0.537 | 0.511 |
| auditkit_citation_support | 0.380 | 0.805 | 0.932 |

## ignores_distractors

| metric | base | main | control |
| --- | ---: | ---: | ---: |
| supported_correct with distractor passage | 0.067 (2/30) | 0.600 (18/30) | 0.667 (20/30) |
| supported_correct, same evidence without distractor | 0.100 (3/30) | 0.633 (19/30) | 0.700 (21/30) |

## abstention (documents-only requirement)

| metric | base | main | control |
| --- | ---: | ---: | ---: |
| correct_abstain (unanswerable episodes) | 0.000 (0/88) | 0.875 (77/88) | 0.000 (0/88) |
| unsupported_answer_rate (unanswerable; lower is better) | 0.898 (79/88) | 0.114 (10/88) | 0.284 (25/88) |
| false_abstain (answerable; lower is better) | 0.000 (0/205) | 0.176 (36/205) | 0.000 (0/205) |
| auditkit_abstention | 0.700 | 0.840 | 0.700 |

## tool_call_reliability (AuditKIT, Track 3)

| metric | base | main | control |
| --- | ---: | ---: | ---: |
| tool_call_validity | 1.000 | 1.000 | 1.000 |
| redundant_tool_calls (lower is better) | 0.027 | 0.008 | 0.101 |
| agent_loop_detection (1.0 = loop-free) | 0.949 | 0.983 | 0.761 |
| invalid_turn_rate (unparseable or rule-breaking; lower is better) | 0.126 (37/293) | 0.017 (5/293) | 0.263 (77/293) |

## efficiency

| metric | base | main | control |
| --- | ---: | ---: | ---: |
| mean_searches_per_episode | 0.970 | 1.620 | 1.440 |
| mean_input_tokens_per_episode | 2081.200 | 3128.200 | 2889.800 |
| mean_seconds_per_episode | 2.854 | 4.244 | 3.808 |

## Episodes

| system | episodes |
| --- | ---: |
| base | 293 |
| main | 293 |
| control | 293 |

## Paired difference: main vs base

95% interval from resampling whole questions; both systems ran the same episodes.

| metric | before | after | difference | 95% CI | episodes / questions | only-after / only-before good |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| end-to-end good (headline: supported answer when answerable, abstain when not) | 0.013 | 0.673 | +0.660 | [+0.594, +0.725] | 159 / 86 | 106 / 1 |
| supported_correct (answerable episodes) | 0.044 | 0.468 | +0.424 | [+0.324, +0.520] | 205 / 86 | 89 / 2 |
| answer correct, ignoring citations (answerable; F1 >= 0.5) | 0.098 | 0.468 | +0.371 | [+0.267, +0.472] | 205 / 86 | 82 / 6 |
| correct_abstain (unanswerable episodes) | 0.000 | 0.875 | +0.875 | [+0.805, +0.942] | 88 / 73 | 77 / 0 |
| unsupported answer (unanswerable; lower is better) | 0.898 | 0.114 | -0.784 | [-0.869, -0.693] | 88 / 73 | 0 / 69 |
| supported_correct with distractor passage | 0.067 | 0.600 | +0.533 | [+0.367, +0.700] | 30 / 30 | 16 / 0 |

## Paired difference: main vs control

95% interval from resampling whole questions; both systems ran the same episodes.

| metric | before | after | difference | 95% CI | episodes / questions | only-after / only-before good |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| end-to-end good (headline: supported answer when answerable, abstain when not) | 0.277 | 0.673 | +0.396 | [+0.335, +0.456] | 159 / 86 | 75 / 12 |
| supported_correct (answerable episodes) | 0.532 | 0.468 | -0.063 | [-0.135, +0.005] | 205 / 86 | 13 / 26 |
| answer correct, ignoring citations (answerable; F1 >= 0.5) | 0.532 | 0.468 | -0.063 | [-0.135, +0.005] | 205 / 86 | 13 / 26 |
| correct_abstain (unanswerable episodes) | 0.000 | 0.875 | +0.875 | [+0.805, +0.942] | 88 / 73 | 77 / 0 |
| unsupported answer (unanswerable; lower is better) | 0.284 | 0.114 | -0.170 | [-0.247, -0.099] | 88 / 73 | 0 / 15 |
| supported_correct with distractor passage | 0.667 | 0.600 | -0.067 | [-0.233, +0.100] | 30 / 30 | 2 / 4 |
