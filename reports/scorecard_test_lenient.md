# Scorecard (test): main against base

Each number is reported separately with its denominator. There is no composite score.

## when_to_retrieve

| metric | base | main | control |
| --- | ---: | ---: | ---: |
| missed_search_rate (no evidence supplied; lower is better) | 0.072 (13/180) | 0.000 (0/180) | 0.000 (0/180) |
| unnecessary_search_rate (sufficient evidence supplied; lower is better) | 1.000 (60/60) | 0.283 (17/60) | 0.067 (4/60) |
| search_when_evidence_partial (higher is better) | 0.933 (28/30) | 0.733 (22/30) | 0.800 (24/30) |

## what_to_retrieve

| metric | base | main | control |
| --- | ---: | ---: | ---: |
| full_support_found_by_own_queries (open questions) | 0.360 (31/86) | 0.709 (61/86) | 0.826 (71/86) |
| second_hop_recovery (partial or weak first result) | 0.237 (14/59) | 0.475 (28/59) | 0.644 (38/59) |
| auditkit_evidence_set_recall | 0.512 | 0.727 | 0.824 |

## grounding

| metric | base | main | control |
| --- | ---: | ---: | ---: |
| supported_correct (answerable episodes) | 0.034 (7/205) | 0.527 (108/205) | 0.707 (145/205) |
| supported_correct (open questions) | 0.012 (1/86) | 0.442 (38/86) | 0.709 (61/86) |
| answer_correct_ignoring_citations (answerable; F1 >= 0.5) | 0.141 (29/205) | 0.532 (109/205) | 0.712 (146/205) |
| supported_exact_match (answerable; exact match instead of F1 >= 0.5) | 0.020 (4/205) | 0.424 (87/205) | 0.571 (117/205) |
| cites_a_gold_passage_when_answering (answerable) | 0.126 (21/167) | 0.994 (157/158) | 0.990 (191/193) |
| answer_f1_when_answered | 0.166 | 0.643 | 0.690 |
| auditkit_citation_support | 0.302 | 0.771 | 0.941 |

## ignores_distractors

| metric | base | main | control |
| --- | ---: | ---: | ---: |
| supported_correct with distractor passage | 0.067 (2/30) | 0.667 (20/30) | 0.833 (25/30) |
| supported_correct, same evidence without distractor | 0.000 (0/30) | 0.633 (19/30) | 0.800 (24/30) |

## abstention (documents-only requirement)

| metric | base | main | control |
| --- | ---: | ---: | ---: |
| correct_abstain (unanswerable episodes) | 0.024 (2/82) | 0.756 (62/82) | 0.000 (0/82) |
| unsupported_answer_rate (unanswerable; lower is better) | 0.780 (64/82) | 0.098 (8/82) | 0.268 (22/82) |
| false_abstain (answerable; lower is better) | 0.000 (0/205) | 0.190 (39/205) | 0.000 (0/205) |
| auditkit_abstention | 0.721 | 0.794 | 0.714 |

## tool_call_reliability (AuditKIT, Track 3)

| metric | base | main | control |
| --- | ---: | ---: | ---: |
| tool_call_validity | 1.000 | 1.000 | 1.000 |
| redundant_tool_calls (lower is better) | 0.031 | 0.029 | 0.088 |
| agent_loop_detection (1.0 = loop-free) | 0.941 | 0.934 | 0.784 |
| invalid_turn_rate (unparseable or rule-breaking; lower is better) | 0.188 (54/287) | 0.070 (20/287) | 0.251 (72/287) |

## efficiency

| metric | base | main | control |
| --- | ---: | ---: | ---: |
| mean_searches_per_episode | 0.950 | 1.640 | 1.450 |
| mean_input_tokens_per_episode | 2105.100 | 3233.200 | 2957.600 |
| mean_seconds_per_episode | 2.782 | 4.124 | 3.618 |

## Episodes

| system | episodes |
| --- | ---: |
| base | 287 |
| main | 287 |
| control | 287 |

## Paired difference: main vs base

95% interval from resampling whole questions; both systems ran the same episodes.

| metric | before | after | difference | 95% CI | episodes / questions | only-after / only-before good |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| end-to-end good (headline: supported answer when answerable, abstain when not) | 0.020 | 0.583 | +0.563 | [+0.490, +0.636] | 151 / 86 | 85 / 0 |
| supported_correct (answerable episodes) | 0.034 | 0.527 | +0.493 | [+0.391, +0.592] | 205 / 86 | 103 / 2 |
| answer correct, ignoring citations (answerable; F1 >= 0.5) | 0.141 | 0.532 | +0.390 | [+0.278, +0.498] | 205 / 86 | 93 / 13 |
| correct_abstain (unanswerable episodes) | 0.024 | 0.756 | +0.732 | [+0.636, +0.825] | 82 / 66 | 60 / 0 |
| unsupported answer (unanswerable; lower is better) | 0.780 | 0.098 | -0.683 | [-0.786, -0.575] | 82 / 66 | 1 / 57 |
| supported_correct with distractor passage | 0.067 | 0.667 | +0.600 | [+0.433, +0.767] | 30 / 30 | 18 / 0 |

## Paired difference: main vs control

95% interval from resampling whole questions; both systems ran the same episodes.

| metric | before | after | difference | 95% CI | episodes / questions | only-after / only-before good |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| end-to-end good (headline: supported answer when answerable, abstain when not) | 0.404 | 0.583 | +0.179 | [+0.103, +0.252] | 151 / 86 | 54 / 27 |
| supported_correct (answerable episodes) | 0.707 | 0.527 | -0.180 | [-0.268, -0.099] | 205 / 86 | 11 / 48 |
| answer correct, ignoring citations (answerable; F1 >= 0.5) | 0.712 | 0.532 | -0.180 | [-0.267, -0.101] | 205 / 86 | 10 / 47 |
| correct_abstain (unanswerable episodes) | 0.000 | 0.756 | +0.756 | [+0.667, +0.848] | 82 / 66 | 62 / 0 |
| unsupported answer (unanswerable; lower is better) | 0.268 | 0.098 | -0.171 | [-0.261, -0.087] | 82 / 66 | 3 / 17 |
| supported_correct with distractor passage | 0.833 | 0.667 | -0.167 | [-0.333, +0.000] | 30 / 30 | 1 / 6 |
