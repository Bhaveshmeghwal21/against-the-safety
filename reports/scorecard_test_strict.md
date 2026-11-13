# Scorecard (test): main against base

**13 November 2026 correction:** This table preserves the original scorer and selection record. Evidence alternatives, binary answers, fixed eligibility and source offsets are corrected in the [dated remediation report](2026-11-13-evaluation-remediation.md). Later DPO prompts exposed held-out HotpotQA evidence. Annotation-based unsupported flags do not establish semantic hallucination; main remains submitted.


Each number is reported separately with its denominator. There is no composite score.

## when_to_retrieve

| metric | base | main | control |
| --- | ---: | ---: | ---: |
| missed_search_rate (no evidence supplied; lower is better) | 0.861 (155/180) | 0.000 (0/180) | 0.000 (0/180) |
| unnecessary_search_rate (sufficient evidence supplied; lower is better) | 0.150 (9/60) | 0.283 (17/60) | 0.067 (4/60) |
| search_when_evidence_partial (higher is better) | 0.133 (4/30) | 0.733 (22/30) | 0.800 (24/30) |

## what_to_retrieve

| metric | base | main | control |
| --- | ---: | ---: | ---: |
| full_support_found_by_own_queries (open questions) | 0.047 (4/86) | 0.709 (61/86) | 0.826 (71/86) |
| second_hop_recovery (partial or weak first result) | 0.051 (3/59) | 0.475 (28/59) | 0.644 (38/59) |
| auditkit_evidence_set_recall | 0.327 | 0.727 | 0.824 |

## grounding

| metric | base | main | control |
| --- | ---: | ---: | ---: |
| supported_correct (answerable episodes) | 0.010 (2/205) | 0.527 (108/205) | 0.707 (145/205) |
| supported_correct (open questions) | 0.000 (0/86) | 0.442 (38/86) | 0.709 (61/86) |
| answer_correct_ignoring_citations (answerable; F1 >= 0.5) | 0.015 (3/205) | 0.532 (109/205) | 0.712 (146/205) |
| supported_exact_match (answerable; exact match instead of F1 >= 0.5) | 0.010 (2/205) | 0.424 (87/205) | 0.571 (117/205) |
| cites_a_gold_passage_when_answering (answerable) | 0.500 (3/6) | 0.994 (157/158) | 0.990 (191/193) |
| answer_f1_when_answered | 0.406 | 0.643 | 0.689 |
| auditkit_citation_support | 0.029 | 0.771 | 0.941 |

## ignores_distractors

| metric | base | main | control |
| --- | ---: | ---: | ---: |
| supported_correct with distractor passage | 0.033 (1/30) | 0.667 (20/30) | 0.833 (25/30) |
| supported_correct, same evidence without distractor | 0.000 (0/30) | 0.633 (19/30) | 0.800 (24/30) |

## abstention (documents-only requirement)

| metric | base | main | control |
| --- | ---: | ---: | ---: |
| correct_abstain (unanswerable episodes) | 0.000 (0/82) | 0.756 (62/82) | 0.000 (0/82) |
| unsupported_answer_rate (unanswerable; lower is better) | 0.024 (2/82) | 0.098 (8/82) | 0.256 (21/82) |
| false_abstain (answerable; lower is better) | 0.000 (0/205) | 0.190 (39/205) | 0.000 (0/205) |
| auditkit_abstention | 0.714 | 0.794 | 0.714 |

## tool_call_reliability (AuditKIT, Track 3)

| metric | base | main | control |
| --- | ---: | ---: | ---: |
| tool_call_validity | 1.000 | 1.000 | 1.000 |
| redundant_tool_calls (lower is better) | 0.000 | 0.029 | 0.088 |
| agent_loop_detection (1.0 = loop-free) | 1.000 | 0.934 | 0.784 |
| invalid_turn_rate (unparseable or rule-breaking; lower is better) | 0.972 (279/287) | 0.070 (20/287) | 0.254 (73/287) |

## efficiency

| metric | base | main | control |
| --- | ---: | ---: | ---: |
| mean_searches_per_episode | 0.140 | 1.640 | 1.450 |
| mean_input_tokens_per_episode | 1071.100 | 3233.200 | 2957.600 |
| mean_seconds_per_episode | 1.532 | 4.133 | 3.614 |

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
| end-to-end good (headline: supported answer when answerable, abstain when not) | 0.000 | 0.583 | +0.583 | [+0.513, +0.651] | 151 / 86 | 88 / 0 |
| supported_correct (answerable episodes) | 0.010 | 0.527 | +0.517 | [+0.412, +0.618] | 205 / 86 | 107 / 1 |
| answer correct, ignoring citations (answerable; F1 >= 0.5) | 0.015 | 0.532 | +0.517 | [+0.410, +0.620] | 205 / 86 | 108 / 2 |
| correct_abstain (unanswerable episodes) | 0.000 | 0.756 | +0.756 | [+0.667, +0.848] | 82 / 66 | 62 / 0 |
| unsupported answer (unanswerable; lower is better) | 0.024 | 0.098 | +0.073 | [+0.000, +0.148] | 82 / 66 | 7 / 1 |
| supported_correct with distractor passage | 0.033 | 0.667 | +0.633 | [+0.467, +0.800] | 30 / 30 | 19 / 0 |

## Paired difference: main vs control

95% interval from resampling whole questions; both systems ran the same episodes.

| metric | before | after | difference | 95% CI | episodes / questions | only-after / only-before good |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| end-to-end good (headline: supported answer when answerable, abstain when not) | 0.404 | 0.583 | +0.179 | [+0.103, +0.252] | 151 / 86 | 54 / 27 |
| supported_correct (answerable episodes) | 0.707 | 0.527 | -0.180 | [-0.268, -0.099] | 205 / 86 | 11 / 48 |
| answer correct, ignoring citations (answerable; F1 >= 0.5) | 0.712 | 0.532 | -0.180 | [-0.267, -0.101] | 205 / 86 | 10 / 47 |
| correct_abstain (unanswerable episodes) | 0.000 | 0.756 | +0.756 | [+0.667, +0.848] | 82 / 66 | 62 / 0 |
| unsupported answer (unanswerable; lower is better) | 0.256 | 0.098 | -0.159 | [-0.241, -0.081] | 82 / 66 | 3 / 16 |
| supported_correct with distractor passage | 0.833 | 0.667 | -0.167 | [-0.333, +0.000] | 30 / 30 | 1 / 6 |
