# AuditKIT metrics: saved DoRA benchmark traces

AuditKIT 1.1.3; 938 input episodes; CPU scoring, no new model calls.

| Metric | Cohort | DoRA | DoRA+FB | DoRA+FB+V | Scored n (same order) |
| --- | --- | ---: | ---: | ---: | --- |
| evidence_set_recall | rag_stress | 0.8028 | 0.8028 | 0.8028 | 563 / 563 / 563 |
| citation_support | rag_stress | 0.9663 | 0.9663 | 0.7531 | 563 / 563 / 563 |
| citation_precision | rag_stress | 0.5995 | 0.5995 | 0.5178 | 563 / 563 / 563 |
| citation_recall | rag_stress | 0.4057 | 0.4057 | 0.3391 | 563 / 563 / 563 |
| tool_call_validity | agent | 1.0000 | 1.0000 | 1.0000 | 400 / 400 / 400 |
| redundant_tool_calls | agent | 0.0033 | 0.0033 | 0.0033 | 400 / 400 / 400 |
| agent_loop_detection | agent | 0.9947 | 0.9947 | 0.9947 | 938 / 938 / 938 |
| abstention | abstention | 0.6524 | 0.6525 | 0.7164 | 935 / 938 / 938 |
| lexical_groundedness | answerable_answered | 0.9768 | 0.9768 | 0.9929 | 544 / 544 / 424 |
| context_overlap | answerable_answered | 0.6692 | 0.6692 | 0.6999 | 544 / 544 / 424 |
| answer_overlap | answerable_answered | 0.6607 | 0.6607 | 0.7650 | 544 / 544 / 424 |
| f1_score | answerable_answered | 0.6224 | 0.6224 | 0.7246 | 544 / 544 / 424 |
| context_coverage | answerable_context | 0.8773 | 0.8773 | 0.8773 | 563 / 563 / 563 |
| hit_rate@3 | answerable_first_search | 0.9810 | 0.9810 | 0.9810 | 211 / 211 / 211 |
| precision@3 | answerable_first_search | 0.5498 | 0.5498 | 0.5498 | 211 / 211 / 211 |
| recall@3 | answerable_first_search | 0.6077 | 0.6077 | 0.6077 | 211 / 211 / 211 |
| mrr@3 | answerable_first_search | 0.8799 | 0.8799 | 0.8799 | 211 / 211 / 211 |
| average_precision@3 | answerable_first_search | 0.7242 | 0.7242 | 0.7242 | 211 / 211 / 211 |
| ndcg@3 | answerable_first_search | 0.7817 | 0.7817 | 0.7817 | 211 / 211 / 211 |

Metric scopes and interpretation:

- All 938 original episodes are audited; the separate scorer-v2 exclusions and retained counts are recorded in computed_round5.json and can vary by checkpoint.
- Citation support checks existence/current-version of mapped document spans, not semantic entailment. These traces have document ids, so spans cover the visible prefix (800 characters maximum).
- RGB alternative positives are OR evidence sets; historical benchmark scores are preserved.
- This diagnostic uses recorded-v2; compare other models only after rescoring their traces with the same adapter.
- Lexical scores are token-overlap diagnostics, not LLM-judged faithfulness. Answer scores use the canonical answer, not alias matching.
- First-search ranking is measured only for answerable episodes with a search; supplied context and later searches are excluded from that ranked list.
- Abstention accuracy is measured on valid final actions; invalid actions are counted separately.
- Precomputed scoring latency is not model inference latency; no fresh inference performance claim is made.

Untested metric families:

- faithfulness/context_precision/context_recall/context_relevance/answer_relevancy/response_groundedness/hallucination/task_completion/tool_selection: Requires an LLM judge; no judge backend was configured or called.
- tool_call_f1/trajectory_match/parallel_tool_calls: No unique reference tool-call route or parallel-call ground truth in these traces.
- wrong_version_rate/freshness/acl_compliance/context_retention/numeric_accuracy/tool_permission: Required versions, dates, permission policies, conversational anchors or typed numeric labels are absent. The citation scorer's default wrong_version_rate=0 is not reported as a measured temporal result.
- security/toxicity/bias/guard/pairwise/perplexity: Require their own tasks, labels, models, comparisons or logits; not inferred from RAG traces.

Sources: [AuditKIT metrics](https://github.com/Lexsi-Labs/AuditKIT/blob/main/src/auditkit/metrics/README.md), [Lexsi Track 3](https://hackathon.lexsi.ai/#tracks).
