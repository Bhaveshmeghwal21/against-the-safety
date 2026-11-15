**DoRA comparative study: previous Modal run vs Kaggle dora_lr5e5**

Prepared 13 November 2026 from completed run records. This study performs no new model inference.

**The previous Modal checkpoint is stronger on the reported benchmark.** The Kaggle checkpoint reduces invalid outputs and false refusals, but produces substantially more answers when evidence is hidden. The fixed verifier reduces the gap without reversing the result.

All differences below are new minus previous. Benchmark differences are percentage points (pp), not relative percent changes. AuditKIT tables retain the original 0–1 scale.

**Records and comparison conditions.** The original baseline AuditKIT JSON path currently contains prose beginning with "Metrics", not valid JSON. Baseline audit values therefore come from the full-precision [preserved report snapshot](dora_evaluation_data.json); the altered file was not changed. Baseline benchmark JSON agrees exactly with that snapshot. New training metadata was downloaded separately and its adapter-configuration hash and asset manifest match the evaluation identity.

The two evaluations use the same pinned Tiny Aya base and reference-adapter hashes, main/A/0.5 verifier, AuditKIT 1.1.3 metric implementation hashes and recorded-v2 audit contract. The newer wrapper uses offline SQuAD-v1 answer scoring and a revised audit wrapper; its source hashes differ from the older wrapper. These records support a descriptive comparison, not bit-identical end-to-end execution.

| Recorded setting | Previous Modal | New Kaggle |
| --- | --- | --- |
| Learning rate | 0.0002 | 5e-05 |
| Training GPUs | 1 L4 | 4 L4 |
| Rank / alpha / dropout | 16 / 32 / 0.05 | 16 / 32 / 0.05 |
| Epochs / seed / maximum length | 2 / 13 / 2560 | 2 / 13 / 2560 |
| Effective batch / per-GPU accumulation | 16 / 16 | 16 / 4 |
| Rows / source episodes / optimizer steps | 1611 / 1331 / 202 | 1611 / 1331 / 202 |
| Reported training loss | 0.094968 | 0.161658 |
| Training wall time | 219.08 min | 62.73 min |
| Reported peak GPU memory | 5.98 GB (one GPU) | 6.09 GB (rank 0; not total across four GPUs) |

The previous-to-new reported training wall-time ratio is 3.49× (219.08 to 62.73 minutes). Estimated allocated GPU-hours (GPUs × training wall time) are 3.65 versus 4.18; this is not a measured billing comparison. Preparation, evaluation, queues and platform pricing are excluded. The reported software versions, adapter parameters and token counts match. Matching row counts alone do not prove every training byte was identical.

**Overall benchmark.** FB converts invalid final actions to abstentions. V uses the same historical main LoRA answer verifier; these configurations do not represent separately trained models.

| Configuration | Previous macro | New macro | Change (pp) |
| --- | --- | --- | --- |
| dora | 44.71% | 34.00% | -10.71 |
| dora+FB | 48.18% | 35.91% | -12.27 |
| dora+FB+V | 50.09% | 47.34% | -2.75 |

**Performance by benchmark component.** Macro success gives equal weight to these four components.

| Component | Previous raw | New raw | Raw Δ pp | Previous FB+V | New FB+V | FB+V Δ pp |
| --- | --- | --- | --- | --- | --- | --- |
| SQuAD single | 60.91% | 48.18% | -12.73 | 60.45% | 57.27% | -3.18 |
| SQuAD two-part | 26.67% | 13.94% | -12.73 | 38.79% | 33.33% | -5.45 |
| HotpotQA | 53.66% | 43.90% | -9.76 | 52.44% | 51.22% | -1.22 |
| RGB | 37.61% | 29.98% | -7.63 | 48.70% | 47.54% | -1.16 |

Raw success declines in every component. Single and two-part SQuAD each lose 12.73 pp; two-part SQuAD remains the weakest component. After both repairs, two-part SQuAD still has the largest gap (−5.45 pp).

**Answering versus abstaining.** Supported correctness requires the benchmark answer and evidence/aspect conditions. An unsupported answer means the agent answered when evidence was hidden, even if its answer text happened to be factually correct.

| Configuration | Unsupported / hidden: old → new | Δ pp | Supported correct / 563: old → new | False refusals / 563: old → new |
| --- | --- | --- | --- | --- |
| dora | 117/364 (32.14%) → 227/371 (61.19%) | +29.04 | 176 (31.26%) → 174 (30.91%) | 112 (19.89%) → 69 (12.26%) |
| dora+FB | 117/364 (32.14%) → 227/371 (61.19%) | +29.04 | 176 (31.26%) → 174 (30.91%) | 130 (23.09%) → 78 (13.85%) |
| dora+FB+V | 64/364 (17.58%) → 91/371 (24.53%) | +6.95 | 164 (29.13%) → 165 (29.31%) | 194 (34.46%) → 171 (30.37%) |

Raw invalid final actions fall from 46/938 (4.90%) to 22/938 (2.35%). The agent answers more answerable episodes (433 → 485), yet supported successes fall slightly (176 → 174). Fewer refusals therefore do not translate into more supported correct answers. The large unsupported-answer increase is the main observed behavioral weakness.

The verifier adds 1.92 pp over FB for the previous run and 11.43 pp for the new run. It removes 53 versus 136 unsupported answers, while false refusals rise by 64 versus 93 answerable episodes relative to FB. The repaired new checkpoint still has 91 unsupported answers versus 64 previously, with almost identical supported successes (165 versus 164).

**All measured AuditKIT outputs.** Counts are old → new. Higher scores are generally preferred except redundant_tool_calls; changes on different conditional cohorts do not isolate model quality. agent_loop_detection is a loop-free score (higher is better).

**dora: AuditKIT comparison**

| Metric | Cohort | Previous | New | Δ (0–1) | Scored n |
| --- | --- | --- | --- | --- | --- |
| evidence_set_recall | rag_stress | 0.9449 | 0.8970 | -0.0480 | 563 → 563 |
| citation_support | rag_stress | 0.7691 | 0.8615 | +0.0924 | 563 → 563 |
| citation_precision | rag_stress | 0.5731 | 0.5613 | -0.0118 | 563 → 563 |
| citation_recall | rag_stress | 0.3643 | 0.3806 | +0.0163 | 563 → 563 |
| tool_call_validity | agent | 1.0000 | 1.0000 | +0.0000 | 722 → 566 |
| redundant_tool_calls | agent | 0.0249 | 0.0156 | -0.0093 | 722 → 566 |
| agent_loop_detection | agent | 0.9499 | 0.9765 | +0.0267 | 938 → 938 |
| abstention | abstention | 0.7321 | 0.6725 | -0.0596 | 892 → 916 |
| lexical_groundedness | answerable_answered | 0.9868 | 0.9841 | -0.0027 | 433 → 485 |
| context_overlap | answerable_answered | 0.6901 | 0.6607 | -0.0294 | 433 → 485 |
| answer_overlap | answerable_answered | 0.7941 | 0.7070 | -0.0871 | 433 → 485 |
| f1_score | answerable_answered | 0.7607 | 0.6725 | -0.0882 | 433 → 485 |
| context_coverage | answerable_context | 0.9607 | 0.9281 | -0.0326 | 563 → 563 |
| hit_rate@3 | answerable_first_search | 0.9549 | 0.9572 | +0.0023 | 377 → 304 |
| precision@3 | answerable_first_search | 0.5208 | 0.5175 | -0.0032 | 377 → 304 |
| recall@3 | answerable_first_search | 0.6516 | 0.6590 | +0.0074 | 377 → 304 |
| mrr@3 | answerable_first_search | 0.8700 | 0.8690 | -0.0011 | 377 → 304 |
| average_precision@3 | answerable_first_search | 0.7390 | 0.7412 | +0.0022 | 377 → 304 |
| ndcg@3 | answerable_first_search | 0.7890 | 0.7904 | +0.0014 | 377 → 304 |

**dora+FB: AuditKIT comparison**

| Metric | Cohort | Previous | New | Δ (0–1) | Scored n |
| --- | --- | --- | --- | --- | --- |
| evidence_set_recall | rag_stress | 0.9449 | 0.8970 | -0.0480 | 563 → 563 |
| citation_support | rag_stress | 0.7691 | 0.8615 | +0.0924 | 563 → 563 |
| citation_precision | rag_stress | 0.5731 | 0.5613 | -0.0118 | 563 → 563 |
| citation_recall | rag_stress | 0.3643 | 0.3806 | +0.0163 | 563 → 563 |
| tool_call_validity | agent | 1.0000 | 1.0000 | +0.0000 | 722 → 566 |
| redundant_tool_calls | agent | 0.0249 | 0.0156 | -0.0093 | 722 → 566 |
| agent_loop_detection | agent | 0.9499 | 0.9765 | +0.0267 | 938 → 938 |
| abstention | abstention | 0.7260 | 0.6706 | -0.0554 | 938 → 938 |
| lexical_groundedness | answerable_answered | 0.9868 | 0.9841 | -0.0027 | 433 → 485 |
| context_overlap | answerable_answered | 0.6901 | 0.6607 | -0.0294 | 433 → 485 |
| answer_overlap | answerable_answered | 0.7941 | 0.7070 | -0.0871 | 433 → 485 |
| f1_score | answerable_answered | 0.7607 | 0.6725 | -0.0882 | 433 → 485 |
| context_coverage | answerable_context | 0.9607 | 0.9281 | -0.0326 | 563 → 563 |
| hit_rate@3 | answerable_first_search | 0.9549 | 0.9572 | +0.0023 | 377 → 304 |
| precision@3 | answerable_first_search | 0.5208 | 0.5175 | -0.0032 | 377 → 304 |
| recall@3 | answerable_first_search | 0.6516 | 0.6590 | +0.0074 | 377 → 304 |
| mrr@3 | answerable_first_search | 0.8700 | 0.8690 | -0.0011 | 377 → 304 |
| average_precision@3 | answerable_first_search | 0.7390 | 0.7412 | +0.0022 | 377 → 304 |
| ndcg@3 | answerable_first_search | 0.7890 | 0.7904 | +0.0014 | 377 → 304 |

**dora+FB+V: AuditKIT comparison**

| Metric | Cohort | Previous | New | Δ (0–1) | Scored n |
| --- | --- | --- | --- | --- | --- |
| evidence_set_recall | rag_stress | 0.9449 | 0.8970 | -0.0480 | 563 → 563 |
| citation_support | rag_stress | 0.6554 | 0.6963 | +0.0409 | 563 → 563 |
| citation_precision | rag_stress | 0.5074 | 0.4911 | -0.0163 | 563 → 563 |
| citation_recall | rag_stress | 0.3068 | 0.3254 | +0.0185 | 563 → 563 |
| tool_call_validity | agent | 1.0000 | 1.0000 | +0.0000 | 722 → 566 |
| redundant_tool_calls | agent | 0.0249 | 0.0156 | -0.0093 | 722 → 566 |
| agent_loop_detection | agent | 0.9499 | 0.9765 | +0.0267 | 938 → 938 |
| abstention | abstention | 0.7164 | 0.7196 | +0.0032 | 938 → 938 |
| lexical_groundedness | answerable_answered | 0.9973 | 0.9949 | -0.0024 | 369 → 392 |
| context_overlap | answerable_answered | 0.7114 | 0.7059 | -0.0055 | 369 → 392 |
| answer_overlap | answerable_answered | 0.8375 | 0.7814 | -0.0561 | 369 → 392 |
| f1_score | answerable_answered | 0.7996 | 0.7437 | -0.0559 | 369 → 392 |
| context_coverage | answerable_context | 0.9607 | 0.9281 | -0.0326 | 563 → 563 |
| hit_rate@3 | answerable_first_search | 0.9549 | 0.9572 | +0.0023 | 377 → 304 |
| precision@3 | answerable_first_search | 0.5208 | 0.5175 | -0.0032 | 377 → 304 |
| recall@3 | answerable_first_search | 0.6516 | 0.6590 | +0.0074 | 377 → 304 |
| mrr@3 | answerable_first_search | 0.8700 | 0.8690 | -0.0011 | 377 → 304 |
| average_precision@3 | answerable_first_search | 0.7390 | 0.7412 | +0.0022 | 377 → 304 |
| ndcg@3 | answerable_first_search | 0.7890 | 0.7904 | +0.0014 | 377 → 304 |

Evidence-set recall falls from 0.9449 to 0.8970 and context coverage from 0.9607 to 0.9281, both on the fixed 563 answerable episodes. First-search hit rate stays near 95.5%, but the searched-answerable cohort shrinks from 377 to 304. Similar first-search precision is therefore compatible with worse evidence coverage overall; it does not establish that retrieval improved.

Raw citation support rises (0.7691 → 0.8615) and citation recall rises (0.3643 → 0.3806), while citation precision falls (0.5731 → 0.5613). Citation support checks mapped spans and document identity, not semantic entailment. These improvements cannot override lower supported success.

Conditional raw answer F1 falls from 0.7607 to 0.6725, on 433 versus 485 answered-answerable episodes. After verification it is 0.7996 versus 0.7437, on 369 versus 392 episodes. Verifier filtering changes which answers remain. Lexical groundedness stays high but measures token overlap, not an LLM judgment of factual faithfulness.

Tool-call validity remains 1.0 on episodes with calls; it does not guarantee valid final actions. Redundant-call fraction falls from 0.02493 to 0.01561, with cohorts of 722 and 566. The loop-free score improves from 0.9499 to 0.9765 on all 938 episodes. These are useful behavior improvements without an observed increase in end-to-end success.

**Limits of the comparison.** Both runs start with the same 938 benchmark episodes, but scorer-v2 retains 927 versus 934, including 364 versus 371 hidden-evidence episodes. Both retain 563 answerable episodes. AuditKIT uses all 938 inputs with metric-specific cohorts. No paired confidence intervals or common-intersection analysis were computed; the results are descriptive, not a statistical significance claim. The published suite has already been inspected and is not a fresh holdout.

Learning rate changed fourfold and training/evaluation distribution changed from one GPU to four. The higher training loss is consistent with weaker fit at this recipe, but does not prove undertraining or identify a causal learning-rate effect. This experiment does not establish that four GPUs reduce accuracy. Judge-based faithfulness, hallucination and safety metrics remain unmeasured.

**Next experiment.** Keep the previous Modal checkpoint as the stronger descriptive reference for benchmark success and evidence-sensitive abstention. Evaluate the already trained dora_lr1e4 and dora_lr5e5 checkpoints on development data. A matched four-L4 run at the original 2e-4 rate, with the same rank, dropout, epochs, seed and global batch, would help separate the learning-rate change from the platform change. Use development results to select a recipe, then repeat the selected recipe at another seed; do not tune the verifier threshold on these benchmark results.

**Source files and downloadable tables**

- [Full comparison CSV](dora_kaggle_lr5e5_comparison.csv): 27 benchmark comparisons and 57 AuditKIT comparisons.
- [Comparison JSON](dora_kaggle_lr5e5_comparison.json): full precision, source hashes, training metadata and caveats.
- [Previous report](dora_evaluation_report.md) and [preserved data snapshot](dora_evaluation_data.json).
- [New benchmark JSON](kaggle_lr5e5_evaluation/reports/computed_round5.json).
- [New AuditKIT JSON](kaggle_lr5e5_evaluation/auditkit/auditkit_metrics.json).
- [New training report](kaggle_lr5e5_evaluation/train_report.json).
