# Evidence-order DoRA compared with all previous local benchmark records

Prepared 14 November 2026. Source: completed Kaggle evaluation `mlsingh1234/evaluation/v5`, script version `369426017`. This study performs no new inference or training.

**The latest approach regresses on overall benchmark success.** Its raw macro is 27.82%, versus 34.00% for its Kaggle parent and 44.71% for the original Modal DoRA. It answers more supported questions, but also answers substantially more questions whose evidence is hidden. The fixed verifier helps, but does not reverse the overall regression.

## Scope and comparability

The primary comparison covers every locally recorded adapter family on the same 938-input round-5 suite: base, main, candA, candF1, original Modal DoRA, Kaggle lr5e-5 DoRA, and the latest evidence-order DoRA. Round 5 and 5b duplicate the same base/main/candA rows; those rows are counted once. Hosted-model intersections and earlier test/development suites are included separately below. The latest checkpoint has not been evaluated on those older suites, so they cannot establish its rank there.

All three DoRA runs share the pinned base/reference-adapter hashes, fixed main/A/0.5 verifier, AuditKIT 1.1.3 metric implementation hashes, and recorded-v2 audit contract. The audit wrapper/source hashes and offline scoring packaging differ across old and new runs; this is a descriptive comparison, not a claim of identical execution. Scorer-v2 exclusions vary by checkpoint. No common-intersection recomputation, paired confidence interval, or significance test is claimed. The test suite was already public when the DoRA experiments ran.

The historical Modal audit JSON path was previously overwritten with prose; the full-precision [preserved snapshot](dora_evaluation_data.json) supplies its audit values. The lr1e-4 adapter has training metadata but no separate benchmark evaluation available locally.

## All comparable round-5 model families

Macro is an equal mean of SQuAD single, SQuAD two-part, HotpotQA and RGB success. FB converts invalid endings to abstentions; V applies the saved main-adapter verifier. These are inference repairs of one generator's traces, not separate trained models.

| Model | Raw macro ↑ | +FB ↑ | +FB+V ↑ | Retained n | Hidden n |
| --- | --- | --- | --- | --- | --- |
| Base | 2.22% | 11.32% | — | 932 | 369 |
| Main LoRA | 41.43% | 47.37% | 51.74% | 927 | 364 |
| candA LoRA | 47.34% | 49.38% | 54.10% | 928 | 365 |
| candF1 LoRA | 44.93% | 52.15% | 56.97% | 927 | 364 |
| Modal DoRA | 44.71% | 48.18% | 50.09% | 927 | 364 |
| Kaggle DoRA lr5e-5 | 34.00% | 35.91% | 47.34% | 934 | 371 |
| Evidence-order DoRA | 27.82% | 28.43% | 45.41% | 937 | 374 |

The latest model is descriptively **6/7 raw**, and **6/6 with FB+V**, among these families with recorded configurations. candA has the strongest raw adapter score; candF1 has the strongest repaired adapter score. The historical selected submission remains main+FB+V: candF1 failed the preregistered unsupported-answer non-inferiority rule despite its higher macro. See [round5b_final.md](round5b_final.md).

## Per-component results for every compatible configuration

| Model | Configuration | SQuAD single | SQuAD two-part | HotpotQA | RGB |
| --- | --- | --- | --- | --- | --- |
| Base | raw | 1.82% | 3.64% | 0.00% | 3.44% |
| Base | FB | 12.73% | 17.58% | 4.88% | 10.11% |
| Main LoRA | raw | 54.55% | 22.42% | 52.44% | 36.30% |
| Main LoRA | FB | 60.00% | 32.73% | 56.10% | 40.65% |
| Main LoRA | FB+V | 59.09% | 39.39% | 56.10% | 52.39% |
| candA LoRA | raw | 63.64% | 25.00% | 62.20% | 38.53% |
| candA LoRA | FB | 65.45% | 29.27% | 63.41% | 39.39% |
| candA LoRA | FB+V | 61.36% | 39.02% | 63.41% | 52.60% |
| candF1 LoRA | raw | 54.09% | 33.33% | 54.32% | 37.96% |
| candF1 LoRA | FB | 62.73% | 46.06% | 59.26% | 40.56% |
| candF1 LoRA | FB+V | 61.36% | 52.73% | 61.73% | 52.06% |
| Modal DoRA | raw | 60.91% | 26.67% | 53.66% | 37.61% |
| Modal DoRA | FB | 62.73% | 35.76% | 54.88% | 39.35% |
| Modal DoRA | FB+V | 60.45% | 38.79% | 52.44% | 48.70% |
| Kaggle DoRA lr5e-5 | raw | 48.18% | 13.94% | 43.90% | 29.98% |
| Kaggle DoRA lr5e-5 | FB | 50.91% | 14.55% | 47.56% | 30.62% |
| Kaggle DoRA lr5e-5 | FB+V | 57.27% | 33.33% | 51.22% | 47.54% |
| Evidence-order DoRA | raw | 39.55% | 4.85% | 39.02% | 27.87% |
| Evidence-order DoRA | FB | 39.55% | 4.85% | 41.46% | 27.87% |
| Evidence-order DoRA | FB+V | 50.00% | 29.09% | 54.88% | 47.66% |

## Direct DoRA progression and deltas

Differences are latest minus the named reference in percentage points (pp), not relative percent change.

| Reference | Configuration | Macro Δ ↑ | Supported correct Δ ↑ | Unsupported Δ ↓ | False abstain Δ ↓ |
| --- | --- | --- | --- | --- | --- |
| Modal DoRA | dora | -16.89 | +2.84 | +49.68 | -16.70 |
| Modal DoRA | dora+FB | -19.75 | +2.84 | +49.68 | -19.72 |
| Modal DoRA | dora+FB+V | -4.69 | +2.66 | +16.11 | -9.77 |
| Kaggle DoRA lr5e-5 | dora | -6.18 | +3.20 | +20.63 | -9.06 |
| Kaggle DoRA lr5e-5 | dora+FB | -7.48 | +3.20 | +20.63 | -10.48 |
| Kaggle DoRA lr5e-5 | dora+FB+V | -1.93 | +2.49 | +9.16 | -5.68 |

| Model | Configuration | Supported correct / answerable | Unsupported / hidden | False abstain / answerable | Raw invalid / input |
| --- | --- | --- | --- | --- | --- |
| Modal DoRA | raw | 176/563 (31.26%) | 117/364 (32.14%) | 112/563 (19.89%) | 46/938 (4.90%) |
| Modal DoRA | FB | 176/563 (31.26%) | 117/364 (32.14%) | 130/563 (23.09%) | 46/938 (4.90%) |
| Modal DoRA | FB+V | 164/563 (29.13%) | 64/364 (17.58%) | 194/563 (34.46%) | 46/938 (4.90%) |
| Kaggle DoRA lr5e-5 | raw | 174/563 (30.91%) | 227/371 (61.19%) | 69/563 (12.26%) | 22/938 (2.35%) |
| Kaggle DoRA lr5e-5 | FB | 174/563 (30.91%) | 227/371 (61.19%) | 78/563 (13.85%) | 22/938 (2.35%) |
| Kaggle DoRA lr5e-5 | FB+V | 165/563 (29.31%) | 91/371 (24.53%) | 171/563 (30.37%) | 22/938 (2.35%) |
| Evidence-order DoRA | raw | 192/563 (34.10%) | 306/374 (81.82%) | 18/563 (3.20%) | 3/938 (0.32%) |
| Evidence-order DoRA | FB | 192/563 (34.10%) | 306/374 (81.82%) | 19/563 (3.37%) | 3/938 (0.32%) |
| Evidence-order DoRA | FB+V | 179/563 (31.79%) | 126/374 (33.69%) | 139/563 (24.69%) | 3/938 (0.32%) |

`invalid_before_repair` records the generator's original invalid endings and therefore repeats across its repaired configurations. It must not be read as invalid outputs remaining after FB. Answerable denominators are 563 for all three DoRA runs; hidden denominators differ.

The latest raw model gets **192/563** supported correct answers, up from **174/563** in its parent and **176/563** in Modal. It falsely abstains on only **18/563**, versus **69/563** and **112/563**. However, hidden-evidence answers rise to **306/374 (81.82%)**, from **227/371 (61.19%)** and **117/364 (32.14%)**. The model has shifted toward answering regardless of available evidence. This is an interpretation of observed behavior, not a proven causal explanation of the training change.

With FB+V, the latest macro rises from **27.82% to 45.41%** (+17.58 pp), and unsupported answers fall from **306 to 126** on the same 374 hidden episodes. The cost is supported correct answers falling from **192 to 179** and false abstentions rising from **18 to 139**. Its repaired unsupported rate is still **33.69%**, versus **24.53%** for its parent's repaired configuration and **17.58%** for Modal.

## All 19 recorded AuditKIT outputs across the three DoRA checkpoints

Values remain on the original 0–1 scale; parentheses show each metric's contributing n. Higher is generally better except redundant tool calls. Lexical overlap and conditional metrics are proxies and must be interpreted with their cohorts.

### dora

| Metric/cohort | Modal DoRA | Kaggle DoRA lr5e-5 | Evidence-order DoRA | Latest − parent |
| --- | --- | --- | --- | --- |
| rag_stress: evidence_set_recall | 0.9449 (n=563) | 0.8970 (n=563) | 0.8028 (n=563) | -0.0941 |
| rag_stress: citation_support | 0.7691 (n=563) | 0.8615 (n=563) | 0.9663 (n=563) | +0.1048 |
| rag_stress: citation_precision | 0.5731 (n=563) | 0.5613 (n=563) | 0.5995 (n=563) | +0.0382 |
| rag_stress: citation_recall | 0.3643 (n=563) | 0.3806 (n=563) | 0.4057 (n=563) | +0.0251 |
| agent: tool_call_validity | 1.0000 (n=722) | 1.0000 (n=566) | 1.0000 (n=400) | +0.0000 |
| agent: redundant_tool_calls | 0.0249 (n=722) | 0.0156 (n=566) | 0.0033 (n=400) | -0.0123 |
| agent: agent_loop_detection | 0.9499 (n=938) | 0.9765 (n=938) | 0.9947 (n=938) | +0.0181 |
| abstention: abstention | 0.7321 (n=892) | 0.6725 (n=916) | 0.6524 (n=935) | -0.0201 |
| answerable_answered: lexical_groundedness | 0.9868 (n=433) | 0.9841 (n=485) | 0.9768 (n=544) | -0.0073 |
| answerable_answered: context_overlap | 0.6901 (n=433) | 0.6607 (n=485) | 0.6692 (n=544) | +0.0085 |
| answerable_answered: answer_overlap | 0.7941 (n=433) | 0.7070 (n=485) | 0.6607 (n=544) | -0.0463 |
| answerable_answered: f1_score | 0.7607 (n=433) | 0.6725 (n=485) | 0.6224 (n=544) | -0.0501 |
| answerable_context: context_coverage | 0.9607 (n=563) | 0.9281 (n=563) | 0.8773 (n=563) | -0.0508 |
| answerable_first_search: hit_rate@3 | 0.9549 (n=377) | 0.9572 (n=304) | 0.9810 (n=211) | +0.0238 |
| answerable_first_search: precision@3 | 0.5208 (n=377) | 0.5175 (n=304) | 0.5498 (n=211) | +0.0322 |
| answerable_first_search: recall@3 | 0.6516 (n=377) | 0.6590 (n=304) | 0.6077 (n=211) | -0.0513 |
| answerable_first_search: mrr@3 | 0.8700 (n=377) | 0.8690 (n=304) | 0.8799 (n=211) | +0.0110 |
| answerable_first_search: average_precision@3 | 0.7390 (n=377) | 0.7412 (n=304) | 0.7242 (n=211) | -0.0170 |
| answerable_first_search: ndcg@3 | 0.7890 (n=377) | 0.7904 (n=304) | 0.7817 (n=211) | -0.0087 |

### dora+FB

| Metric/cohort | Modal DoRA | Kaggle DoRA lr5e-5 | Evidence-order DoRA | Latest − parent |
| --- | --- | --- | --- | --- |
| rag_stress: evidence_set_recall | 0.9449 (n=563) | 0.8970 (n=563) | 0.8028 (n=563) | -0.0941 |
| rag_stress: citation_support | 0.7691 (n=563) | 0.8615 (n=563) | 0.9663 (n=563) | +0.1048 |
| rag_stress: citation_precision | 0.5731 (n=563) | 0.5613 (n=563) | 0.5995 (n=563) | +0.0382 |
| rag_stress: citation_recall | 0.3643 (n=563) | 0.3806 (n=563) | 0.4057 (n=563) | +0.0251 |
| agent: tool_call_validity | 1.0000 (n=722) | 1.0000 (n=566) | 1.0000 (n=400) | +0.0000 |
| agent: redundant_tool_calls | 0.0249 (n=722) | 0.0156 (n=566) | 0.0033 (n=400) | -0.0123 |
| agent: agent_loop_detection | 0.9499 (n=938) | 0.9765 (n=938) | 0.9947 (n=938) | +0.0181 |
| abstention: abstention | 0.7260 (n=938) | 0.6706 (n=938) | 0.6525 (n=938) | -0.0181 |
| answerable_answered: lexical_groundedness | 0.9868 (n=433) | 0.9841 (n=485) | 0.9768 (n=544) | -0.0073 |
| answerable_answered: context_overlap | 0.6901 (n=433) | 0.6607 (n=485) | 0.6692 (n=544) | +0.0085 |
| answerable_answered: answer_overlap | 0.7941 (n=433) | 0.7070 (n=485) | 0.6607 (n=544) | -0.0463 |
| answerable_answered: f1_score | 0.7607 (n=433) | 0.6725 (n=485) | 0.6224 (n=544) | -0.0501 |
| answerable_context: context_coverage | 0.9607 (n=563) | 0.9281 (n=563) | 0.8773 (n=563) | -0.0508 |
| answerable_first_search: hit_rate@3 | 0.9549 (n=377) | 0.9572 (n=304) | 0.9810 (n=211) | +0.0238 |
| answerable_first_search: precision@3 | 0.5208 (n=377) | 0.5175 (n=304) | 0.5498 (n=211) | +0.0322 |
| answerable_first_search: recall@3 | 0.6516 (n=377) | 0.6590 (n=304) | 0.6077 (n=211) | -0.0513 |
| answerable_first_search: mrr@3 | 0.8700 (n=377) | 0.8690 (n=304) | 0.8799 (n=211) | +0.0110 |
| answerable_first_search: average_precision@3 | 0.7390 (n=377) | 0.7412 (n=304) | 0.7242 (n=211) | -0.0170 |
| answerable_first_search: ndcg@3 | 0.7890 (n=377) | 0.7904 (n=304) | 0.7817 (n=211) | -0.0087 |

### dora+FB+V

| Metric/cohort | Modal DoRA | Kaggle DoRA lr5e-5 | Evidence-order DoRA | Latest − parent |
| --- | --- | --- | --- | --- |
| rag_stress: evidence_set_recall | 0.9449 (n=563) | 0.8970 (n=563) | 0.8028 (n=563) | -0.0941 |
| rag_stress: citation_support | 0.6554 (n=563) | 0.6963 (n=563) | 0.7531 (n=563) | +0.0568 |
| rag_stress: citation_precision | 0.5074 (n=563) | 0.4911 (n=563) | 0.5178 (n=563) | +0.0266 |
| rag_stress: citation_recall | 0.3068 (n=563) | 0.3254 (n=563) | 0.3391 (n=563) | +0.0138 |
| agent: tool_call_validity | 1.0000 (n=722) | 1.0000 (n=566) | 1.0000 (n=400) | +0.0000 |
| agent: redundant_tool_calls | 0.0249 (n=722) | 0.0156 (n=566) | 0.0033 (n=400) | -0.0123 |
| agent: agent_loop_detection | 0.9499 (n=938) | 0.9765 (n=938) | 0.9947 (n=938) | +0.0181 |
| abstention: abstention | 0.7164 (n=938) | 0.7196 (n=938) | 0.7164 (n=938) | -0.0032 |
| answerable_answered: lexical_groundedness | 0.9973 (n=369) | 0.9949 (n=392) | 0.9929 (n=424) | -0.0020 |
| answerable_answered: context_overlap | 0.7114 (n=369) | 0.7059 (n=392) | 0.6999 (n=424) | -0.0060 |
| answerable_answered: answer_overlap | 0.8375 (n=369) | 0.7814 (n=392) | 0.7650 (n=424) | -0.0164 |
| answerable_answered: f1_score | 0.7996 (n=369) | 0.7437 (n=392) | 0.7246 (n=424) | -0.0191 |
| answerable_context: context_coverage | 0.9607 (n=563) | 0.9281 (n=563) | 0.8773 (n=563) | -0.0508 |
| answerable_first_search: hit_rate@3 | 0.9549 (n=377) | 0.9572 (n=304) | 0.9810 (n=211) | +0.0238 |
| answerable_first_search: precision@3 | 0.5208 (n=377) | 0.5175 (n=304) | 0.5498 (n=211) | +0.0322 |
| answerable_first_search: recall@3 | 0.6516 (n=377) | 0.6590 (n=304) | 0.6077 (n=211) | -0.0513 |
| answerable_first_search: mrr@3 | 0.8700 (n=377) | 0.8690 (n=304) | 0.8799 (n=211) | +0.0110 |
| answerable_first_search: average_precision@3 | 0.7390 (n=377) | 0.7412 (n=304) | 0.7242 (n=211) | -0.0170 |
| answerable_first_search: ndcg@3 | 0.7890 (n=377) | 0.7904 (n=304) | 0.7817 (n=211) | -0.0087 |

The latest raw evidence-set recall is **0.8028**, below its parent's **0.8970** and Modal's **0.9449**, all on 563 answerable episodes. Its context coverage also falls to **0.8773**. First-search hit@3 rises to **0.9810**, but only **211** answerable episodes contribute, versus 304 in the parent and 377 in Modal; it does not establish improved overall retrieval. Conditional answer F1 falls to **0.6224** on **544** answered cases, from **0.6725** on 485 and **0.7607** on 433. The cohorts expand as the model answers more often.

Citation support (**0.9663**) and lexical groundedness (**0.9768**) are high, and loop-free behavior (**0.9947**) improves. These measures do not verify that all facts needed to answer the question were present or that the answer was correct. They therefore do not cancel the benchmark's evidence-missing regressions.

## Hosted-model historical comparisons: different common intersections

These published tables compare their original actors on a shared intersection. The latest DoRA was not included when those intersections were constructed. Do not insert its 937-retained score into either ranking or infer paired superiority from the numbers below.

### [azure_round5_comparison](azure_round5_comparison.md)

Recorded scope: r5 sealed test; common intersection; scorer v2; common n=876.

| System / prompt protocol | Macro | Supported correct | Unsupported / hidden | n |
| --- | --- | --- | --- | --- |
| main | 41.35% | 29.06% | 86/315 | 876 |
| candA | 47.50% | 33.87% | 98/315 | 876 |
| candF1 | 45.03% | 35.47% | 88/315 | 876 |
| azure_exact | 21.16% | 13.73% | 36/315 | 876 |
| azure_native | 28.65% | 39.22% | 62/315 | 876 |
| main+FB+V | 50.75% | 27.63% | 25/315 | 876 |
| candA+FB+V | 53.00% | 31.02% | 33/315 | 876 |
| candF1+FB+V | 55.67% | 32.44% | 26/315 | 876 |
| azure_exact+FB | 42.80% | 13.73% | 36/315 | 876 |
| azure_native+FB | 54.77% | 39.22% | 62/315 | 876 |

### [deepseek_comparison](deepseek_comparison.md)

Recorded scope: round-5 scorer-v2 shared intersection, all compared actors; common n=863.

| System / prompt protocol | Macro | Supported correct | Unsupported / hidden | n |
| --- | --- | --- | --- | --- |
| main | 41.52% | 29.21% | 82/305 | 863 |
| main+FB+V | 50.73% | 27.78% | 22/305 | 863 |
| candA | 47.23% | 33.87% | 96/305 | 863 |
| candA+FB+V | 52.98% | 31.18% | 31/305 | 863 |
| candF1 | 44.94% | 35.48% | 84/305 | 863 |
| candF1+FB+V | 55.54% | 32.62% | 25/305 | 863 |
| azure_exact | 21.03% | 13.80% | 36/305 | 863 |
| azure_exact+FB | 42.38% | 13.80% | 36/305 | 863 |
| azure_native | 28.99% | 39.43% | 59/305 | 863 |
| azure_native+FB | 54.90% | 39.43% | 59/305 | 863 |
| azure_deepseek_exact | 24.41% | 26.16% | 158/305 | 863 |
| azure_deepseek_exact+FB | 32.78% | 26.16% | 158/305 | 863 |
| azure_deepseek_native | 27.30% | 53.58% | 36/305 | 863 |
| azure_deepseek_native+FB | 61.10% | 53.58% | 36/305 | 863 |

Provider model names, exact/native protocol settings, availability failures and cost assumptions are documented in the linked original reports. This study uses those recorded values without new provider calls or claims about current hosted models.

## Earlier test suites: historical context, no latest-checkpoint measurement

### [round3_test](round3_test.md)

| System | Original three-component macro |
| --- | --- |
| main | 48.87% |
| candD | 55.50% |
| candA | 54.93% |

Latest evidence-order DoRA: **not measured on this suite**. Dataset composition and/or scoring differ from the round-5 headline; do not compare these macros as repeated measurements of one test.

### [round3_test_candA](round3_test_candA.md)

| System | Original three-component macro |
| --- | --- |
| main | 48.87% |
| candA | 54.93% |

Latest evidence-order DoRA: **not measured on this suite**. Dataset composition and/or scoring differ from the round-5 headline; do not compare these macros as repeated measurements of one test.

### [round4_test](round4_test.md)

| System | Original macro | Unsupported / hidden | Retained n |
| --- | --- | --- | --- |
| base | 1.61% | 112/187 | 449 |
| main | 35.68% | 40/187 | 449 |
| candA | 37.48% | 55/187 | 449 |
| candE1 | 37.70% | 33/187 | 449 |

Latest evidence-order DoRA: **not measured on this suite**. Dataset composition and/or scoring differ from the round-5 headline; do not compare these macros as repeated measurements of one test.

### [round4_wrappers](round4_wrappers.md)

| System | Original macro | Unsupported / hidden | Retained n |
| --- | --- | --- | --- |
| main | 35.68% | 40/187 | 449 |
| candA | 37.48% | 55/187 | 449 |
| FB(main) | 42.72% | 40/187 | 449 |
| FB(candA) | 41.33% | 55/187 | 449 |
| G(main, candA) | 44.55% | 25/187 | 449 |

Latest evidence-order DoRA: **not measured on this suite**. Dataset composition and/or scoring differ from the round-5 headline; do not compare these macros as repeated measurements of one test.

### [round5_r4test_wrappers](round5_r4test_wrappers.md)

| System | Original macro | Unsupported / hidden | Retained n |
| --- | --- | --- | --- |
| main | 34.97% | 39/181 | 443 |
| main+FB | 41.90% | 39/181 | 443 |
| main+FB+V | 46.08% | 16/181 | 443 |
| candA | 36.85% | 54/182 | 444 |
| candA+FB | 40.73% | 54/182 | 444 |
| candA+FB+V | 49.05% | 18/182 | 444 |
| candE1 | 35.60% | 32/184 | 446 |
| candE1+FB | 50.68% | 32/184 | 446 |
| candE1+FB+V | 55.50% | 11/184 | 446 |

Latest evidence-order DoRA: **not measured on this suite**. Dataset composition and/or scoring differ from the round-5 headline; do not compare these macros as repeated measurements of one test.

### [benchmarks_lenient](benchmarks_lenient.md)

| Earlier external suite | System | Original success | k/n |
| --- | --- | --- | --- |
| 2wiki | base | 1.80% | 3/166 |
| 2wiki | main | 56.60% | 94/166 |
| 2wiki | candA | 62.00% | 103/166 |
| 2wiki | control | 24.10% | 40/166 |
| musique | base | 1.80% | 3/168 |
| musique | main | 53.60% | 90/168 |
| musique | candA | 57.70% | 97/168 |
| musique | control | 25.00% | 42/168 |
| squad2 | base | 1.00% | 2/202 |
| squad2 | main | 47.50% | 96/202 |
| squad2 | candA | 55.90% | 113/202 |
| squad2 | control | 4.00% | 8/202 |
| rgb | base | 15.70% | 71/453 |
| rgb | main | 61.80% | 280/453 |
| rgb | candA | 66.00% | 299/453 |
| rgb | control | 53.00% | 240/453 |
| rgb_int | base | 7.00% | 10/142 |
| rgb_int | main | 2.80% | 4/142 |
| rgb_int | candA | 4.90% | 7/142 |
| rgb_int | control | 0.70% | 1/142 |

Latest checkpoint: not evaluated here. Strict and lenient parser results are not interchangeable.

### [benchmarks_strict](benchmarks_strict.md)

| Earlier external suite | System | Original success | k/n |
| --- | --- | --- | --- |
| 2wiki | base | 0.00% | 0/166 |
| 2wiki | main | 56.60% | 94/166 |
| 2wiki | candA | 62.00% | 103/166 |
| 2wiki | control | 24.10% | 40/166 |
| musique | base | 0.60% | 1/168 |
| musique | main | 53.60% | 90/168 |
| musique | candA | 57.70% | 97/168 |
| musique | control | 25.00% | 42/168 |
| squad2 | base | 0.00% | 0/202 |
| squad2 | main | 47.50% | 96/202 |
| squad2 | candA | 55.90% | 113/202 |
| squad2 | control | 4.00% | 8/202 |
| rgb | base | 3.50% | 16/453 |
| rgb | main | 61.60% | 279/453 |
| rgb | candA | 66.00% | 299/453 |
| rgb | control | 53.00% | 240/453 |
| rgb_int | base | 2.80% | 4/142 |
| rgb_int | main | 2.80% | 4/142 |
| rgb_int | candA | 4.90% | 7/142 |
| rgb_int | control | 0.70% | 1/142 |

Latest checkpoint: not evaluated here. Strict and lenient parser results are not interchangeable.

### Earlier task scorecards

| Original split/parser | System | Supported correct | Unsupported |
| --- | --- | --- | --- |
| [scorecard_dev_lenient](scorecard_dev_lenient.md) | base | 4.40% (9/205) | 89.80% (79/88) |
| [scorecard_dev_lenient](scorecard_dev_lenient.md) | main | 46.80% (96/205) | 11.40% (10/88) |
| [scorecard_dev_lenient](scorecard_dev_lenient.md) | control | 53.20% (109/205) | 28.40% (25/88) |
| [scorecard_test_lenient](scorecard_test_lenient.md) | base | 3.40% (7/205) | 78.00% (64/82) |
| [scorecard_test_lenient](scorecard_test_lenient.md) | main | 52.70% (108/205) | 9.80% (8/82) |
| [scorecard_test_lenient](scorecard_test_lenient.md) | control | 70.70% (145/205) | 26.80% (22/82) |
| [scorecard_test_strict](scorecard_test_strict.md) | base | 1.00% (2/205) | 2.40% (2/82) |
| [scorecard_test_strict](scorecard_test_strict.md) | main | 52.70% (108/205) | 9.80% (8/82) |
| [scorecard_test_strict](scorecard_test_strict.md) | control | 70.70% (145/205) | 25.60% (21/82) |
| [scorecard_test2_lenient](scorecard_test2_lenient.md) | base | 2.50% (5/199) | 87.80% (65/74) |
| [scorecard_test2_lenient](scorecard_test2_lenient.md) | main | 46.20% (92/199) | 4.10% (3/74) |
| [scorecard_test2_lenient](scorecard_test2_lenient.md) | control | 59.80% (119/199) | 18.90% (14/74) |
| [scorecard_test2_lenient](scorecard_test2_lenient.md) | candA | 52.80% (105/199) | 8.10% (6/74) |
| [scorecard_test2_strict](scorecard_test2_strict.md) | base | 0.00% (0/199) | 1.40% (1/74) |
| [scorecard_test2_strict](scorecard_test2_strict.md) | main | 46.20% (92/199) | 4.10% (3/74) |
| [scorecard_test2_strict](scorecard_test2_strict.md) | control | 59.80% (119/199) | 18.90% (14/74) |
| [scorecard_test2_strict](scorecard_test2_strict.md) | candA | 52.80% (105/199) | 8.10% (6/74) |

These original scorecards cover base/main/control/candA where present. Their evaluation questions, variant populations and parsers differ. They are retained as history rather than pooled with the latest test.

### Development and selection records

Development results must remain separate from final-test rankings. The following records preserve earlier selection decisions, failed DPO variants, wrapper checks and specialty tests:

| Historical record | Recorded selection/description |
| --- | --- |
| [audit_slice_dev.json](audit_slice_dev.json) | See linked record |
| [gate_baseline.json](gate_baseline.json) | See linked record |
| [retrieval_ceiling_dev.json](retrieval_ceiling_dev.json) | See linked record |
| [round2_selection.json](round2_selection.json) | See linked record |
| [round3_selection.json](round3_selection.json) | None |
| [round3_selection_interp.json](round3_selection_interp.json) | None |
| [round3_test.json](round3_test.json) | See linked record |
| [round3_test_candA.json](round3_test_candA.json) | See linked record |
| [round4_brightpro.json](round4_brightpro.json) | See linked record |
| [round4_dev_intervals.json](round4_dev_intervals.json) | See linked record |
| [round4_selection.json](round4_selection.json) | None |
| [round5_selection.json](round5_selection.json) | None |
| [round5_selection_r4cands_with_V.json](round5_selection_r4cands_with_V.json) | None |
| [round5_verifier.json](round5_verifier.json) | main-A |
| [round5_verifier_r4test.json](round5_verifier_r4test.json) | See linked record |
| [round5b_selection.json](round5b_selection.json) | candF1 |

The training and development metadata have now been downloaded and matched to the evaluated adapter hash. Full training used LR 2e-5, one epoch, 541 rows, 34 steps, and 646.8 seconds on four L4s. The development screen **failed**: unsupported answers rose from 17/73 to 39/73, and evidence-supported good outcomes fell from 64/159 to 54/159. See the [verified training snapshot](evidence_order_training_data.json) and [cost/efficiency report](dora_lora_cost_efficiency.md). This updates the earlier report's missing-metadata limitation; the final benchmark remains a negative result.

## Interpretation and next decision

1. **Keep the earlier stronger checkpoint for use.** Original Modal DoRA is strongest among the three DoRA versions by raw and repaired macro. Across historical adapters, main+FB+V remains the recorded selection under the original evidence-safety rule.
2. **Treat this latest continuation as a failed overall experiment.** It improves answer rate, format validity and some citation/tool proxies, but fails the central requirement to withhold answers when evidence is unavailable. Low false-abstention alone is not a successful result.
3. **Inspect the training and DEV artifacts before spending more GPU time.** Verify the real action mixture, three-search abstention eligibility, paired/replay token shares, parent hashes, and the same-episode development screen. The current evidence is insufficient to identify whether sampling, extra optimization, passage order, or another change caused the regression.
4. **Use controlled development ablations for any follow-up.** Compare equal-budget continuation with replay only against balanced sampling plus permutation, then choose on held-out development behavior. Fix the choice before a final benchmark; repeated tuning on these published test results does not yield a fresh generalization estimate.

## Data and provenance

Latest checkpoint: `/kaggle/input/notebooks/mlsingh1234/tranining-2/ats-evidence/dora_evidence_order/dora/adapter`.

Adapter weights SHA-256: `89552acaf005b840318d75feb98b9a4522110e84ad62a2415f98815db2b6b436`.

[Latest evaluation summary](evidence_order_evaluation/EVALUATION_SUMMARY.md) · [Latest full AuditKIT data](evidence_order_evaluation/auditkit/auditkit_metrics.json) · [Original Modal report](dora_evaluation_report.md) · [Previous DoRA comparison](dora_kaggle_lr5e5_comparison.md).

[Benchmark CSV](evidence_order_all_previous_comparison.csv), [AuditKIT CSV](evidence_order_all_previous_comparison_auditkit.csv), and [full precision source snapshots and hashes](evidence_order_all_previous_comparison.json) support reproducible inspection. All benchmark rate deltas use full-precision inputs.
