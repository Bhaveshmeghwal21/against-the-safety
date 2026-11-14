# README LoRA results versus all completed DoRA runs

Prepared 14 November 2026 from local completed records. No new training or inference.

**Original Modal DoRA improves on raw main LoRA, but no DoRA checkpoint beats the best recorded LoRA configuration.** Kaggle lr5e-5 and the latest evidence-order continuation are below main, candA and candF1 on the shared benchmark macro. The comparison must use the same test suite and the same output-repair setting.

## Match the README's result to the correct test

| README result | Original population/metric | Matching DoRA evaluation? |
| --- | --- | --- |
| First-round main 58.3%, control 40.4% | 151 headline HotpotQA episodes, out of 287 total | No |
| Fresh test2 main 60.1%, candA 65.7%, control 28.0% | 143 headline HotpotQA episodes, out of 273 total | No |
| External main 49.9%, candA 54.7%, control 29.3% | 1,131 pooled episodes, five older benchmark suites | No |
| Round-3 main 48.9%, candD 55.5%, candA 54.9% | 739-episode fresh suite, macro of six sets | No |
| Round-5 main+FB+V 51.7%, candF1+FB+V 57.0% | 938 inputs; four-component macro after scorer-v2 exclusions | Yes, all three DoRA runs |

A 58.3% HotpotQA headline score cannot be subtracted from a 27.82% four-component macro to estimate a DoRA effect. Even SQuAD/RGB names recur across different question samples, variants and scoring versions. Control and the round-3 patch lack matching round-5 rows; missing results are not zero scores.

## Direct comparison on the same 938 input episodes

FB = fail-closed finalization. V = the fixed main-LoRA verifier with wording A, threshold 0.5. The repaired DoRA systems use that LoRA verifier; they are not evaluations of a DoRA-trained verifier.

| Model | Raw macro ↑ | +FB macro ↑ | +FB+V macro ↑ |
| --- | --- | --- | --- |
| Main LoRA | 41.43% | 47.37% | 51.74% |
| candA LoRA | 47.34% | 49.38% | 54.10% |
| candF1 LoRA | 44.93% | 52.15% | 56.97% |
| Modal DoRA | 44.71% | 48.18% | 50.09% |
| Kaggle DoRA lr5e-5 | 34.00% | 35.91% | 47.34% |
| Evidence-order DoRA | 27.82% | 28.43% | 45.41% |

### Each DoRA checkpoint minus each previous LoRA checkpoint

Macro differences below are percentage points, DoRA minus LoRA. Positive favors DoRA. Compare raw with raw and repaired with the same repair configuration.

| DoRA checkpoint | LoRA reference | Raw Δ pp | +FB Δ pp | +FB+V Δ pp |
| --- | --- | --- | --- | --- |
| Modal DoRA | Main LoRA | +3.28 | +0.81 | -1.65 |
| Modal DoRA | candA LoRA | -2.63 | -1.21 | -4.01 |
| Modal DoRA | candF1 LoRA | -0.22 | -3.98 | -6.88 |
| Kaggle DoRA lr5e-5 | Main LoRA | -7.43 | -11.46 | -4.40 |
| Kaggle DoRA lr5e-5 | candA LoRA | -13.34 | -13.47 | -6.76 |
| Kaggle DoRA lr5e-5 | candF1 LoRA | -10.93 | -16.24 | -9.63 |
| Evidence-order DoRA | Main LoRA | -13.61 | -18.94 | -6.34 |
| Evidence-order DoRA | candA LoRA | -19.52 | -20.95 | -8.69 |
| Evidence-order DoRA | candF1 LoRA | -17.10 | -23.72 | -11.56 |

### Answer quality and missing-evidence behavior

**raw**

| Model | Supported correct / answerable ↑ | Unsupported / hidden ↓ | False abstain / answerable ↓ |
| --- | --- | --- | --- |
| Main LoRA | 163/563 (28.95%) | 108/364 (29.67%) | 125/563 (22.20%) |
| candA LoRA | 190/563 (33.75%) | 129/365 (35.34%) | 90/563 (15.99%) |
| candF1 LoRA | 199/563 (35.35%) | 114/364 (31.32%) | 89/563 (15.81%) |
| Modal DoRA | 176/563 (31.26%) | 117/364 (32.14%) | 112/563 (19.89%) |
| Kaggle DoRA lr5e-5 | 174/563 (30.91%) | 227/371 (61.19%) | 69/563 (12.26%) |
| Evidence-order DoRA | 192/563 (34.10%) | 306/374 (81.82%) | 18/563 (3.20%) |

**FB+V**

| Model | Supported correct / answerable ↑ | Unsupported / hidden ↓ | False abstain / answerable ↓ |
| --- | --- | --- | --- |
| Main LoRA | 155/563 (27.53%) | 37/364 (10.16%) | 220/563 (39.08%) |
| candA LoRA | 174/563 (30.91%) | 45/365 (12.33%) | 182/563 (32.33%) |
| candF1 LoRA | 182/563 (32.33%) | 34/364 (9.34%) | 213/563 (37.83%) |
| Modal DoRA | 164/563 (29.13%) | 64/364 (17.58%) | 194/563 (34.46%) |
| Kaggle DoRA lr5e-5 | 165/563 (29.31%) | 91/371 (24.53%) | 171/563 (30.37%) |
| Evidence-order DoRA | 179/563 (31.79%) | 126/374 (33.69%) | 139/563 (24.69%) |

On raw outcomes, original Modal DoRA exceeds main LoRA by **3.28 pp macro**, while unsupported answers increase from **108/364 to 117/364**. With FB+V, Modal DoRA is **1.65 pp below main**, with **64/364** unsupported answers versus main's **37/364**. The later DoRA runs move further toward answering: the latest answers **306/374** hidden-evidence episodes. It is not a better operating point for the repository's evidence-grounding objective.

### Per-component metrics, raw and repaired

| Model | Setting | SQuAD single | SQuAD two-part | HotpotQA | RGB |
| --- | --- | --- | --- | --- | --- |
| Main LoRA | raw | 54.55% | 22.42% | 52.44% | 36.30% |
| candA LoRA | raw | 63.64% | 25.00% | 62.20% | 38.53% |
| candF1 LoRA | raw | 54.09% | 33.33% | 54.32% | 37.96% |
| Modal DoRA | raw | 60.91% | 26.67% | 53.66% | 37.61% |
| Kaggle DoRA lr5e-5 | raw | 48.18% | 13.94% | 43.90% | 29.98% |
| Evidence-order DoRA | raw | 39.55% | 4.85% | 39.02% | 27.87% |
| Main LoRA | FB+V | 59.09% | 39.39% | 56.10% | 52.39% |
| candA LoRA | FB+V | 61.36% | 39.02% | 63.41% | 52.60% |
| candF1 LoRA | FB+V | 61.36% | 52.73% | 61.73% | 52.06% |
| Modal DoRA | FB+V | 60.45% | 38.79% | 52.44% | 48.70% |
| Kaggle DoRA lr5e-5 | FB+V | 57.27% | 33.33% | 51.22% | 47.54% |
| Evidence-order DoRA | FB+V | 50.00% | 29.09% | 54.88% | 47.66% |

## The README's original LoRA metrics, retained as a separate panel

These first-round HotpotQA metrics are quoted from the machine-readable source linked by the README. They are useful history, but no completed DoRA evaluation on this same split has been downloaded. The smaller cohort and older audit adapter prevent a method ranking against the newer round-5 AuditKIT means.

| Original HotpotQA metric | Main LoRA | Control LoRA |
| --- | --- | --- |
| Supported correct answer ↑ | 0.527 (108/205) | 0.707 (145/205) |
| Unsupported answer ↓ | 0.098 (8/82) | 0.268 (22/82) |
| False abstain ↓ | 0.190 (39/205) | 0.000 (0/205) |
| AuditKIT evidence-set recall ↑ | 0.727 | 0.824 |
| AuditKIT citation support ↑ | 0.771 | 0.941 |
| Answer F1 when answered ↑ | 0.643 | 0.690 |
| Tool-call validity ↑ | 1.000 | 1.000 |
| Loop-free ↑ | 0.934 | 0.784 |
| Redundant calls ↓ | 0.029 | 0.088 |

For orientation only, the same-named metrics from the three raw DoRA evaluations are shown below with their actual cohorts. This is a **different test and audit population**; no cross-panel delta is computed.

| Round-5 AuditKIT metric | Modal DoRA | Kaggle DoRA lr5e-5 | Evidence-order DoRA |
| --- | --- | --- | --- |
| evidence_set_recall | 0.9449 (n=563) | 0.8970 (n=563) | 0.8028 (n=563) |
| citation_support | 0.7691 (n=563) | 0.8615 (n=563) | 0.9663 (n=563) |
| f1_score | 0.7607 (n=433) | 0.6725 (n=485) | 0.6224 (n=544) |
| tool_call_validity | 1.0000 (n=722) | 1.0000 (n=566) | 1.0000 (n=400) |
| agent_loop_detection | 0.9499 (n=938) | 0.9765 (n=938) | 0.9947 (n=938) |
| redundant_tool_calls | 0.0249 (n=722) | 0.0156 (n=566) | 0.0033 (n=400) |

### README five-external-benchmark panel

| Older external benchmark | Main LoRA | candA LoRA | Control LoRA | All three DoRA |
| --- | --- | --- | --- | --- |
| 2wiki | 0.566 | 0.620 | 0.241 | Not measured |
| musique | 0.536 | 0.577 | 0.250 | Not measured |
| squad2 | 0.475 | 0.559 | 0.040 | Not measured |
| rgb | 0.618 | 0.660 | 0.530 | Not measured |
| rgb_int | 0.028 | 0.049 | 0.007 | Not measured |
| All five pooled (1,131) | 0.499 | 0.547 | 0.293 | Not measured |

The old pooled 0.499/0.547 scores are weighted by those five suites' episode counts. The current macro weights four round-5 components equally. Their numerical values cannot be used to conclude LoRA or DoRA won on the old external suite.

## What the available results establish

- **Best raw adapter macro on the shared suite:** candA LoRA, 47.34%.
- **Best repaired adapter macro:** candF1 LoRA+FB+V, 56.97%; it failed the original unsupported-answer non-inferiority rule, so it remains a published alternative.
- **Historical selected system:** main LoRA+FB+V, 51.74%, under the registered selection rule.
- **Best of your DoRA runs:** original Modal DoRA, 44.71% raw and 50.09% with FB+V.
- **Latest evidence-order DoRA:** 27.82% raw and 45.41% repaired; its increased answer rate does not compensate for answering without evidence.

These are checkpoint comparisons, not proof that LoRA is inherently superior to DoRA. Data, continuation, LR, runtime/hardware and optimization histories vary. Even original main LoRA versus Modal DoRA was not a fully controlled multi-seed method study; equal direction rank also leaves DoRA with extra magnitude parameters.

All direct rows use the same 938 input episodes, but retained counts and hidden-evidence denominators vary by checkpoint (scorer-v2 exclusions). Rankings and deltas are descriptive; this report adds no paired intervals or common-intersection rescore. Control and patch require new, matching evaluations to enter the current suite's ranking.

## Sources and downloadable tables

[README](../README.md) · [Original LoRA scorecard](scorecard_test_lenient.md) · [External LoRA report](benchmarks_lenient.md) · [Round-5 LoRA report](round5b_final.md) · [Full latest study](evidence_order_all_previous_comparison.md).

[DoRA-minus-LoRA macro deltas CSV](readme_lora_vs_all_dora.csv) · [Full-precision source snapshot](readme_lora_vs_all_dora.json).
