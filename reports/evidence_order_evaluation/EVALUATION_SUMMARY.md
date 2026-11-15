# DoRA checkpoint evaluation

Checkpoint: `/kaggle/input/notebooks/mlsingh1234/tranining-2/ats-evidence/dora_evidence_order/dora/adapter`

Adapter SHA-256: `89552acaf005b840318d75feb98b9a4522110e84ad62a2415f98815db2b6b436`

938 input benchmark episodes; scorer v2; fixed main-adapter verifier, wording A, threshold 0.5.

[Benchmark table](reports/benchmark_table.md) | [CSV](reports/benchmark_table.csv) | [Full benchmark metrics](reports/computed_round5.json)

| Configuration | Macro success |
| --- | ---: |
| dora | 27.82% |
| dora+FB | 28.43% |
| dora+FB+V | 45.41% |

Benchmark reports include per-component results, retained counts, unsupported hidden-evidence answers, supported correct answers and false refusals. Compare these tradeoffs together.

[Offline SQuAD scoring provenance](offline_scoring.json). QA scores use the SQuAD-v1 rules without a remote metric download.

[Recorded-v2 AuditKIT metrics](auditkit/auditkit_metrics.md) | [CSV](auditkit/auditkit_metrics.csv) | [Full metric cohorts](auditkit/auditkit_metrics.json)

AuditKIT scores all 938 original saved episodes with metric-specific cohorts. Its means have different denominators from benchmark macro success. No LLM judge or additional model generation is used for this audit.

This is a checkpoint-specific reproduction evaluation of the published suite. Use development data for hyperparameter and checkpoint selection. No model ranking, measured improvement, or cost comparison is inferred here.
