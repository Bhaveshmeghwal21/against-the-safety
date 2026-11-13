# Nimbus Home support suite (fictional, 24 tasks, exploratory)

**13 November 2026 correction:** This table preserves the original scorer and selection record. Evidence alternatives, binary answers, fixed eligibility and source offsets are corrected in the [dated remediation report](2026-11-13-evaluation-remediation.md). Later DPO prompts exposed held-out HotpotQA evidence. Annotation-based unsupported flags do not establish semantic hallucination; main remains submitted.


| metric | base | main | control |
| --- | ---: | ---: | ---: |
| ordinary (one document), supported_correct | 0.000 (0/6) | 0.500 (3/6) | 1.000 (6/6) |
| two documents, supported_correct | 0.000 (0/6) | 0.167 (1/6) | 0.500 (3/6) |
| dated change (current edition needed), supported_correct | 0.000 (0/6) | 0.500 (3/6) | 0.333 (2/6) |
| missing evidence, correct_abstain | 0.000 (0/6) | 1.000 (6/6) | 0.000 (0/6) |
| dated change: cited a superseded edition (lower is better) | 0.000 (0/6) | 0.667 (4/6) | 0.500 (3/6) |
| missing evidence: gave an unsupported answer (lower is better) | 0.000 (0/6) | 0.000 (0/6) | 0.167 (1/6) |
| all tasks: invalid turn (lower is better) | 1.000 (24/24) | 0.042 (1/24) | 0.375 (9/24) |
| AuditKIT freshness of retrieved documents (higher is better) | 0.667 | 0.813 | 0.786 |
| AuditKIT stale hit rate (lower is better) | 0.333 | 0.187 | 0.214 |
| mean searches per task | 0.040 | 1.790 | 1.750 |
