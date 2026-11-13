# Round 4 development selection

**13 November 2026 correction:** This table preserves the original scorer and selection record. Evidence alternatives, binary answers, fixed eligibility and source offsets are corrected in the [dated remediation report](2026-11-13-evaluation-remediation.md). Later DPO prompts exposed held-out HotpotQA evidence. Annotation-based unsupported flags do not establish semantic hallucination; main remains submitted.


Primary success requires the original correctness check and all required evidence aspects cited.

| system | macro | single | compound | Hotpot | unsupported / hidden | invalid | eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| main | 0.485 | 0.486 | 0.297 | 0.673 | 18/132 | 0.134 | reference |
| candA | 0.500 | 0.590 | 0.279 | 0.629 | 20/132 | 0.053 | reference |
| candE1 | 0.468 | 0.429 | 0.351 | 0.623 | 17/132 | 0.249 | False |
| candE2 | 0.497 | 0.538 | 0.324 | 0.629 | 21/132 | 0.150 | False |
| candF1 | 0.552 | 0.567 | 0.486 | 0.604 | 13/132 | 0.112 | False |
| candF2 | 0.538 | 0.595 | 0.396 | 0.623 | 22/132 | 0.090 | False |

Selected for frozen holdout: **none; diagnostic candE1 only**.

Development selection only; release recommendation requires the frozen final holdout.

Post-evaluation trace audit: these are automatic gold-mask and citation-coverage scores. Confirmed alternative-evidence masking errors, source-question inconsistencies and semantic scoring limitations are documented in the [dated results report](../research/2026-11-06-round4-results.md). Raw counts and the frozen selection outcome have not been changed.
