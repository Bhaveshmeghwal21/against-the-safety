# Round 5 priority 1: the answer checker on development data

**13 November 2026 correction:** This table preserves the original scorer and selection record. Evidence alternatives, binary answers, fixed eligibility and source offsets are corrected in the [dated remediation report](2026-11-13-evaluation-remediation.md). Later DPO prompts exposed held-out HotpotQA evidence. Annotation-based unsupported flags do not establish semantic hallucination; main remains submitted.


Cases: {'unsupported': 20, 'supported_correct': 76, 'wrong_answer': 87}. Gate: block at least 40% of unsupported answers and at most 10% of supported-correct answers (deterministic checks plus the model check).

| configuration | unsupported blocked | supported correct blocked | wrong blocked | model only: unsupported / supported / wrong | first token is a candidate | passes |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| base-A | 60% | 24% | 53% | 60% / 24% / 48% | 100% | False |
| base-B | 75% | 21% | 62% | 75% / 21% / 62% | 100% | False |
| main-A | 70% | 8% | 47% | 70% / 8% / 44% | 100% | True |
| main-B | 100% | 87% | 99% | 100% / 87% / 99% | 100% | False |

Selected: **main-A**.

Main on development data (scorer v2, fail-closed applied):

| | macro | supported correct | unsupported / hidden | false refusals | HotpotQA unsupported |
| --- | ---: | ---: | ---: | ---: | ---: |
| FB | 0.515 | 0.243 | 18/131 | 0.471 | 2 |
| FB + V | 0.538 | 0.228 | 5/131 | 0.635 | 1 |
