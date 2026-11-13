# Third-round selection on the development set (rule fixed in plan section 24)

**13 November 2026 correction:** This table preserves the original scorer and selection record. Evidence alternatives, binary answers, fixed eligibility and source offsets are corrected in the [dated remediation report](2026-11-13-evaluation-remediation.md). Later DPO prompts exposed held-out HotpotQA evidence. Annotation-based unsupported flags do not establish semantic hallucination; main remains submitted.


Eligible: (G1) pooled unsupported-answer rate at most 0.03 above main's; (G2) at most 11 of 73 unsupported answers on D1; (G3) pooled invalid-turn rate at most 0.03 above main's.
Winner: the eligible candidate with the highest macro average of the four components (ties: first in the preference list). It replaces main only if its macro is at least main's plus 0.05 and its D1 score is at least main's minus 0.03.

| system | D1 HotpotQA dev ends well | D2 single-passage ends well | D3 compound ends well | D4 prompt variants ends well | macro | pooled unsupported answers | pooled invalid | G1 | G2 | G3 | eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| main | 0.673 (107/159) | 0.325 (41/126) | 0.510 (51/100) | 0.497 (96/193) | 0.501 | 17/282 | 67/578 | - | - | - | yes |
| candD_a20 | 0.642 (102/159) | 0.429 (54/126) | 0.520 (52/100) | 0.513 (99/193) | 0.526 | 34/282 | 39/578 | no | yes | yes | no |
| candD_a35 | 0.648 (103/159) | 0.500 (63/126) | 0.540 (54/100) | 0.539 (104/193) | 0.557 | 39/282 | 20/578 | no | yes | yes | no |
| candD_a50 | 0.635 (101/159) | 0.587 (74/126) | 0.540 (54/100) | 0.575 (111/193) | 0.584 | 46/282 | 12/578 | no | yes | yes | no |

Detail per component (supported correct / false abstain among answerable; correct abstain / unsupported answers among hidden-evidence; invalid turns):

- main, D1 HotpotQA dev: supported 37/86, false abstain 20/86, correct abstain 70/73, unsupported 2/73, invalid 4/159
- main, D2 single-passage: supported 3/67, false abstain 47/67, correct abstain 38/59, unsupported 1/59, invalid 36/126
- main, D3 compound: supported 1/38, false abstain 26/38, correct abstain 50/62, unsupported 1/62, invalid 13/100
- main, D4 prompt variants: supported 28/105, false abstain 31/105, correct abstain 68/88, unsupported 13/88, invalid 14/193
- candD_a20, D1 HotpotQA dev: supported 35/86, false abstain 21/86, correct abstain 67/73, unsupported 4/73, invalid 4/159
- candD_a20, D2 single-passage: supported 11/67, false abstain 45/67, correct abstain 43/59, unsupported 4/59, invalid 22/126
- candD_a20, D3 compound: supported 1/38, false abstain 19/38, correct abstain 51/62, unsupported 6/62, invalid 8/100
- candD_a20, D4 prompt variants: supported 33/105, false abstain 29/105, correct abstain 66/88, unsupported 20/88, invalid 5/193
- candD_a35, D1 HotpotQA dev: supported 35/86, false abstain 23/86, correct abstain 68/73, unsupported 5/73, invalid 2/159
- candD_a35, D2 single-passage: supported 17/67, false abstain 45/67, correct abstain 46/59, unsupported 3/59, invalid 13/126
- candD_a35, D3 compound: supported 1/38, false abstain 16/38, correct abstain 53/62, unsupported 7/62, invalid 2/100
- candD_a35, D4 prompt variants: supported 42/105, false abstain 24/105, correct abstain 62/88, unsupported 24/88, invalid 3/193
- candD_a50, D1 HotpotQA dev: supported 33/86, false abstain 24/86, correct abstain 68/73, unsupported 5/73, invalid 3/159
- candD_a50, D2 single-passage: supported 24/67, false abstain 40/67, correct abstain 50/59, unsupported 3/59, invalid 7/126
- candD_a50, D3 compound: supported 4/38, false abstain 14/38, correct abstain 50/62, unsupported 10/62, invalid 2/100
- candD_a50, D4 prompt variants: supported 51/105, false abstain 11/105, correct abstain 60/88, unsupported 28/88, invalid 0/193

**Winner: none eligible.** Macro n/a against main's 0.501; D1 n/a against 0.673.
**Headline model after development: main.** Main stays the headline model; the winner, if any, is published as a variant.
If there is a winner, main and the winner are evaluated once on the fresh test set (r3t_*).
