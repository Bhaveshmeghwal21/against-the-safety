# Third-round selection on the development set (rule fixed in plan section 24)

Eligible: (G1) pooled unsupported-answer rate at most 0.03 above main's; (G2) at most 11 of 73 unsupported answers on D1; (G3) pooled invalid-turn rate at most 0.03 above main's.
Winner: the eligible candidate with the highest macro average of the four components (ties: first in the preference list). It replaces main only if its macro is at least main's plus 0.05 and its D1 score is at least main's minus 0.03.

| system | D1 HotpotQA dev ends well | D2 single-passage ends well | D3 compound ends well | D4 prompt variants ends well | macro | pooled unsupported answers | pooled invalid | G1 | G2 | G3 | eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| main | 0.673 (107/159) | 0.325 (41/126) | 0.510 (51/100) | 0.497 (96/193) | 0.501 | 17/282 | 67/578 | - | - | - | yes |
| candC | 0.566 (90/159) | 0.762 (96/126) | 0.600 (60/100) | 0.699 (135/193) | 0.657 | 70/282 | 4/578 | no | no | yes | no |
| candD | 0.629 (100/159) | 0.730 (92/126) | 0.570 (57/100) | 0.601 (116/193) | 0.633 | 72/282 | 4/578 | no | yes | yes | no |

Detail per component (supported correct / false abstain among answerable; correct abstain / unsupported answers among hidden-evidence; invalid turns):

- main, D1 HotpotQA dev: supported 37/86, false abstain 20/86, correct abstain 70/73, unsupported 2/73, invalid 4/159
- main, D2 single-passage: supported 3/67, false abstain 47/67, correct abstain 38/59, unsupported 1/59, invalid 36/126
- main, D3 compound: supported 1/38, false abstain 26/38, correct abstain 50/62, unsupported 1/62, invalid 13/100
- main, D4 prompt variants: supported 28/105, false abstain 31/105, correct abstain 68/88, unsupported 13/88, invalid 14/193
- candC, D1 HotpotQA dev: supported 32/86, false abstain 16/86, correct abstain 58/73, unsupported 15/73, invalid 0/159
- candC, D2 single-passage: supported 43/67, false abstain 20/67, correct abstain 53/59, unsupported 4/59, invalid 2/126
- candC, D3 compound: supported 18/38, false abstain 0/38, correct abstain 42/62, unsupported 19/62, invalid 2/100
- candC, D4 prompt variants: supported 79/105, false abstain 1/105, correct abstain 56/88, unsupported 32/88, invalid 0/193
- candD, D1 HotpotQA dev: supported 35/86, false abstain 19/86, correct abstain 65/73, unsupported 8/73, invalid 0/159
- candD, D2 single-passage: supported 38/67, false abstain 23/67, correct abstain 54/59, unsupported 3/59, invalid 2/126
- candD, D3 compound: supported 16/38, false abstain 3/38, correct abstain 41/62, unsupported 20/62, invalid 2/100
- candD, D4 prompt variants: supported 69/105, false abstain 3/105, correct abstain 47/88, unsupported 41/88, invalid 0/193

**Winner: none eligible.** Macro n/a against main's 0.501; D1 n/a against 0.673.
**Headline model after development: main.** Main stays the headline model; the winner, if any, is published as a variant.
If there is a winner, main and the winner are evaluated once on the fresh test set (r3t_*).
