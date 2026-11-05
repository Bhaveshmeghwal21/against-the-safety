# Second-round selection on the development split (rule fixed in plan section 20)

Eligible: unsupported answers on unanswerable headline episodes at most 11 of 73, and at most 10% of headline episodes without a valid final turn.
Best candidate: the eligible one with the higher end-to-end good rate (ties: higher supported-correct rate, then candA).
Second-round model: the best candidate if its end-to-end good rate is at least main's, otherwise main.

| system | end-to-end good (headline) | supported correct (answerable) | false abstain (answerable) | correct abstain (unanswerable headline) | unsupported answers (unanswerable headline) | no valid final turn (headline) | eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| main | 0.673 (107/159) | 0.468 (96/205) | 0.176 (36/205) | 0.959 (70/73) | 2/73 | 4/159 | yes |
| candA | 0.629 (100/159) | 0.517 (106/205) | 0.088 (18/205) | 0.849 (62/73) | 9/73 | 4/159 | yes |
| candB | 0.604 (96/159) | 0.473 (97/205) | 0.166 (34/205) | 0.877 (64/73) | 5/73 | 4/159 | yes |

**Best candidate: candA**.
**Second-round model: main.** Both candA and main are evaluated on test2.
