# Numeric accuracy, first sealed test

Episodes whose gold answer is a single number (derived from the gold string; the datasets carry no numeric type). A numeric answer counts as correct when the numbers in the answer include the gold value. Intervals are 95% Wilson intervals. The subset is small, so read the counts.

| System | Numeric-answer episodes (questions) | Answered | Numerically correct | Numeric accuracy (95% interval) | Correct among answered (95% interval) |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 43 (16) | 34 | 24 | 55.8% (41 to 70) | 70.6% (54 to 83) |
| main | 43 (16) | 30 | 27 | 62.8% (48 to 76) | 90.0% (74 to 97) |
| main+FB+V (submitted) | 43 (16) | 28 | 26 | 60.5% (46 to 74) | 92.9% (77 to 98) |
| candF1+FB+V (alternative) | 43 (16) | 29 | 26 | 60.5% (46 to 74) | 89.7% (74 to 96) |
