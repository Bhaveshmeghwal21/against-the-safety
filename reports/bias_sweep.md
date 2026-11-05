# Abstain-bias sweep (development split, lenient reading, headline episodes: 159)

The bias is added at decoding time to the logit of `abstain` after `<`; no weights change. Negative values make abstaining less likely.
Rule fixed before the sweep: highest development end-to-end good rate, ties toward the bias closest to 0.
Selected: **0** (good 107/159). The test split was not run at other biases, so the test result for the trained adapter (bias 0) stays the only test number.

| bias | end-to-end good | supported answer (answerable) | false abstain (answerable) | correct abstain (unanswerable) | answered anyway (unanswerable) | no valid final turn |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| -16 | 0.264 (42/159) | 0.488 (42/86) | 0.000 (0/86) | 0.000 (0/73) | 0.740 (54/73) | 0.182 (29/159) |
| -12 | 0.264 (42/159) | 0.488 (42/86) | 0.000 (0/86) | 0.000 (0/73) | 0.740 (54/73) | 0.182 (29/159) |
| -9 | 0.264 (42/159) | 0.488 (42/86) | 0.000 (0/86) | 0.000 (0/73) | 0.740 (54/73) | 0.182 (29/159) |
| -6 | 0.547 (87/159) | 0.477 (41/86) | 0.116 (10/86) | 0.630 (46/73) | 0.356 (26/73) | 0.025 (4/159) |
| -3 | 0.654 (104/159) | 0.430 (37/86) | 0.233 (20/86) | 0.918 (67/73) | 0.068 (5/73) | 0.025 (4/159) |
| -2 | 0.660 (105/159) | 0.430 (37/86) | 0.233 (20/86) | 0.932 (68/73) | 0.055 (4/73) | 0.025 (4/159) |
| -1 | 0.673 (107/159) | 0.430 (37/86) | 0.233 (20/86) | 0.959 (70/73) | 0.027 (2/73) | 0.025 (4/159) |
| 0 (selected) | 0.673 (107/159) | 0.430 (37/86) | 0.233 (20/86) | 0.959 (70/73) | 0.027 (2/73) | 0.025 (4/159) |
| +1 | 0.673 (107/159) | 0.430 (37/86) | 0.256 (22/86) | 0.959 (70/73) | 0.027 (2/73) | 0.025 (4/159) |
