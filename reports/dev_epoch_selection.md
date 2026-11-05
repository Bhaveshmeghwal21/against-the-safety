# Dev split: epoch selection (headline episodes: open and support-masked)

| system | end-to-end good | supported_correct (answerable) | false_abstain (answerable) | correct_abstain (unanswerable) | unsupported answer (unanswerable) | invalid turn | mean searches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.013 (2/159) | 0.023 (2/86) | 0.000 (0/86) | 0.000 (0/73) | 0.877 (64/73) | 0.101 (16/159) | 0.96 |
| main, epoch 1 | 0.270 (43/159) | 0.395 (34/86) | 0.000 (0/86) | 0.123 (9/73) | 0.877 (64/73) | 0.000 (0/159) | 1.06 |
| main, epoch 2 (selected) | 0.673 (107/159) | 0.430 (37/86) | 0.233 (20/86) | 0.959 (70/73) | 0.027 (2/73) | 0.025 (4/159) | 2.18 |
| control, epoch 1 | 0.289 (46/159) | 0.535 (46/86) | 0.000 (0/86) | 0.000 (0/73) | 0.740 (54/73) | 0.119 (19/159) | 1.78 |
| control, epoch 2 | 0.277 (44/159) | 0.512 (44/86) | 0.000 (0/86) | 0.000 (0/73) | 0.164 (12/73) | 0.440 (70/159) | 1.94 |
