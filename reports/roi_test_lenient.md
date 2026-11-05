# ROI (test, headline episodes: open and support-masked)

Reference price 0.35 USD per T4 GPU-hour (range 0.16 to 0.59). Seconds are amortized wall time on a shared, batched T4; they depend on batch size and are for comparison between rows, not a latency promise.

| system | seconds / episode | model calls / episode | input tokens / episode | output tokens / episode | end-to-end good | supported_correct (answerable) | episodes needing review per 1,000 | GPU cost per 1,000 episodes (USD) | GPU cost per good outcome (USD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 2.76 | 1.92 | 1867 | 106 | 0.020 | 0.012 | 980 | 0.268 | 0.01350 |
| main | 5.04 | 3.16 | 3805 | 112 | 0.583 | 0.442 | 417 | 0.490 | 0.00084 |
| control | 4.38 | 2.95 | 3497 | 98 | 0.404 | 0.709 | 596 | 0.426 | 0.00105 |

## Training cost

| run | rows trained | processed tokens | GPU-hours | actual cost (Kaggle free quota) | cost at reference price (USD) | peak GPU memory (GB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| main | 1611 | 2,330,132 | 3.90 | 0.00 | 1.36 | 5.83 |
| control | 1697 | 2,272,531 | 3.59 | 0.00 | 1.26 | 5.74 |

## Break-even

The tuned model produces 563 more end-to-end good outcomes per 1,000 episodes than the base model. Training the main adapter costs 1.36 USD at the reference price, which equals the GPU cost of about 2,782 tuned-model episodes. Whether it pays back depends on what a bad outcome costs the operator, which this project did not measure. Illustration only, with an assumed cost per avoided bad outcome:

| assumed cost of one bad outcome (USD) | queries needed to recover the training cost |
| ---: | ---: |
| 0.10 | 24.2 |
| 1.00 | 2.4 |
| 5.00 | 0.5 |
