# Single-request latency and throughput

One request at a time (batch size 1), greedy decoding, at most 160 new tokens per turn, Tesla T4, model in 4-bit, Hugging Face `generate` with no serving-engine tuning. 150 requests from the three r5t suites (seed 61), 3 untimed warm-up requests per arm. Percentiles and requests per second use AuditKIT's `LatencyStats` and `Throughput`. The base arm also applies the fail-closed rule and the code checks of the self-check (cited passage was shown, answer text is in it), which cost no model call; it skips only the model self-check. The pre-registration is [here](../research/2026-11-16-judge-latency-preregistration.md).

| System | Requests | p50 (s) | p95 (s) | p99 (s) | Mean (s) | Max (s) | Requests per second | Mean output tokens | Answers shown after the repairs | Peak GPU memory (GB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiny Aya base, same loop, fail-closed rule and code checks, no model self-check | 150 | 8.77 | 19.44 | 23.44 | 10.32 | 24.81 | 0.097 | 123 | 7 | 3.63 |
| Submitted: main adapter, fail-closed rule, self-check | 150 | 16.27 | 23.84 | 26.17 | 15.51 | 27.89 | 0.064 | 95 | 56 | 3.98 |

## Where the time goes (mean per request)

| System | Generation (s) | Search (s) | Self-check (s) | Other (s) | Estimated cost per 1,000 requests on a busy T4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tiny Aya base, same loop, fail-closed rule and code checks, no model self-check | 10.25 | 0.01 | 0.00 | 0.05 | $1.51 |
| Submitted: main adapter, fail-closed rule, self-check | 15.21 | 0.02 | 0.28 | 0.00 | $2.27 |

## By suite (p50 and p95, seconds)

| System | r5t_squad | r5t_hotpot | r5t_rgb |
| --- | ---: | ---: | ---: |
| Tiny Aya base, same loop, fail-closed rule and code checks, no model self-check | 13.71 and 22.27 | 8.28 and 12.31 | 8.14 and 17.35 |
| Submitted: main adapter, fail-closed rule, self-check | 15.45 and 25.79 | 18.27 and 23.84 | 12.08 and 19.48 |

Cost uses $0.526 per T4 hour (`reports/azure_roi.md`) and assumes the GPU is busy all the time. These are single-request figures on a research stack, measured on a shared Kaggle host. A serving engine with batching would lower the cost per request and could raise the latency of an individual request.
