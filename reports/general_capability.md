# General-capability retention check

Zero-shot multiple choice, plain text, no chat template, 4-bit, one run per system. The same prompt goes to all three systems; `base` is Tiny Aya Global with the adapters switched off. Accuracy in percent, 95% bootstrap interval over questions in brackets. Absolute scores are not comparable with published ones (other prompts and protocols); only differences within this table carry meaning. The pre-registration is [here](../research/2026-11-16-general-capability-preregistration.md).

| Benchmark (language, role) | Questions | Base | Main (submitted) | candF1 (alternative) |
| --- | ---: | ---: | ---: | ---: |
| MMLU (English) | 14,042 | 54.5 [53.6, 55.3] | 54.1 [53.2, 54.9] | 54.2 [53.3, 55.0] |
| ARC-Challenge (English) | 1,172 | 73.5 [71.0, 75.9] | 72.7 [70.2, 75.1] | 73.0 [70.6, 75.4] |
| ARC-Easy (English) | 2,376 | 85.4 [84.0, 86.8] | 85.5 [84.0, 86.9] | 85.6 [84.2, 87.0] |
| TruthfulQA MC1 (English) | 817 | 51.8 [48.3, 55.2] | 48.2 [44.9, 51.7] | 50.1 [46.6, 53.5] |
| Global-MMLU-Lite, English | 400 | 60.5 [55.5, 65.2] | 58.8 [53.8, 63.5] | 59.8 [55.0, 64.5] |
| Global-MMLU-Lite, Hindi | 400 | 41.0 [36.0, 45.8] | 42.0 [37.2, 46.8] | 40.5 [35.8, 45.2] |
| Global-MMLU-Lite, Spanish | 400 | 57.2 [52.5, 62.0] | 56.2 [51.5, 61.0] | 56.2 [51.5, 61.0] |

## Difference from the base model (points, paired 95% interval)

| Benchmark | Main minus base | candF1 minus base |
| --- | ---: | ---: |
| MMLU (English) | -0.4 [-0.8, +0.1] | -0.3 [-0.7, +0.2] |
| ARC-Challenge (English) | -0.8 [-2.0, +0.4] | -0.4 [-1.6, +0.8] |
| ARC-Easy (English) | +0.0 [-0.6, +0.7] | +0.2 [-0.5, +0.8] |
| TruthfulQA MC1 (English) | -3.5 [-5.6, -1.6] | -1.7 [-3.5, +0.1] |
| Global-MMLU-Lite, English | -1.7 [-4.5, +0.8] | -0.7 [-3.5, +1.8] |
| Global-MMLU-Lite, Hindi | +1.0 [-1.5, +3.7] | -0.5 [-3.5, +2.5] |
| Global-MMLU-Lite, Spanish | -1.0 [-3.0, +1.0] | -1.0 [-3.5, +1.5] |

## Registered rule: the lower end of the interval must be at least -3.0 points

| Benchmark | Main: lower end | Main | candF1: lower end | candF1 |
| --- | ---: | --- | ---: | --- |
| MMLU (English) | -0.8 | retained | -0.7 | retained |
| ARC-Challenge (English) | -2.0 | retained | -1.6 | retained |
| ARC-Easy (English) | -0.6 | retained | -0.5 | retained |
