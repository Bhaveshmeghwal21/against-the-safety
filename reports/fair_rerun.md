# Format-aware hosted comparison

Shared eligible episodes; SQuAD resampled by article, other suites by question; seed 47, 3000 draws. The adapters learned the answer format from training; hosted models were explicitly told it in the fair run. The original round-five test was already opened, so this is a retrospective system comparison, not a fresh blind test. Replay of saved local searches matched current results, but cannot establish equivalence for unseen queries.

Shared eligible episodes: 925. Provider-blocked or missing cases are excluded.

| System | Macro success | Correct and cited, of answerable | Answers on hidden evidence | Answered at all | Correct among answered | Wrong or unsupported shown / all cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| main+FB+V | 58.4% | 46.8% | 42/372 | 41.3% | 67.8% | 13.3% |
| candF1+FB+V | 64.5% | 53.9% | 40/372 | 42.1% | 76.6% | 9.8% |
| gpt_old | 33.1% | 55.7% | 91/372 | 68.0% | 49.0% | 34.7% |
| gpt_old+FB | 60.8% | 55.7% | 91/372 | 68.0% | 49.0% | 34.7% |
| gpt_fair | 66.3% | 77.9% | 104/372 | 70.5% | 66.1% | 23.9% |
| gpt_fair+FB | 70.7% | 77.9% | 104/372 | 70.5% | 66.1% | 23.9% |
| deepseek_old | 28.9% | 63.7% | 60/372 | 61.1% | 62.3% | 23.0% |
| deepseek_old+FB | 65.0% | 63.7% | 60/372 | 61.1% | 62.3% | 23.0% |
| deepseek_fair | 36.8% | 72.7% | 78/372 | 64.2% | 67.7% | 20.8% |
| deepseek_fair+FB | 68.8% | 72.7% | 78/372 | 64.2% | 67.7% | 20.8% |

## Paired differences

First system minus second; 95% clustered bootstrap intervals.
- gpt_fair+FB/main+FB+V: macro: +0.123 [+0.068, +0.178]; unsupported_rate: +0.167 [+0.108, +0.228]; supported_correct: +0.311 [+0.247, +0.374]
- gpt_fair+FB/candF1+FB+V: macro: +0.062 [+0.011, +0.111]; unsupported_rate: +0.172 [+0.114, +0.233]; supported_correct: +0.241 [+0.179, +0.298]
- deepseek_fair+FB/main+FB+V: macro: +0.105 [+0.056, +0.153]; unsupported_rate: +0.097 [+0.042, +0.153]; supported_correct: +0.259 [+0.200, +0.316]
- deepseek_fair+FB/candF1+FB+V: macro: +0.044 [-0.003, +0.091]; unsupported_rate: +0.102 [+0.046, +0.160]; supported_correct: +0.188 [+0.133, +0.246]
- gpt_fair+FB/gpt_old+FB: macro: +0.099 [+0.055, +0.140]; unsupported_rate: +0.035 [+0.008, +0.065]; supported_correct: +0.222 [+0.165, +0.280]
- deepseek_fair+FB/deepseek_old+FB: macro: +0.039 [+0.003, +0.074]; unsupported_rate: +0.048 [+0.016, +0.085]; supported_correct: +0.090 [+0.047, +0.133]
