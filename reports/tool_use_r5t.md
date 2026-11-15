# Tool use and task success per suite, first sealed test

Tool-call metrics are AuditKIT's (saved by each evaluation run) for the raw agent, before the fail-closed rule, over every episode. Task success is scorer v3 with both repairs (`reports/reference_policies.json`). No new model run.

| System | Suite | Episodes | Tool-call validity (higher is better) | Redundant tool calls (lower is better) | Loop-free episodes (higher is better) | Searches per episode | Invalid endings before the fail-closed rule (lower is better) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | SQuAD | 385 | 1.000 | 0.0% | 100.0% | 0.98 | 28.3% |
| base | HotpotQA | 82 | 1.000 | 0.0% | 100.0% | 0.98 | 9.8% |
| base | RGB | 471 | 1.000 | 0.5% | 98.9% | 0.83 | 16.3% |
| main | SQuAD | 385 | 1.000 | 5.7% | 87.5% | 1.62 | 12.2% |
| main | HotpotQA | 82 | 1.000 | 2.0% | 93.9% | 2.18 | 7.3% |
| main | RGB | 471 | 1.000 | 3.1% | 94.1% | 1.11 | 5.7% |
| candF1 | SQuAD | 385 | 1.000 | 7.6% | 84.2% | 1.45 | 15.6% |
| candF1 | HotpotQA | 82 | 1.000 | 3.0% | 90.2% | 2.09 | 7.3% |
| candF1 | RGB | 471 | 1.000 | 2.2% | 96.2% | 1.06 | 4.0% |

| System | SQuAD single | SQuAD two-part | HotpotQA | RGB | Macro success |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 1.8% | 3.7% | 0.0% | 7.0% | 3.1% |
| main+FB+V | 59.1% | 40.4% | 58.5% | 74.1% | 58.0% |
| candF1+FB+V | 61.4% | 54.0% | 64.6% | 76.0% | 64.0% |
| always_refuse | 40.0% | 43.5% | 46.3% | 38.0% | 42.0% |
