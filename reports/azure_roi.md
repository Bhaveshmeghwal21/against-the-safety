# Azure versus self-hosting: illustrative ROI

Workload: 10,000 episodes per month with the same mix, average token lengths, tool calls and cache-hit pattern as the three round-5 test sets. The quality comparison is in [azure_round5_comparison.md](azure_round5_comparison.md); the systems have different quality and the estimates are not a like-for-like outcome guarantee.

| System | Recorded episodes | Estimated GPU hours / 10,000 | Estimated compute or token cost / 10,000 |
| --- | ---: | ---: | ---: |
| main+FB+V, busy T4 | 938 | 11.98 | $6.30 |
| candF1+FB+V, busy T4 | 938 | 11.40 | $5.99 |
| Azure GPT-6 Luna, exact | 938 | provider-managed | $1.57 |
| Azure GPT-6 Luna, native | 938 | provider-managed | $2.66 |

## Inputs and limitations

- A Linux Standard_NC4as_T4_v3 GPU VM in East US is estimated at **$0.526/hour** using the [Azure Retail Prices API](https://prices.azure.com/api/retail/prices), queried for the East US VM SKU and consumption price; 730 hours continuously costs **$383.98/month** for the VM. The table assumes perfectly stoppable, fully utilized busy GPU hours; a real always-on or lightly used service can cost much more. It excludes disk, network, storage, retrieval server, CPUs beyond the VM, operations and taxes; availability and startup credits do not guarantee quota.
- Saved actor times apportion each model batch across episodes; saved verifier runtime is a separately batched average. The estimated checker workload counts answers that passed deterministic checks, matching the product pipeline. These are throughput estimates, not individual-request latency; batch size, prompt length and utilization affect cost.
- Azure uses Microsoft published Global Standard short-context prices per million tokens: $0.10 ordinary input, $0.01 cached input, $0.125 cache writes and $0.50 output. Each token is assigned once to its category; completion tokens include reasoning. Regional deployment, contract terms, credits and current rates can differ. [GPT-6 prices](https://azure.microsoft.com/en-us/blog/gpt-6-astra-sol-and-luna-for-production-agents-in-microsoft-foundry/), [cache behavior](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/prompt-caching), [token categories](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/provisioned-throughput-sizing).
- Estimates include recorded successful API calls, including successful calls before a later provider block. Failed requests, pilots, and any in-flight calls lost in an interrupted run may be absent. This is not the Azure invoice. Check actual usage and billing in the Azure portal. Cache behavior at other traffic levels may differ.
- The local model has a non-commercial license; this is an illustrative non-commercial workload, not an assertion that commercial deployment is permitted.

- main+FB+V: actor 20.0 output tokens/s effective batched throughput; actor plus estimated verifier 4.31 GPU seconds/episode; 476/938 episodes invoke the checker in this approximation.
- candF1+FB+V: actor 20.2 output tokens/s effective batched throughput; actor plus estimated verifier 4.10 GPU seconds/episode; 450/938 episodes invoke the checker in this approximation.
- Azure exact: 1278 recorded calls, 984,623 input tokens (35,877 cache reads, 299,937 cache writes), 89,144 output tokens; recorded usage estimated $0.147 for 938 episodes, including 0 provider-blocked episodes.
- Azure native: 2534 recorded calls, 2,483,405 input tokens (682,658 cache reads, 897,424 cache writes), 80,264 output tokens; recorded usage estimated $0.249 for 938 episodes, including 3 provider-blocked episodes.

Across all eight suites and both interfaces, the 8,751 recorded calls are estimated at **$0.862** at the assumed prices. This excludes pilots and any unrecorded attempts or calls; actual billing may differ.

Main-adapter training report: 3.90 hours for one historical training run, equivalent to $2.05 at this hypothetical T4 rental rate. This is not the actual Kaggle bill and excludes all other experiments, development, data preparation, and evaluation.
