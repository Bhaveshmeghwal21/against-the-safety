# DeepSeek V4 Flash cost scenario

The scenario scales observed round-5 usage to 10,000 episodes with the same workload and cache pattern. Prices are illustrative retail rates, not the account invoice; quality and verifier settings differ between systems.

| system | recorded round-5 cases | estimated cost / 10,000 |
| --- | ---: | ---: |
| DeepSeek via Azure, exact | 938 | $1.65 |
| DeepSeek via Azure, native | 938 | $5.57 |
| GPT-6 Luna via Azure, exact | 938 | $1.57 |
| GPT-6 Luna via Azure, native | 938 | $2.66 |
| main+FB+V, ideal T4 busy time | 938 | $6.30 |
| candF1+FB+V, ideal T4 busy time | 938 | $5.99 |

DeepSeek pricing assumed $0.19 per million uncached input tokens, $0.028 per million cached input tokens and $0.51 per million output tokens, as published in [Microsoft’s Foundry announcement](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/introducing-deepseek-v4-flash-and-v4-pro-in-microsoft-foundry/4515174) (updated July 2026). Current actual prices, region, offer, discounts and billing may differ. The published table does not give a separate cache-write rate; this estimate treats writes as ordinary input. Verify the actual meters and charges in Azure Cost Management.
GPT-6 costs and local T4 assumptions are inherited from [the earlier ROI analysis](azure_roi.md). Local estimates assume a T4 at $0.526/hour that is fully utilized during all billed time and stopped immediately afterward; a continuously allocated 730-hour VM is $383.98 before disk and other infrastructure.

Output tokens include any billed reasoning. Recorded usage includes successful calls before a later provider block, but excludes pilots, failed calls and responses lost on interruption. User-specific routing and cache patterns can change the estimates.
- DeepSeek exact: 1271 recorded calls over 938 episodes (4 provider-blocked), 981,343 input tokens of which 339,200 cached, 45,615 output tokens, 0 reported cache-write tokens; estimated recorded spend $0.155.
- DeepSeek native: 2740 recorded calls over 938 episodes (5 provider-blocked), 3,313,458 input tokens of which 1,179,904 cached, 164,507 output tokens, 0 reported cache-write tokens; estimated recorded spend $0.522.
- Across the eight suites and both DeepSeek interfaces: 9,318 recorded calls, estimated $1.472 before pilots and unrecorded attempts.
