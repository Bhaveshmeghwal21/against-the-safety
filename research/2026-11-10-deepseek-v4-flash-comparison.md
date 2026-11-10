# DeepSeek V4 Flash comparison with the team RAG system

Date: 10 November 2026. This is an internal evaluation on the team's existing samples, not an official hackathon or benchmark leaderboard.

## Results and judgment

The full run completed 2,069 requested episodes in each interface. The paired round-5 comparison uses **863 of 938 episodes**: nine keys were unavailable for at least one system and 66 more were excluded by scorer v2 for at least one actor. All rows below use the same 305 hidden-evidence cases and the same answerable cases in that intersection.

| System | Round-5 macro | Supported correct | Unsupported / hidden | False refusals |
| --- | ---: | ---: | ---: | ---: |
| main + FB + V | 50.7% | 27.8% | 22/305 (7.2%) | 39.1% |
| candF1 + FB + V | 55.5% | 32.6% | 25/305 (8.2%) | 37.5% |
| GPT-6 Luna native + FB | 54.9% | 39.4% | 59/305 (19.3%) | 2.7% |
| **DeepSeek native + FB** | **61.1%** | **53.6%** | 36/305 (11.8%) | 9.1% |
| DeepSeek original format + FB | 32.8% | 26.2% | 158/305 (51.8%) | 11.5% |

On this sample, DeepSeek native + FB has the highest observed macro and supported-correct rate, and fewer unsupported answers than GPT-6 native + FB. Compared with GPT-6 native + FB, DeepSeek's paired macro difference is +6.2 percentage points (question-level 95% bootstrap interval +2.9 to +9.8), while its unsupported rate is −7.5 points (−11.9 to −3.5). The model and inference settings differ, and neither hosted model uses V.

Against the submitted `main` + FB + V, DeepSeek native + FB has +10.4 macro points (+6.6 to +14.2) and +25.8 points of supported correct answers (+20.6 to +31.1). Its measured unsupported-answer rate is +4.6 points (−0.3 to +9.8); the interval includes zero, so these results do not establish whether its true unsupported rate is higher or lower. This is not evidence of zero unsupported answers or a universal safety guarantee. The previously opened test, selected common intersection and unequal V checker also limit the comparison.

The apparent native macro relies substantially on fail-closed: DeepSeek's raw native macro was 27.3%, rising to 61.1% after invalid outcomes became refusals. On the common round-5 sample, 37.1% of raw native episodes were invalid; the frequent search-budget overruns and ignored additional tool calls described below are real limitations. The original-format DeepSeek result is much weaker on hidden evidence (158/305 unsupported), illustrating that the interface and wrapper materially affect measured behavior. The [full report](../reports/deepseek_comparison.md) includes raw scores, component results, paired intervals and the five earlier suites.

At 10,000 round-5-like episodes, DeepSeek native is estimated at **$5.57** in Azure tokens, versus **$2.66** for GPT-6 native and **$6.30** for idealized main-plus-verifier T4 busy time. The DeepSeek estimates include **$1.47** of recorded token usage across all eight suites and both interfaces, but omit pilots and some interrupted requests. Rates, cache patterns and actual billing can differ. See [cost assumptions](../reports/deepseek_roi.md).

## Model and setup

We used the existing Azure deployment named `DeepSeek-V4-Flash`. Its responses identify the model as `DeepSeek-V4-Flash`, but do not expose an immutable checkpoint version. Azure also lists a distinct `DeepSeek-V4-Flash-0731` deployment name; attempts to use that name on this resource returned deployment-not-found, so these results do not claim to measure the 0731 snapshot. DeepSeek's own direct API now redirects its legacy `deepseek-v4-flash` alias to V4.1 Flash; that separate direct API was not used here. The [DeepSeek model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) describes the open-weight model as a mixture of experts with 284B total parameters and 13B active per token, substantially larger than our 3.35B Tiny Aya base. See [Microsoft's Foundry description](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/introducing-deepseek-v4-flash-and-v4-pro-in-microsoft-foundry/4515174) and [DeepSeek's API documentation](https://api-docs.deepseek.com/).

Both DeepSeek runs use the same harness as the earlier GPT-6 Luna comparison: identical saved questions and variants, corpus and search index, evidence masks, top-three search results, 800-character display and three-search budget. The model generates its own search queries, so different actors can see different results. The raw API response's model name, tool calls, response text and token usage are checkpointed inside `runs/deepseek_exact` and `runs/deepseek_native`.

- **Exact:** the original Tiny Aya prompt and lenient text parser, using Azure's deployment defaults.
- **Native:** the same native function-tool interface and final-answer protocol as the earlier GPT-6 native run, with `reasoning_effort=none`. When the model requests multiple calls in a turn, the harness executes the first, as in the GPT-6 comparison. Minimal output-format normalization is logged and is subject to post-run audit.

Both Azure modes use a maximum of 512 completion tokens per API call and deployment-default temperature; the Tiny Aya actor used greedy decoding and 160 generated tokens. The deployment's exact default temperature was not independently established. Thus these are system comparisons, not a controlled experiment isolating model weights. They are single stochastic runs on previously opened samples.

There are eight internal suites, 2,069 episodes per interface before provider filtering: round-5 SQuAD (385), HotpotQA (82), and RGB (471), plus earlier 2WikiMultihopQA (166), MuSiQue (168), SQuAD 2.0 (202), RGB (453), and RGB integration (142). The final round-5 comparison reconstructs saved `main`, `candA`, and `candF1` plus their fail-closed rule (FB) and verifier (V), along with the saved GPT-6 Luna and new DeepSeek runs. No adapter or Kaggle job is run for this comparison. On the older five suites, only raw `main` and `candA` were saved; no final verifier or `candF1` results exist there.

All reported rows in a suite use the same intersection of cases present and eligible under scorer v2 for every compared actor. Azure content-filter blocks are recorded separately and excluded; they are not counted as refusals or failures. Scorer v2 also excludes hidden-evidence cases when a compared actor sees text containing alternate answer evidence. This prevents unequal denominators but changes the test population; alias matching and semantic support remain imperfect. The paired 95% bootstrap samples complete question IDs within each dataset and does not account for article-level dependence, stochastic reruns, or the differing verifier and inference settings.

The adapters' V checker is not applied to either hosted model. DeepSeek V4 Flash is open-weight, even though we access it as an Azure-hosted API; GPT-6 Luna is closed-source. FB converts a malformed or otherwise invalid terminal turn into an abstention; this can increase a score on hidden-evidence cases without demonstrating that the model itself chose to abstain. The reports show raw invalid rates alongside repaired outcomes. An Azure HTTP 400 may encode content filtering in either a top-level error or `choices[*].finish_reason`; the harness treats both as provider blocks, retains any reported usage, and does not deliberately modify or bypass filtered content. Earlier interrupted calls and pilots are not necessarily present in recorded billing estimates.

The native-tool pilot and the round-5 SQuAD trace inspection showed an important failure mode: an invalid outcome can be a request for a fourth search after the three-search budget has been exhausted, not just a malformed answer. On the full native round-5 SQuAD suite, 152 of 154 invalid outcomes were `budget_exceeded`; treating them as abstentions is a wrapper policy, not a self-chosen refusal.

Across all eight native suites, the completed traces contain 845 invalid episodes, of which 834 ended because the search budget was exceeded. DeepSeek requested multiple tools in 221 API calls; the shared harness executed only the first request in each such turn. The raw metadata retains the other tool calls, but these results cannot say how DeepSeek would perform if multiple calls per turn were supported. The post-run comparison of text-only responses to the emitted steps found no native plain-text normalization events. There were 11 recorded provider-blocked native episodes and five recorded original-format blocks; as explained below, those counts do not include every early filter event.

At the beginning of the full DeepSeek evaluation, the older handler did not recognize the nested form of Azure's HTTP-400 content-filter response, so three worker processes stopped before recording the affected episode IDs. Their uncheckpointed episodes were resumed after the handler was corrected; some in-flight calls were therefore repeated. Recorded block totals are a lower bound on all filter events, because an originally filtered response could have been followed by a different response after resumption. No prompt was modified to evade the filter. These interruptions also mean the recorded-usage estimate is not the full amount that may have been billed.

The deployment subsequently returned HTTP 429 and advertised a 250-request-per-minute limit. We stopped and resumed the workers with a local SQLite-coordinated minimum interval of 0.27 seconds across processes. This affects scheduling rather than the model prompt or scoring. In-flight calls during that restart may also be missing from the recorded token estimate.

## Cost method

The cost report scales recorded round-5 usage to 10,000 episodes at Microsoft's published Azure DeepSeek V4 Flash prices: $0.19 per million uncached input tokens, $0.028 per million cached input tokens, and $0.51 per million output tokens. The [Microsoft Foundry announcement](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/introducing-deepseek-v4-flash-and-v4-pro-in-microsoft-foundry/4515174) was last updated in July 2026, so region, deployment, current meter, subscription contract, and billing must be checked in Azure; this is not an invoice. The announcement lists no separate cache-write price; any reported cache writes are treated as ordinary input in this estimate. Actual usage can differ with traffic and cache hit rates.

Existing GPT-6 and T4 estimates are copied from [the earlier ROI report](../reports/azure_roi.md), including the unrealistic perfect utilization assumed by the busy-time T4 estimate. T4 storage, retrieval infrastructure, operations, taxes, and training beyond the single reported run are excluded. The local Tiny Aya model is non-commercially licensed; this analysis does not grant commercial deployment rights.

## Reproduction

Dot-source `scripts/env.ps1`, set `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_CHAT_URL` to the user's existing Azure chat-completions endpoint, and run the evaluator with `--model DeepSeek-V4-Flash --response-model DeepSeek-V4-Flash --temperature-label 'deployment default (unspecified)' --request-interval 0.27`. Use `--mode exact --out runs/deepseek_exact` and `--mode native --out runs/deepseek_native` for the respective checkpoints. These commands send prompts to Azure and may incur charges when results are not already present. For offline output set `HF_HUB_OFFLINE=1` and run `python scripts/report_deepseek_compare.py` and `python scripts/report_deepseek_roi.py`.

Reports: [paired results](../reports/deepseek_comparison.md), [machine-readable results](../reports/deepseek_comparison.json), [cost scenario](../reports/deepseek_roi.md), and [machine-readable costs](../reports/deepseek_roi.json).
