# CandF1 multilingual XQuAD comparison: preregistration

15 November 2026, before launching the candF1 model on these questions.

## Scope and timing

Kiro's existing multilingual run evaluated the previously trained base and main adapters; it did not train a multilingual adapter. This follow-up evaluates the existing candF1 epoch-1 adapter. There is no training or new adapter. The main-model XQuAD results were already observed before this follow-up was planned. XQuAD is therefore an opened exploratory comparison, not a blind test or a basis for claiming a newly validated winning checkpoint.

## Inputs and protocol

Use the same existing XQuAD Hindi, English and Spanish question sets, corpus, FTS index, reachable-question filtering, English system prompt, three-search budget, top-three passages, 800-character display, lenient parsing, 160 new tokens, 8,000 input tokens, greedy generation and scorer-v3 language normalization as the preceding main run. Run candF1's epoch-1 adapter on all three sets, using both T4s on the friend's Kaggle account, at most four hours. Apply the same fail-closed rule and V: the main adapter as checker, wording A, support threshold 0.50. No second look. The baseline is the saved Kiro run; do not silently replace it or change it.

For each language report candF1 raw, candF1+FB, candF1+FB+V, saved main raw and main+FB+V, base and always-refuse reference. Report correct-and-cited among answerable, hidden-evidence answers, invalid turns, false refusals, answer coverage and correct among answers, plus paired question/article-cluster intervals (3,000 draws, seed 47) for candF1 versus main on shared eligible episodes. Record truncations and replay discrepancies, and stop if input identities or episode/eligibility sets do not match. Treat hidden-evidence outcomes as annotation-based flags, not human-adjudicated hallucinations. Report results as observed regardless of which adapter wins.

Use an isolated private XQuAD-only bundle. Do not open, package, or run any Round 6 sealed-test input; do not modify Round 5 sealed inputs. Do not update Hugging Face or push GitHub without a separate explicit yes. The friend is not a teammate. The primary submission remains the previously selected main system unless separately decided under a defensible fresh-test protocol.

## Input SHA256 values

- `data/bench/xq_hi/questions.jsonl`: `27341d96f3953e1f82997f8375ffacc4f0b61f238ffdec26141861cc35a07412`
- `data/bench/xq_hi/corpus.json`: `75fed84cf74864a884ecd91e48c03740f994d83635265070e525b3a9b86244d1`
- `data/bench/xq_hi/corpus.sqlite`: `1136f3f4ab3566d576c25392e194fbbf7cfc709a6f4c1f0f1c01e1626856d865`
- `data/bench/xq_hi/manifest.json`: `d6f1909b5794a43b6583cc91c4e338784b9bd318770bc867f0714376489e08db`
- `data/bench/xq_en/questions.jsonl`: `815961e7c50a8efc5de608cad27d268cc702f93a205c994aa96869979b7b6d16`
- `data/bench/xq_en/corpus.json`: `b402132db63484d4509a90970b7c354f59b7fa1e7e13e056018d19f468fbf26f`
- `data/bench/xq_en/corpus.sqlite`: `cb9aa180b1b6492712cdaab4e9e694d19027c1d50bd965af8dcf3f591385eadb`
- `data/bench/xq_en/manifest.json`: `ce72ee14b34bc3a81cecec2105f2b1024b4b5ec01040a01bc690e43604f2c9bb`
- `data/bench/xq_es/questions.jsonl`: `d3d3176ec287e870f1a37ba303a1f0ed3af17e2c54ed45925d1e8c9927ad67a9`
- `data/bench/xq_es/corpus.json`: `5bf4956e6c252aa49b0cb6354eb56e829b66224fb0b6d0420641e8e3b8979439`
- `data/bench/xq_es/corpus.sqlite`: `5dc7fabd5afd7ac6e5077e363eea5e8599c27d84ffdb41b267cfaa94913019fb`
- `data/bench/xq_es/manifest.json`: `0ea3d343134804eff34fda2e4eb7e946a069dcb044fab21b13244cd755e924c0`
- `runs/kaggle/ats-train-both/out/main/adapter/adapter_model.safetensors`: `5834043354893027625a7198341616ca904393f6dc29d993912da8a1d7a8916a`
- `runs/account2_round4/ats-train-evidence-r4/out/candF/adapter_ep1/adapter_model.safetensors`: `cefbd4e824d6c59c5376a2ba3511acaa71a2f5784ebdf345a03efcc1368fb461`

## Amendment 1: verifier-only retry after GPU memory failure

The candF1 actor completed all 1,730 episodes, with no truncated inputs. Its checker then failed from GPU out-of-memory using batch size six; no candidate verifier verdicts were saved. The raw and fail-closed actor outcomes were inspected before this amendment. This retry reuses the exact 104 saved answer cases and the same main-adapter V, wording A and threshold 0.50. It does not regenerate answers, re-evaluate questions, train anything, run a second look, or change the selection protocol. The previous XQuAD results and raw candF1 outcomes are already known, so this remains exploratory.

To reduce peak memory, the checker uses batch size one and requests only last-position logits. The latter gives identical final-position logits on a CPU Cohere2 regression test; GPU numerical differences due to batching are still possible. All 104 cases have passages and must receive exactly one finite verifier probability. We will compare the completed +V results using the previously stated metrics and clustered bootstrap, report outcomes as observed, and preserve the failed run and raw report.

Retry input hashes (recorded before any retry model call):

- `runs/kaggle/ats-r6-system-candf1-xq/out/cases/answers.jsonl`: `b0cb94c1cb12a89c6ac0a4a700237a802430d759b903cd886052e3d6542aac6e`
- `runs/kaggle/ats-r6-system-candf1-xq/out/eval/candF1_bench_xq_hi/lenient/scores.jsonl`: `c2173bc8a79e652723c2a6583c06a65d5151fabd337f5530c1a5817f4c30c222`
- `runs/kaggle/ats-r6-system-candf1-xq/out/eval/candF1_bench_xq_hi/lenient/transcripts.jsonl`: `309df1a1cb0de5686eb59347d83fa5d8173e0bf24acfea533eebb67e3ed0c9bb`
- `runs/kaggle/ats-r6-system-candf1-xq/out/eval/candF1_bench_xq_en/lenient/scores.jsonl`: `363389a0adb936e39ff6c3609738f1f0e35f882fda724770dc1eddc29198925c`
- `runs/kaggle/ats-r6-system-candf1-xq/out/eval/candF1_bench_xq_en/lenient/transcripts.jsonl`: `eb9e3a68255d3633ef9aefc73613ebfabcf669da24742cc831f4b6819972d884`
- `runs/kaggle/ats-r6-system-candf1-xq/out/eval/candF1_bench_xq_es/lenient/scores.jsonl`: `97ec1a9a1990b20735a41de4e46907f39fc7d85d97fbcc677d7a7c83d9bf4108`
- `runs/kaggle/ats-r6-system-candf1-xq/out/eval/candF1_bench_xq_es/lenient/transcripts.jsonl`: `a5673c5140512ed102cf1dc19435573c4975300d0db731ae7703bae77fb140ef`
