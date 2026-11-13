# Against-the-safety

We fine-tuned Tiny Aya Global to search documents, cite answers and refuse when evidence is missing.

| System | Macro success | Correct and cited, of answerable | Answers on hidden evidence | Answered at all | Correct among answered |
| --- | ---: | ---: | ---: | ---: | ---: |
| Refuses every question | 42.0% | 0.0% | 0/375 | 0.0% | — |
| Base Tiny Aya | 3.1% | 5.7% | 281/375 | 77.6% | 4.4% |
| Main, no repairs | 47.9% | 48.8% | 115/375 | 56.3% | 51.9% |
| Main with both repairs | 58.0% | 46.3% | 42/375 | 41.2% | 67.3% |
| candF1 with both repairs | 64.0% | 53.3% | 40/375 | 41.8% | 76.4% |
| GPT-6 Luna with fail-closed | 60.4% | 55.3% | 91/374 | 68.0% | 48.7% |
| DeepSeek V4 Flash with fail-closed | 64.9% | 63.7% | 60/373 | 61.1% | 62.3% |

Scorer v3, on the opened round-5 test. The hosted-model macro values use their respective shared subsets; the other hosted columns use each model's available eligible cases. See [reference policies](reports/reference_policies.md), [corrected local scores](reports/round5_corrected.md), and [corrected hosted comparisons](reports/deepseek_comparison_corrected.md). The two repairs are fail-closed handling of invalid turns and a self-check using the main adapter; V is that checker. candF1 is an alternative, not the selected system.

![Corrected comparison of answerable and hidden-evidence cases](docs/figures/azure_tradeoff.svg)

## What it does

The model can search the corpus, then return an answer with chunk citations or say that the evidence is missing. In one recorded main-adapter episode, the question was “Which actress is "Beyond Rangoon" was also in "True Romance" and "Ed Wood?"” The searches were “Which actress is "Beyond Rangoon" was also in "True Romance" and "Ed Wood"?”; “Patricia Arquette”; the answer was **Patricia Arquette**, citing `c09437, c09415`. This is the actor's original trace before the self-check. In a hidden-evidence episode about the author of *The Sunlight on the Garden*, main refused because the accessible passages did not establish the answer. Explore the [recorded trace viewer](https://huggingface.co/spaces/BMEscanor/against-the-safety-trace-viewer); it is a static viewer, not a live model endpoint.

## What the numbers mean

- **Macro success:** the average of success rates across four task components; a correct refusal counts as success.
- **Correct and cited:** supported correct answers as a share of answerable episodes.
- **Answers on hidden evidence:** cases where the annotation-based evidence test did not establish support; lower is better. This is not a human adjudication of hallucination.
- **Answered at all:** coverage across eligible episodes.
- **Correct among answered:** supported correct answers divided by all answers shown; read it alongside coverage.

The refuse-everything row shows why macro success alone can be misleading when many episodes hide evidence. Annotation coverage and model availability differ between runs.

## Against hosted models

The saved GPT-6 Luna and DeepSeek V4 Flash native-tool runs used the same retrieval environment, but their prompts omitted the three-search limit and the answer-format details; invalid endings became refusals. They used no verifier and different decoding, so these are provisional comparisons. DeepSeek's corrected shared-set macro exceeds the selected main system. See the [paired comparison](reports/deepseek_comparison_corrected.md).

For 10,000 episodes, the illustrative estimate is $6.30 for the main system's busy T4 time, $2.66 for GPT-6 Luna and $5.57 for DeepSeek. The T4 uses $0.526/hour and assumes perfect utilization; storage, retrieval infrastructure, operations and taxes are excluded. API estimates use recorded token mix and published prices, not invoices. See [GPU and GPT costs](reports/azure_roi.md) and [DeepSeek costs](reports/deepseek_roi.md).

## Try it

The [quickstart notebook](notebooks/quickstart.ipynb) is prepared but has not yet been verified on a GPU. To ask over your own JSONL passages, use [scripts/ask.py](scripts/ask.py). Tiny Aya is gated and its license is noncommercial; a CUDA GPU and access to the base model are needed.

## Reproduce

```powershell
. .\scripts\env.ps1
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe scripts/remediate_audit.py
```

The [Kaggle instructions](kaggle/README.md) describe training, evaluation and environment setup.

## What failed and what we learned

- A control trained without abstentions answered more but did not reliably refuse missing evidence.
- Reweighting reduced some refusals while increasing unsupported answers.
- A reading-focused patch improved one type of question but failed its safety guard.
- Preference-training checkpoints did not meet the release gate; later training prompts also exposed some held-out evidence. The earlier main and candF1 adapters predate those runs.
- The checker reduces unsupported answers but also withholds some correct answers. A fresh, preregistered checker experiment is being prepared; no result is claimed yet.

The detailed [experimental history](docs/HISTORY.md) preserves earlier results and corrections.

## Artifacts and limits

[Model collection](https://huggingface.co/collections/BMEscanor/against-the-safety-tiny-aya-rag-adapters-6aec1fda812ed14d9f60bfbc) · [Data and demo collection](https://huggingface.co/collections/BMEscanor/against-the-safety-rag-episodes-traces-and-demo-6aec1fdb994e3b036c6c958d) · [Remediation report](reports/2026-11-13-evaluation-remediation.md)

The evaluation covers English short factual questions. The test has already been opened, corrected scoring was retrospective, and unannotated alternative evidence needs human review. Failure rates and costs should not be generalized beyond these conditions.

## Licenses

| | license |
| --- | --- |
| Code in this repository | MIT |
| Adapters | CC BY-NC 4.0, inherited from Tiny Aya (the base model is gated; accept its terms to use it) |
| HotpotQA- and SQuAD-derived episodes and traces | CC BY-SA 4.0, as HotpotQA and SQuAD 2.0 |
| RGB-derived verifier cases and traces in the mixed episodes bundle | CC BY-NC-SA 4.0; retain RGB attribution and noncommercial terms |
| Nimbus Home suite | CC BY 4.0 (written by us, fictional) |
| Benchmark traces | per benchmark, as the source: 2WikiMultihopQA Apache-2.0, MuSiQue CC BY 4.0, SQuAD 2.0 CC BY-SA 4.0, RGB CC BY-NC-SA 4.0 |
| AgentTune (LSAL v1.1), AuditKIT (LSAL v1.2) | source-available, used as dependencies, not redistributed |


