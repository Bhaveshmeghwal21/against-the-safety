"""Build concise submission documents from saved reports and traces."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from make_readme_table import render


def read(path):
    return (ROOT / path).read_text(encoding='utf-8')


old = read('README.md')
history = ROOT / 'docs/HISTORY.md'
if not history.exists():
    section = old[old.index('## Round 2:'):old.index('## How it works')]
    # Preserve the historical prose and tables; rebase relative links from docs/.
    section = re.sub(r'\]\((?!https?://|#|mailto:)([^)]+)\)', r'](../\1)', section)
    section = section.replace('src="docs/', 'src="../docs/')
    history.write_text('# Experimental history\n\nHistorical scorer v1/v2 results follow unchanged in substance. '
                       'The [13 November remediation](../reports/2026-11-13-evaluation-remediation.md) corrects '
                       'evidence eligibility, source offsets and later training-data exposure.\n\n' + section, encoding='utf-8')

roi = json.loads(read('reports/azure_roi.json'))
deep = json.loads(read('reports/deepseek_roi.json'))
local_cost = roi['local']['main+FB+V']['estimated_busy_gpu_usd_per_10000']
azure_cost = roi['cloud']['native']['estimated_usd_per_10000']
deep_cost = deep['deepseek_round5']['native']['estimated_usd_per_10000']
volume = roi['volume_per_month']
price = roi['gpu_usd_per_hour']
table = render()
license_section = old[old.index('## Licenses'):].split('\n---\n')[0] if '## Licenses' in old else ''
if not license_section:
    raise SystemExit('Cannot find the original license section')

trace_path = ROOT / 'runs/account2_round5/ats-eval-jobs-r5test-ref/out/eval/main_bench_r5t_hotpot/lenient/transcripts.jsonl'
trace = next(row for line in trace_path.read_text(encoding='utf-8').splitlines() if (row := json.loads(line))['final']['kind'] == 'answer'
             and row['outcome'] == 'supported_correct' and row['n_searches'] > 0)
queries = '; '.join(f'“{step["query"]}”' for step in trace['steps'] if step['kind'] == 'search')
answer = trace['final']['answer']
citations = ', '.join(trace['final']['citations'])

readme = f'''# Against-the-safety

We fine-tuned Tiny Aya Global to search documents, cite answers and refuse when evidence is missing.

{table}

Scorer v3, on the opened round-5 test. The hosted-model macro values use their respective shared subsets; the other hosted columns use each model's available eligible cases. See [reference policies](reports/reference_policies.md), [corrected local scores](reports/round5_corrected.md), and [corrected hosted comparisons](reports/deepseek_comparison_corrected.md). The two repairs are fail-closed handling of invalid turns and a self-check using the main adapter; V is that checker. candF1 is an alternative, not the selected system.

![Corrected comparison of answerable and hidden-evidence cases](docs/figures/azure_tradeoff.svg)

## What it does

The model can search the corpus, then return an answer with chunk citations or say that the evidence is missing. In one recorded main-adapter episode, the question was “{trace['question']}” The searches were {queries}; the answer was **{answer}**, citing `{citations}`. This is the actor's original trace before the self-check. In a hidden-evidence episode about the author of *The Sunlight on the Garden*, main refused because the accessible passages did not establish the answer. Explore the [recorded trace viewer](https://huggingface.co/spaces/BMEscanor/against-the-safety-trace-viewer); it is a static viewer, not a live model endpoint.

## What the numbers mean

- **Macro success:** the average of success rates across four task components; a correct refusal counts as success.
- **Correct and cited:** supported correct answers as a share of answerable episodes.
- **Answers on hidden evidence:** cases where the annotation-based evidence test did not establish support; lower is better. This is not a human adjudication of hallucination.
- **Answered at all:** coverage across eligible episodes.
- **Correct among answered:** supported correct answers divided by all answers shown; read it alongside coverage.

The refuse-everything row shows why macro success alone can be misleading when many episodes hide evidence. Annotation coverage and model availability differ between runs.

## Against hosted models

The saved GPT-6 Luna and DeepSeek V4 Flash native-tool runs used the same retrieval environment, but their prompts omitted the three-search limit and the answer-format details; invalid endings became refusals. They used no verifier and different decoding, so these are provisional comparisons. DeepSeek's corrected shared-set macro exceeds the selected main system. See the [paired comparison](reports/deepseek_comparison_corrected.md).

For {volume:,} episodes, the illustrative estimate is ${local_cost:.2f} for the main system's busy T4 time, ${azure_cost:.2f} for GPT-6 Luna and ${deep_cost:.2f} for DeepSeek. The T4 uses ${price:.3f}/hour and assumes perfect utilization; storage, retrieval infrastructure, operations and taxes are excluded. API estimates use recorded token mix and published prices, not invoices. See [GPU and GPT costs](reports/azure_roi.md) and [DeepSeek costs](reports/deepseek_roi.md).

## Try it

The [quickstart notebook](notebooks/quickstart.ipynb) is prepared but has not yet been verified on a GPU. To ask over your own JSONL passages, use [scripts/ask.py](scripts/ask.py). Tiny Aya is gated and its license is noncommercial; a CUDA GPU and access to the base model are needed.

## Reproduce

```powershell
. .\\scripts\\env.ps1
.\\.venv\\Scripts\\python.exe -m pytest tests -q
.\\.venv\\Scripts\\python.exe scripts/remediate_audit.py
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

{license_section}
'''
(ROOT / 'README.md').write_text(readme, encoding='utf-8')

abstract = read('article/ABSTRACT.md').strip()
article = f'''# Teaching a small model to search, cite and refuse when evidence is missing

**Team Against-the-safety** · Lexsi AI Safety and Alignment Hackathon · Track 3, PS 4: fine-tuning for RAG

## Abstract

{abstract}

## 1. Problem and approach

A document assistant should search when needed, use the retrieved passages and refuse when they cannot support an answer. We trained Tiny Aya Global with QLoRA on matched evidence states: evidence found, distractors, follow-up searches, evidence already supplied, and evidence hidden. The model calls `search_corpus`, cites chunk IDs when answering, or abstains. Invalid endings are converted to refusals. The main adapter also checks each proposed answer against its cited passages; this self-check is V.

## 2. Lexsi repositories and features used

[AgentTune](https://github.com/Lexsi-Labs/AgentTune) supplies the trainer integration, SQLite retrieval backend, search tool, episode logs and trajectory storage. [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) provides RAG and agent metrics, including evidence recall, citation support, abstention and tool-call validity. The final table uses our corrected scorer v3, not an AuditKIT composite. Later preference-training experiments used our own memory-efficient DPO loop because the generic trainer did not fit the GPU.

## 3. Models and licenses

The base is [Cohere Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global), gated and licensed CC BY-NC 4.0. Its published LoRA adapters inherit the noncommercial terms. The [model collection](https://huggingface.co/collections/BMEscanor/against-the-safety-tiny-aya-rag-adapters-6aec1fda812ed14d9f60bfbc) includes the selected main adapter and the candF1 alternative. Repository code is MIT. AgentTune and AuditKIT are source-available dependencies under their respective LSAL terms.

## 4. Datasets and licenses

Training and evaluation use HotpotQA and SQuAD-derived material under CC BY-SA 4.0, RGB-derived material under CC BY-NC-SA 4.0, and a fictional Nimbus Home support suite under CC BY 4.0. Other benchmark traces retain their source licenses: 2WikiMultihopQA Apache-2.0 and MuSiQue CC BY 4.0. The mixed data card identifies the component licenses; do not treat the whole bundle as commercially usable.

## 5. Collections and demo

[Models](https://huggingface.co/collections/BMEscanor/against-the-safety-tiny-aya-rag-adapters-6aec1fda812ed14d9f60bfbc) · [Datasets and demo](https://huggingface.co/collections/BMEscanor/against-the-safety-rag-episodes-traces-and-demo-6aec1fdb994e3b036c6c958d) · [Static trace viewer](https://huggingface.co/spaces/BMEscanor/against-the-safety-trace-viewer)

The viewer displays saved traces and does not run the model. The GitHub repository is private at present, so its links are not anonymously accessible.

## 6. Experiments and findings

{table}

Corrected scorer v3; hosted macro scores use the respective shared subsets while their remaining columns use each model's own available cases. Macro averages four task components and counts correct refusals as success, so the refuse-everything reference is essential. Coverage and correct-among-answered expose the tradeoff. Hidden-evidence answers are annotation-based support flags, not human-verified hallucinations. The held-out test was opened before the correction, so further selection needs a fresh test.

The hosted prompts in the saved runs omitted the search limit and exact answer format; a fair rerun has been prepared but not performed. Both hosted systems used fail-closed handling without V. The illustrative cost at {volume:,} episodes is ${local_cost:.2f} for busy T4 time, ${azure_cost:.2f} for GPT-6 Luna and ${deep_cost:.2f} for DeepSeek; the GPU assumption is ${price:.3f}/hour at perfect utilization, excluding other infrastructure costs.

Training without refusal examples increased coverage but weakened abstention. The reading patch and preference-training checkpoints failed their registered release conditions. Some later preference-training prompts exposed held-out HotpotQA evidence; main and candF1 were trained earlier. The checker blocks some unsupported outputs but also removes correct answers. English factual questions, annotation coverage and small subsets limit generalization.

## 7. What more credits would enable

We would train quote-first answer targets, balance supported answers and refusals, state the semicolon format for two-part answers, and expand the parent-question pool. A separately preregistered experiment would train an answer checker and test guarded second looks on a fresh holdout; it has not yet produced a result.

## 8. Team

[BMEscanor](https://huggingface.co/BMEscanor) and [abd420](https://huggingface.co/abd420). A friend provided Kaggle compute with permission; the friend is not listed as a teammate.
'''
(ROOT / 'article/ARTICLE.md').write_text(article, encoding='utf-8')
print(f"README words: {len(re.findall(r'\S+', read('README.md')))}; article words: {len(re.findall(r'\S+', article))}")
