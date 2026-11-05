# Against-the-safety

Team **Against-the-safety**, Lexsi AI Safety and Alignment Hackathon, Track 3 (agentic fine-tuning), **PS 4: fine-tuning for retrieval-augmented generation**.
[Competition](https://hackathon.lexsi.ai/).

We fine-tuned **Tiny Aya Global** (a Cohere model, 3.35B parameters) with QLoRA so that it searches a document corpus, answers with
citations to the passages it was shown, and abstains when those passages cannot support an answer. Training went through
[AgentTune](https://github.com/Lexsi-Labs/AgentTune); evaluation used [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) alongside our own deterministic scoring.
Everything ran on free Kaggle T4 GPUs.

## Result

Held-out test split: 100 HotpotQA questions, opened once, after every choice below had been fixed on the development split.
Same questions, same retrieval index, same three-search budget, same decoding for every system.
**Main** is trained on the full six-state mixture (search, answer, abstain). **Control** is trained on the same budget with supported tool episodes only (no abstention).

| test split | Tiny Aya base | **main adapter** | control adapter |
| --- | ---: | ---: | ---: |
| end-to-end good: supported answer when answerable, abstain when not (151 headline episodes) | 0.020 | **0.583** | 0.404 |
| supported correct answer, answerable episodes (205) | 0.034 | 0.527 | **0.707** |
| cites a gold passage when it answers | 0.126 | **0.994** | 0.990 |
| abstains when the evidence is missing (82) | 0.024 | **0.756** | 0.000 |
| answers anyway when the evidence is missing (82, lower is better) | 0.780 | **0.098** | 0.268 |
| abstains although the evidence was there (205, lower is better) | **0.000** | 0.190 | **0.000** |
| invalid or rule-breaking turns (287, lower is better) | 0.188 | **0.070** | 0.251 |

Paired differences (95% intervals from resampling whole questions): end-to-end good, main minus base **+0.563 [+0.490, +0.636]**; main minus control **+0.179 [+0.103, +0.252]**.
Supported correct answers, main minus control: **−0.180 [−0.268, −0.099]**.

This is a trade-off. The main adapter nearly stops answering from missing evidence (22 of 82 unsupported answers for the control, 8 for main) and it
declines about one answerable question in five. The control answers more of what is answerable, never abstains, and when the evidence is missing it keeps searching
until the budget runs out (58 of its 65 missing-evidence headline episodes end without a valid final turn). Which one is better depends on what a wrong answer costs.

Further numbers (all in [`reports/`](reports/)):

- Per evidence state, with five outcomes per episode: [`tradeoff_test_lenient.md`](reports/tradeoff_test_lenient.md).
- Every metric with its denominator and no composite score: [`scorecard_test_lenient.md`](reports/scorecard_test_lenient.md).
- Strict reading of the output format (exactly the trained protocol; the base model breaks it on 279 of 287 episodes): [`scorecard_test_strict.md`](reports/scorecard_test_strict.md).
- Cost per good outcome and break-even against the training cost: [`roi_test_lenient.md`](reports/roi_test_lenient.md).
- Epoch choice on the development split: [`dev_epoch_selection.md`](reports/dev_epoch_selection.md).
- Decoding-time abstention sweep: [`bias_sweep.md`](reports/bias_sweep.md).
- A 24-task practical check on a fictional product's documents (Nimbus Home): [`scorecard_product_lenient.md`](reports/scorecard_product_lenient.md).

## What PS 4 asks for, and where each part is measured

| PS 4 part | measurement (test split) |
| --- | --- |
| when to retrieve | searched although the evidence was already supplied (60, lower is better): base 1.000, main 0.283, control 0.067; did not search although nothing was supplied (180, lower is better): base 0.072, main 0, control 0 |
| what to retrieve | full supporting set found by the model's own queries: base 0.360, main 0.709, control 0.826; AuditKIT evidence-set recall 0.512 / 0.727 / 0.824 |
| grounding | supported correct answers (table above); AuditKIT citation support 0.302 / 0.771 / 0.941 |
| ignoring distractors | correct with a distractor passage present 0.067 / 0.667 / 0.833, same evidence without it 0.000 / 0.633 / 0.800 |
| abstention | table above; AuditKIT abstention 0.721 / 0.794 / 0.714 |
| tool-call reliability (AuditKIT) | tool-call validity 1.0 for all; loop-free 0.941 / 0.934 / 0.784; redundant calls 0.031 / 0.029 / 0.088 |

## How it works

The model talks to AgentTune's `search_corpus` tool with a budget of three searches, three passages per search, 800 characters shown per passage:

```
assistant: [{"tool_name":"search_corpus","parameters":{"query":"..."}}]
user:      [chunk_id=c00012] Title ... passage text ...
assistant: <answer>short answer</answer><citations>c00012,c00031</citations>        or        <abstain>what is missing</abstain>
```

Data: 600 train, 100 dev and 100 test questions sampled from HotpotQA (distractor setting), all passages in one shared SQLite FTS5 index. For each question the
builder writes up to five scripted episodes that differ only in what is masked or supplied, and labels each with one of six evidence states: clean support (S1), a distractor ranked above the
support (S2), second-hop recovery (S3), missing evidence (S4, label: abstain), evidence supplied in the prompt, fully or partly (S5), and a weak or masked first search that a reworded query recovers (S6).
"Missing" means the real supporting passages are hidden from search. Masked episodes are rejected when the gold answer text still appears in a passage the model could see
(`alternate_evidence_path`, 63 of the test candidates), and questions whose support the scripted search cannot reach within the budget are reported and never used: the test set has 86 of its 100 questions in the headline set (13 unreachable, 1 rejected).
The main mixture has 1,611 rows (537 search, 537 answer, 537 abstain); the control has 1,697 rows of supported tool episodes. Loss is on the model's own turns only.

Training settings were the same for both adapters and fixed before any result: QLoRA r=16, alpha=32, dropout 0.05, all linear layers, learning rate 2e-4 cosine, effective batch 16,
2 epochs, seed 13, context 2,560 tokens. Epoch 2 was kept because it had the higher end-to-end good rate on the development split (0.673 against 0.270 at epoch 1).

### Where AgentTune and AuditKIT are used

| | what we use | where |
| --- | --- | --- |
| AgentTune | `create_distill_trainer` with a QLoRA trainer factory (training); `SQLiteFTSBackend` and `SearchCorpusTool` (retrieval and the tool the model calls); `EventLog` and `TrajectoryStore` (every training episode and every evaluation rollout is stored as AgentTune events and trajectories); `qa_metrics` F1 and exact match | `src/ragharness/trainer.py`, `retrieval.py`, `tools.py`, `episodes.py`, `evaluation.py` |
| AuditKIT | `evidence_set_recall`, `citation_support`, `abstention`, `freshness` (RAG); `tool_call_validity`, `redundant_tool_calls`, `agent_loop_detection` (agent); each episode becomes an `ak.Sample` with a `RAGTrace`, scored with `ak.evaluate` | `src/ragharness/audit.py`, `scorecard.py` |

### Two readings of every model run

Each run is scored twice from the same transcripts: **lenient** (the first valid action in the output wins; this is the primary reading, because the base model often writes a tool call and then an invented answer in the same turn) and **strict** (the exact trained protocol, a compliance measure).
Both readings and the choice of primary were fixed before the test split was opened.

## Limits you should know about

- The main adapter abstains on 19% of answerable test episodes. On the development split it was 23%.
- The evaluation sets are small: 86 answerable test questions, 65 with masked evidence in the headline set. Differences of a few points are not established; intervals are in the reports.
- The "good" label for an answer uses F1 ≥ 0.5 against the gold answer plus citing a shown gold passage. F1 can credit near-misses on short answers ("30 days" against "45 days"), so the reports also give exact-match rates (main 0.424, control 0.571, base 0.020 on answerable episodes).
- Everything is English, short factual questions (HotpotQA style). The Nimbus Home suite is 24 tasks and exploratory: main answers 1 of 6 two-document questions and cites a superseded policy edition in 4 of 6 dated-change tasks.
- The Nimbus suite was run twice. The first run exposed gold answers that F1 could not tell apart ("30 days" for "45 days"); we shortened the gold answers to the discriminating tokens, fixed one question (P05), and re-ran every system. Both runs' traces are published in the Nimbus dataset (`traces/product_run1` and `traces/product`); the corrected run is the one reported.
- The check that masked episodes really lack the answer is a text match (the gold answer string in a visible passage). It does not prove that no other passage in the corpus could support a different valid answer.
- No second round of training: choosing a different mixture after seeing the test split would have meant opening it again.

## Artifacts

| | link |
| --- | --- |
| Adapter, main | https://huggingface.co/BMEscanor/against-the-safety-tiny-aya-rag-main |
| Adapter, control | https://huggingface.co/BMEscanor/against-the-safety-tiny-aya-rag-control |
| Episodes, evaluation traces, reports | https://huggingface.co/datasets/BMEscanor/against-the-safety-rag-episodes |
| Nimbus Home suite | https://huggingface.co/datasets/BMEscanor/nimbus-home-support-suite |
| Trace viewer (static Space, no model runs) | https://huggingface.co/spaces/BMEscanor/against-the-safety-trace-viewer |

Loading an adapter needs the gated base model: accept its terms at [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global), then follow the loading code on the model cards.

## Reproduce

Python 3.12. Everything, including caches and temp files, stays inside this folder.

CPU side (data, index, episodes, training sets, harness tests):

```powershell
py -3.12 -m venv .venv
. .\scripts\env.ps1                      # redirects every cache and temp path into .cache/
python -m pip install --no-deps -r requirements-cpu.lock.txt
python -m pip install --no-deps -e .
python scripts/prepare_data.py           # sample HotpotQA splits (600 train / 100 dev / 100 test), write manifests
python scripts/build_index.py            # shared SQLite FTS index
python scripts/retrieval_ceiling.py      # oracle retrieval ceiling on dev
python scripts/build_episodes.py         # six-state matched episodes, AgentTune trajectories, Tiny Aya rows
python scripts/make_training_set.py      # main, control and dev sets
python scripts/leakage_report.py         # dev/test overlap with SFT rows
python scripts/shortcut_report.py        # label-leak checks on training episodes
python scripts/build_product.py          # Nimbus Home documents and tasks
python -m pytest tests -q
```

GPU side (two T4s on Kaggle, launched from this machine with the Kaggle CLI; details and the failures the launcher guards against are in [kaggle/README.md](kaggle/README.md)):

```powershell
python scripts/kaggle_launch.py dataset                       # bundle code + exact data, upload, wait until Kaggle serves it
python scripts/kaggle_launch.py run train-both                # main on GPU 0, control on GPU 1, 2 epochs each (about 4 hours)
python scripts/kaggle_launch.py output ats-train-both
python scripts/kaggle_launch.py run eval-dev  --adapters-from ats-train-both
python scripts/kaggle_launch.py run eval-test --adapters-from ats-train-both     # open once
python scripts/kaggle_launch.py run eval-product --adapters-from ats-train-both
```

Reports from the downloaded outputs:

```powershell
python scripts/scorecard.py --help       # scorecards with paired bootstrap intervals
python scripts/tradeoff.py --help        # five-outcome accuracy and coverage tables
python scripts/roi.py --help             # cost per good outcome
python scripts/epoch_table.py --help     # development epoch comparison
```

Measured on a T4: training 8 s per row (about 1,640 tokens), 5.8 GB peak; evaluation about 20 minutes per system per split with batching, 9.3 GB peak.
Training cost: main 3.9 GPU-hours, control 3.6 GPU-hours (free Kaggle quota; about 1.36 and 1.26 USD at 0.35 USD per T4-hour).

## Layout

```
src/ragharness/   protocol, episodes, builder, retrieval, rollout, generation, metrics, AuditKIT adapter, scorecards, release cards
scripts/          data and index build, training, evaluation, reports, Kaggle launcher, release builder and publisher
kaggle/           notes for the Kaggle runs; scripts/kaggle/bootstrap.py is the kernel entry point
reports/          every table cited above (markdown and json)
demo/             static trace viewer (also published as a Hugging Face Space)
research/         planning record, including pre-registered choices (section 18)
```

## Licenses

| | license |
| --- | --- |
| This repository's code | MIT |
| Adapters | CC BY-NC 4.0 (inherited from Tiny Aya; the base model is gated, accept its terms to use it) |
| HotpotQA-derived dataset | CC BY-SA 4.0 (HotpotQA is CC BY-SA 4.0) |
| Nimbus Home suite | CC BY 4.0 (written by us, fictional) |
| AgentTune (LSAL v1.1) and AuditKIT (LSAL v1.2) | source-available, used as dependencies; not redistributed here |
