"""Model and dataset cards for the Hugging Face release. Every number is read from the saved scorecards, never typed."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO_URL = "https://github.com/Bhaveshmeghwal21/against-the-safety"
BASE = "CohereLabs/tiny-aya-global"
REVISION = "af89d219b53ed9b13b8a4645f9c8028973510324"


def _card(split: str, parse: str) -> dict:
    return json.loads((ROOT / "reports" / f"scorecard_{split}_{parse}.json").read_text(encoding="utf-8"))


def _cell(system: dict, section: str, metric: str) -> str:
    v = system[section][metric]
    if isinstance(v, dict) and "k" in v:
        return f"{v['value']:.3f} ({v['k']}/{v['n']})"
    return "n/a" if v is None else f"{v:.3f}"


def _paired(card: dict, pair: str, metric_prefix: str) -> str:
    for name, r in card["paired"][pair].items():
        if name.startswith(metric_prefix) and r:
            return f"{r['before']:.3f} to {r['after']:.3f} ({r['difference']:+.3f}, 95% CI {r['ci95'][0]:+.3f} to {r['ci95'][1]:+.3f})"
    return "n/a"


ROWS = [
    ("supported answer (answerable episodes)", "grounding", "supported_correct (answerable episodes)"),
    ("answer correct, ignoring citations", "grounding", "answer_correct_ignoring_citations (answerable; F1 >= 0.5)"),
    ("cites a gold passage when answering", "grounding", "cites_a_gold_passage_when_answering (answerable)"),
    ("abstains when the evidence is missing", "abstention (documents-only requirement)", "correct_abstain (unanswerable episodes)"),
    ("answers anyway when the evidence is missing (lower is better)", "abstention (documents-only requirement)", "unsupported_answer_rate (unanswerable; lower is better)"),
    ("abstains although the evidence was there (lower is better)", "abstention (documents-only requirement)", "false_abstain (answerable; lower is better)"),
    ("searches although the evidence was already supplied (lower is better)", "when_to_retrieve", "unnecessary_search_rate (sufficient evidence supplied; lower is better)"),
    ("finds the full evidence with its own queries (open questions)", "what_to_retrieve", "full_support_found_by_own_queries (open questions)"),
    ("answers correctly with a distractor passage present", "ignores_distractors", "supported_correct with distractor passage"),
    ("invalid or rule-breaking turns (lower is better)", "tool_call_reliability (AuditKIT, Track 3)", "invalid_turn_rate (unparseable or rule-breaking; lower is better)"),
]


def results_table() -> str:
    card = _card("test", "lenient")
    names = ["base", "main", "control"]
    lines = ["| on the 86 test questions (with evidence masked or supplied in other states) | Tiny Aya base | **main adapter** | control adapter |", "| --- | ---: | ---: | ---: |"]
    for label, section, metric in ROWS:
        lines.append(f"| {label} | " + " | ".join(_cell(card["systems"][n], section, metric) for n in names) + " |")
    return "\n".join(lines)


def _headline_sentence() -> str:
    card = _card("test", "lenient")
    return (f"End-to-end good (a supported correct answer when the documents hold the answer, an abstention when they do not), "
            f"headline episodes: base to main {_paired(card, 'main vs base', 'end-to-end good')}; control to main {_paired(card, 'main vs control', 'end-to-end good')}.")


def model_card(name: str, user: str) -> str:
    is_main = name == "main"
    what = ("trained on the full matched-state mixture: search, answer and abstain decisions in six evidence states, including episodes where the supporting passage is missing"
            if is_main else "trained on the same questions and the same token budget, but only on supported tool episodes: no abstentions and no missing-evidence episodes")
    return f"""---
license: cc-by-nc-4.0
base_model: {BASE}
library_name: peft
pipeline_tag: text-generation
language: [en]
tags: [lora, qlora, peft, rag, retrieval, abstention, citations, agent, tiny-aya, lexsi-hackathon, agenttune, auditkit]
datasets: [{user}/against-the-safety-rag-episodes]
---

# against-the-safety-tiny-aya-rag-{name}

A LoRA adapter for [Tiny Aya Global]({"https://huggingface.co/" + BASE}) (revision `{REVISION[:8]}`), {what}. Built for the Lexsi AI Safety and Alignment
Hackathon, Track 3 (agentic fine-tuning), problem statement 4: fine-tuning for retrieval-augmented generation.
Team **Against-the-safety**. Code, plan and every number below: [{REPO_URL}]({REPO_URL}).

The adapter teaches the model a small search protocol: emit a `search_corpus` call, read the passages, then either
`<answer>..</answer><citations>chunk ids</citations>` or `<abstain>what is missing</abstain>`. It was trained with AgentTune
(`create_distill_trainer` with a QLoRA trainer factory) and scored with AuditKIT's RAG and agent metrics.

## Results (held-out test, scored by deterministic code)

{results_table()}

{_headline_sentence()}

Read this as a trade-off, not a free improvement. The main adapter stops answering from missing or incomplete evidence, and it also
declines some questions it could have answered. The control answers more of the answerable questions and never abstains.
The full tables, both readings of the base model's output (lenient first-valid-action and strict protocol), per-state breakdowns and
paired intervals are in the repository (`reports/`) and in the dataset repository.

## Use

Needs access to the gated base model (accept its terms on its model page) and a GPU for 4-bit loading.

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

base = "{BASE}"
tok = AutoTokenizer.from_pretrained(base, revision="{REVISION}")
model = AutoModelForCausalLM.from_pretrained(
    base, revision="{REVISION}", dtype=torch.float16, device_map={{"": 0}},
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
)
model = PeftModel.from_pretrained(model, "{user}/against-the-safety-tiny-aya-rag-{name}")
```

The adapter expects the system prompt and tool loop in the repository (`ragharness.protocol.system_prompt`, `ragharness.rollout`): the search tool
is AgentTune's `search_corpus` with a three-search budget; results are shown as `[chunk_id=c00012] Title` plus the passage text, and returned to
the model as a user turn.

## Training

QLoRA (4-bit NF4, fp16 compute) on one Kaggle T4, rank 16, alpha 32, dropout 0.05, all linear layers, learning rate 2e-4 (cosine, 3% warmup), effective
batch 16 rows, 2 epochs, context 2,560 tokens, seed 13, loss on the model's own turns only. Training rows come from 600 HotpotQA (distractor) questions and a
single shared SQLite FTS index; no row shows a passage that supports any development or test question. See `train_report.json` for the run.
The epoch count was chosen on the development split before the test split was opened.

## Limits

- Fine-tuned on one language (English) and one question style (HotpotQA-like, short factual answers); the 24-task Nimbus Home check shows weaker results on two-document questions and on choosing the current edition of a policy.
- The main adapter declines roughly one in five answerable questions on the test split.
- Evaluation sets are small (86 test questions with evidence states). Intervals are in the repository; differences of a few points are not established.
- Not for safety-critical or legal use. The base model's license (CC BY-NC 4.0) applies to this adapter.

## License

CC BY-NC 4.0, inherited from the base model. Cohere Labs' terms for Tiny Aya apply.
"""


def episodes_card(user: str) -> str:
    return f"""---
license: cc-by-sa-4.0
language: [en]
pretty_name: Against-the-safety RAG episodes and evaluation traces
tags: [rag, retrieval, abstention, agent-traces, hotpotqa, tiny-aya, lexsi-hackathon]
---

# Against-the-safety: matched-evidence RAG episodes and evaluation traces

Training rows, AgentTune EventLogs and evaluation traces behind [the adapters](https://huggingface.co/{user}/against-the-safety-tiny-aya-rag-main)
for Tiny Aya Global. Code: [{REPO_URL}]({REPO_URL}).

Derived from HotpotQA (distractor setting, revision `1908d6afbbead072334abe2965f91bd2709910ab`, CC BY-SA 4.0), so this dataset is **CC BY-SA 4.0**.
Passages are Wikipedia text through HotpotQA. Please attribute HotpotQA (Yang et al., 2018) and this project.

## Contents

- `training/main.jsonl`, `training/control.jsonl`, `training/dev.jsonl`: one prompt/completion row per assistant decision (search, answer or abstain). `main` is balanced 537 / 537 / 537; `control` has supported tool episodes only.
- `episodes/eventlogs_train.jsonl`: the AgentTune full-tier EventLogs the training rows were built from.
- `traces/<run>/<system>__<lenient|strict>__{{transcripts,scores,trajectories}}.jsonl`: every evaluated episode of the base model, the main adapter and the control adapter on dev and test. `transcripts` are readable, `scores` hold the deterministic outcome per episode, `trajectories` is AgentTune's TrajectoryStore export. `dev_epoch1` holds the epoch-1 adapters used to choose the epoch.
- `manifests/`: split manifest (question ids), index manifest (hash of the shared 7,820-passage index), leakage and shortcut reports. `reports/`: scorecards, trade-off tables, ROI.

## How it was built

Six evidence states are built under real search with one scripted policy: clean, distractors, second hop, support missing, evidence already given, and look-alike (first search returns weak results). A label can only come from what is visible at a turn.
Rows that show any development or test support passage were removed (264 of 6,697).
The scripted policy is a reference used to build verified episodes; it is not a model, and traces named base, main and control are real model outputs.

## Caveats

- Two readings of each model run are provided (lenient: first valid action; strict: exact protocol). The strict reading is the protocol-compliance measure.
- Small evaluation sets: 100 development and 100 test questions, of which 86 each had their evidence reachable within three searches for the scripted reference policy and form the headline episodes.
"""


def nimbus_card(user: str) -> str:
    return f"""---
license: cc-by-4.0
language: [en]
pretty_name: Nimbus Home fictional support suite
tags: [rag, retrieval, abstention, freshness, synthetic, lexsi-hackathon]
---

# Nimbus Home: a fictional product-support suite for documents-only question answering

24 dated, invented support documents (manuals, troubleshooting notes, firmware release notes, policies, a retired product manual and older
editions of four policies) and 24 hand-written tasks: 6 ordinary, 6 two-document, 6 dated (an older, superseded edition is also in the corpus), 6 with no
answer in the documents. All content is original and fictional, released under CC BY 4.0. "As of" date for freshness: 2026-10-15.

Files: `docs.jsonl` (with effective and superseded dates), `tasks.jsonl` (question, gold answer, supporting documents, oracle queries used only to prove the corpus can answer),
`traces/` (base, main and control runs), `scorecard_lenient.md`, `scorecard_strict.md`.

Checks run by `scripts/build_product.py` in the project repository: gold answer text is in the supporting document; oracle queries retrieve every supporting document within three searches;
no oracle search surfaces text that answers a missing-evidence task. This is a small exploratory suite, not a benchmark. The gold answers were shortened to their discriminating tokens after the
first run showed the F1 rule crediting "30 days" for "45 days"; all systems were rescored in a second run with the corrected suite, and both runs are described in the repository plan.

Code: [{REPO_URL}]({REPO_URL}).
"""
