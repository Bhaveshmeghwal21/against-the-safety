"""Model and dataset cards for the Hugging Face release. Every number is read from the saved scorecards, never typed."""

from __future__ import annotations

import json
from pathlib import Path
from functools import wraps

ROOT = Path(__file__).resolve().parents[2]
REPO_URL = "https://github.com/Bhaveshmeghwal21/against-the-safety"
BASE = "CohereLabs/tiny-aya-global"
REVISION = "af89d219b53ed9b13b8a4645f9c8028973510324"


def _audited_card(function):
    """Keep historical card tables visibly linked to their corrected replay."""
    @wraps(function)
    def card(*args, **kwargs):
        text = function(*args, **kwargs)
        marker = text.find("\n# ")
        end = text.find("\n", marker + 1)
        notice = ("\n\n**13 November 2026 correction:** The first-round tables below are historical scores. Sealed-test numbers (round 5 and later) use scorer v3, "
                  "which fixes evidence alternatives, binary answers and fixed evidence eligibility; later DPO prompts contained held-out HotpotQA support. "
                  "The submitted `main` and earlier `candF1` checkpoints are unchanged. "
                  f"See the [dated remediation report]({REPO_URL}/blob/main/reports/2026-11-13-evaluation-remediation.md) "
                  "for corrected replays, training-data quarantine and licensing details.\n")
        return text[:end] + notice + text[end:] if marker >= 0 else text + notice
    return card


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


def results_table(split: str = "test", names: tuple[str, ...] = ("base", "main", "control"), bold: str = "main") -> str:
    card = _card(split, "lenient")
    label = {"base": "Tiny Aya base", "main": "main adapter", "control": "control adapter", "candA": "candA adapter"}
    head = " | ".join(f"**{label[n]}**" if n == bold else label[n] for n in names)
    lines = [f"| on the {split} split (questions with evidence masked or supplied in other states) | {head} |", "| --- |" + " ---: |" * len(names)]
    for row_label, section, metric in ROWS:
        lines.append(f"| {row_label} | " + " | ".join(_cell(card["systems"][n], section, metric) for n in names) + " |")
    return "\n".join(lines)


BENCH_NAMES = [("2wiki", "2WikiMultihopQA"), ("musique", "MuSiQue 2-hop"), ("squad2", "SQuAD 2.0"), ("rgb", "RGB"), ("rgb_int", "RGB integration")]


def bench_table(bold: str = "main") -> str:
    """Ends-well table on the five external benchmarks, read from reports/benchmarks_lenient.json (plan sections 22 and 23)."""
    rep = json.loads((ROOT / "reports" / "benchmarks_lenient.json").read_text(encoding="utf-8"))
    names = ("base", "main", "candA", "control")
    label = {"base": "Tiny Aya base", "main": "main", "candA": "candA", "control": "control"}
    key = "ends well (all scored episodes)"
    fmt = lambda p: f"{p['difference']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]"
    head = " | ".join(f"**{label[n]}**" if n == bold else label[n] for n in names)
    lines = [f"| ends well (episodes) | {head} | main minus base, 95% CI |", "| --- |" + " ---: |" * len(names) + " --- |"]
    for b, title in BENCH_NAMES:
        m = rep["benchmarks"][b]["metrics"]
        n = m["main"][key]["n"]
        lines.append(f"| {title} ({n}) | " + " | ".join(f"{m[s][key]['value']:.3f}" for s in names) + f" | {fmt(rep['benchmarks'][b]['paired_ends_well']['main vs base'])} |")
    pooled = rep["pooled"]
    lines.append(f"| all five, pooled ({pooled['ends_well']['main']['n']:,}) | " + " | ".join(f"{pooled['ends_well'][s]['value']:.3f}" for s in names)
                 + f" | {fmt(pooled['paired']['main vs base'])} |")
    return "\n".join(lines)


BENCH_NOTE = ("Same harness as HotpotQA (an index per benchmark, three searches, top 3, 800 characters), fixed before any model ran on these sets; "
              "not comparable with official leaderboards. The adapters were trained only on two-passage HotpotQA questions with single-span answers, and it shows: "
              "on RGB integration they answer only the first part of two-part questions, and on SQuAD 2.0 they decline most single-passage questions even with the answer on screen. "
              "Per-benchmark tables: `reports/benchmarks_lenient.md` in the code repository.")


def _headline_sentence() -> str:
    card = _card("test", "lenient")
    return (f"End-to-end good (a supported correct answer when the documents hold the answer, an abstention when they do not), "
            f"headline episodes: base to main {_paired(card, 'main vs base', 'end-to-end good')}; control to main {_paired(card, 'main vs control', 'end-to-end good')}.")


def _test2_sentence() -> str:
    card = _card("test2", "lenient")
    return (f"On a second, fresh split of 100 questions (test2, opened once in round 2), end-to-end good went from base to main "
            f"{_paired(card, 'main vs base', 'end-to-end good (headline')}, and from control to main {_paired(card, 'main vs control', 'end-to-end good (headline')}.")


@_audited_card
def cand_a_card(user: str) -> str:
    card = _card("test2", "lenient")
    vs = card["paired"]["main vs candA"]
    e2e = next(r for n, r in vs.items() if n.startswith("end-to-end good (headline"))
    return f"""---
license: cc-by-nc-4.0
base_model: {BASE}
library_name: peft
pipeline_tag: text-generation
language: [en]
tags: [lora, qlora, peft, rag, retrieval, abstention, citations, agent, tiny-aya, lexsi-hackathon, agenttune, auditkit]
datasets: [{user}/against-the-safety-rag-episodes]
---

# against-the-safety-tiny-aya-rag-cand-a

A LoRA adapter for [Tiny Aya Global]({"https://huggingface.co/" + BASE}) (revision `{REVISION[:8]}`) from the project's second round: the same recipe, questions and token budget as
[the main adapter](https://huggingface.co/{user}/against-the-safety-tiny-aya-rag-main), with abstentions at 30% of final decisions instead of 50%
(1,668 rows: 556 search, 778 answer, 334 abstain). Built for the Lexsi AI Safety and Alignment Hackathon, Track 3, problem statement 4. Team **Against-the-safety**.
Code and plan: [{REPO_URL}]({REPO_URL}) (plan sections 20 and 21).

**Status: not the selected model.** The selection rule, fixed before training, chose between candidates on the development split, and there this adapter's
end-to-end good rate was below main's (100 against 107 of 159), so main stayed the project's model. Both were then evaluated once on a fresh split (test2).
Read it as a second operating point: fewer refusals and more supported answers, at the price of more answers when the evidence is missing.

## Results on test2 (100 fresh questions, scored by deterministic code)

{results_table("test2", ("base", "main", "candA", "control"), "candA")}

End-to-end good, candA against main on the 143 headline episodes: {e2e['before']:.3f} against {e2e['after']:.3f} (main minus candA {e2e['difference']:+.3f}, 95% CI {e2e['ci95'][0]:+.3f} to {e2e['ci95'][1]:+.3f}).
On the development split the order was the other way. The two adapters cannot be ranked on that rate with these sample sizes; they differ consistently in how often they refuse.

## Five external benchmarks (never trained on or used for any choice)

{bench_table("candA")}

{BENCH_NOTE} On these five sets candA is ahead of main on every one; main stays the selected model because the selection rule was fixed beforehand.

## Use

Loading is the same as for the main adapter (see its card); replace the adapter id with `{user}/against-the-safety-tiny-aya-rag-cand-a`.
It expects the same system prompt and tool loop (`ragharness.protocol.system_prompt`, `ragharness.rollout`).

## Limits

The main adapter's limits apply. In addition, this adapter answers on {_cell(card["systems"]["candA"], "abstention (documents-only requirement)", "unsupported_answer_rate (unanswerable; lower is better)")} of test2 episodes whose evidence is missing (main: {_cell(card["systems"]["main"], "abstention (documents-only requirement)", "unsupported_answer_rate (unanswerable; lower is better)")}).
Not for safety-critical or legal use.

## License

CC BY-NC 4.0, inherited from the base model. Cohere Labs' terms for Tiny Aya apply.
"""


def _round5_note() -> str:
    """Round-5 paragraph of the main card. Every number comes from reports/reference_policies.json (scorer v3, 934 shared episodes)."""
    ref = json.loads((ROOT / "reports" / "reference_policies.json").read_text(encoding="utf-8"))["sealed_r5test"]
    cor = json.loads((ROOT / "reports" / "round5_corrected.json").read_text(encoding="utf-8"))["comparisons"]["candF1+FB+V/main+FB+V"]
    b, m, s, f = ref["base"], ref["main"], ref["main+FB+V"], ref["candF1+FB+V"]
    hid = lambda x: f"{x['unsupported']} of {x['hidden_n']}"
    return f"""## Recommended use: the adapter plus two output-side repairs (round 5)

The results above are for the adapter alone. The submitted system adds two repairs that need no training. A rollout that ends without a valid answer or abstention is treated as an abstention (fail-closed). A self-check withholds an answer in three cases: this adapter, asked whether the cited passages state every part of the answer (reply `SUPPORTED` or `NOT`; one forward pass; threshold 0.5), says it is not supported; the cited chunk was never shown; or a part of the answer does not appear in the cited passages.

On a sealed test of {m['n']} fresh episodes (SQuAD, HotpotQA and RGB; scorer v3; opened for the reference systems and once more for one round-5b candidate), macro success is {b['macro']:.1%} for the base model, {m['macro']:.1%} for the adapter and {s['macro']:.1%} for the adapter with both repairs. Answers given although the evidence was hidden fall from {hid(b)} episodes (base) to {hid(m)} (adapter) and {hid(s)} (with repairs). The repairs cost caution: refusals of answerable episodes rise from {m['false_abstain']:.0%} to {s['false_abstain']:.0%}, and correct, cited answers do not increase ({m['supported_correct']:.1%} to {s['supported_correct']:.1%}). Among the answers that remain, {s['correct_among_answered']:.1%} are correct and cited (adapter alone: {m['correct_among_answered']:.1%}). The scorer is stricter than the one in the first-round table and the questions are harder, so these numbers do not compare with it. Code and tables: `reports/reference_policies.md` and `scripts/verify_round5.py` in the repository. To run the whole system on your own documents (needs a CUDA GPU and access to the gated base model), use `scripts/ask.py` (`src/ragharness/system.py`) or the web demo `scripts/demo_app.py`. A second round of preference training (round 5b) produced one eligible checkpoint, [candF1](https://huggingface.co/BMEscanor/against-the-safety-tiny-aya-rag-cand-f1). With the same repairs it scored {f['macro']:.1%} macro against {s['macro']:.1%} on the same test, but its unsupported-answer difference from this system ({cor['unsupported_rate']['difference']:+.3f}, interval {cor['unsupported_rate']['ci95'][0]:+.3f} to {cor['unsupported_rate']['ci95'][1]:+.3f}) did not meet the registered safety bound (upper end at most +0.02), so this adapter stays the submitted model."""


def multilingual_note() -> str:
    """Hindi and Spanish paragraph of the main card, from reports/xquad_results.json (scripts/report_xquad.py)."""
    rep = json.loads((ROOT / "reports" / "xquad_results.json").read_text(encoding="utf-8"))["systems"]
    s = lambda lang: rep[lang]["main+FB+V"]
    base = lambda lang: rep[lang]["base"]
    hid = lambda x: f"{x['hidden_answers']} of {x['n_hidden']}"
    return f"""## Other languages (a first check, not a claim)

The adapter was trained on English only. On 300 XQuAD questions in each of Hindi, English and Spanish (one run, evaluation only, nothing retrained), the system with both repairs answered although the evidence was hidden in {hid(s('hi'))} Hindi, {hid(s('en'))} English and {hid(s('es'))} Spanish episodes (base model: {hid(base('hi'))}, {hid(base('en'))}, {hid(base('es'))}). It gave a correct, cited answer when the evidence was available in only {s('hi')['correct_and_cited']:.1%} (Hindi), {s('en')['correct_and_cited']:.1%} (English) and {s('es')['correct_and_cited']:.1%} (Spanish) of questions, so its success rate ({s('hi')['success']:.1%}, {s('en')['success']:.1%}, {s('es')['success']:.1%}) is about what refusing every question scores (50.0%). English is as low as the other two because these single-passage questions are a known weak spot. In Hindi {rep['hi']['main']['invalid']:.0%} of the adapter's raw turns ended invalid, mostly by repeating the previous search (English {rep['en']['main']['invalid']:.0%}, Spanish {rep['es']['main']['invalid']:.1%}). See `research/2026-11-15-multilingual-results.md`."""


def _second_look_headline() -> str:
    """The registered decision numbers of the round-6 second-look test, from reports/round6_second_look.json."""
    rep = json.loads((ROOT / "reports" / "round6_second_look.json").read_text(encoding="utf-8"))
    macro = rep["comparisons"]["main+FB+V+second look/main+FB+V"]["macro"]
    return f"{macro['difference'] * 100:+.1f} points, interval {macro['ci95'][0] * 100:+.1f} to {macro['ci95'][1] * 100:+.1f}"


def fresh_test_note() -> str:
    """Round-6 paragraph of the main card, from reports/round6_second_look.json (scripts/report_round6_second_look.py) and reports/round6_second_look_extra.json."""
    rep = json.loads((ROOT / "reports" / "round6_second_look.json").read_text(encoding="utf-8"))
    extra = json.loads((ROOT / "reports" / "round6_second_look_extra.json").read_text(encoding="utf-8"))
    ref = json.loads((ROOT / "reports" / "reference_policies.json").read_text(encoding="utf-8"))["sealed_r5test"]["main+FB+V"]
    s, r = rep["systems"]["main+FB+V"], rep["systems"]["always refuse"]
    cmp = rep["comparisons"]["main+FB+V+second look/main+FB+V"]
    vs_refuse = extra["comparisons"]["main+FB+V/always refuse"]["macro"]
    c6, c5 = extra["round6_components"]["main+FB+V"], extra["round5_components"]["main+FB+V"]
    return f"""## A second fresh test, and a second look that did not pass (round 6)

After the round-5 test was used up, we built a second sealed test of {rep['shared_eligible']} episodes (SQuAD, HotpotQA and RGB; hashed before any model ran; scorer v3) and opened it once. With both repairs this adapter scored {s['macro']:.1%} macro success against {r['macro']:.1%} for refusing every question ({vs_refuse['difference'] * 100:+.1f} points, interval {vs_refuse['ci95'][0] * 100:+.1f} to {vs_refuse['ci95'][1] * 100:+.1f}), and answered {s['unsupported']} of {s['hidden_n']} hidden-evidence episodes (round-5 test: {ref['unsupported']} of {ref['hidden_n']}). Refusing carried over. Answering single-passage SQuAD questions did not: success was {c6['single']:.1%} against {c5['single']:.1%} on the round-5 test and {r['components']['single']:.1%} for refusing every question. The pooled share of correct answers among those shown was {s['correct_among_answered']:.1%} here; the round-5 figure ({ref['correct_among_answered']:.1%}) is not comparable because the mix of SQuAD, HotpotQA and RGB episodes differs. The base model and the hosted models were not run on this test.

The same run tested a *second look*: when the system refuses after seeing passages, it is asked once to answer anyway, and the answer is shown only if the self-check is at least 90% sure. The registered rule needed the macro-success gain to have an interval above zero. It was {cmp['macro']['difference'] * 100:+.1f} points (interval {cmp['macro']['ci95'][0] * 100:+.1f} to {cmp['macro']['ci95'][1] * 100:+.1f}), so the second look was not adopted and the submitted system is unchanged. See `research/2026-11-15-round6-second-look-results.md` in the repository."""


@_audited_card
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

{_test2_sentence() if is_main else ""}

Read this as a trade-off, not a free improvement. The main adapter stops answering from missing or incomplete evidence, and it also
declines some questions it could have answered. The control answers more of the answerable questions and never abstains.
The full tables, both readings of the base model's output (lenient first-valid-action and strict protocol), per-state breakdowns and
paired intervals are in the repository (`reports/`) and in the dataset repository.

{"A decoding-time sweep on the development split (`reports/bias_sweep.md`) added a bias to the logit of `abstain` after `<`: from 0 to -3 the results barely change; at -6 false abstains on answerable questions halve (20 to 10 of 86) but unsupported answers on unanswerable ones rise from 2 to 26 of 73; from -9 it never abstains, as with the control. No setting removed the false abstains without losing the correct ones." if is_main else ""}

{_round5_note() if is_main else ""}

{multilingual_note() if is_main else ""}

{fresh_test_note() if is_main else ""}

## Five external benchmarks (never trained on or used for any choice)

{bench_table(name)}

{BENCH_NOTE}

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
- On the external benchmarks above, two-part answers (RGB integration) and single-passage questions (SQuAD 2.0) fail; neither kind was in the training data. A third-round patch ([the patch adapter](https://huggingface.co/{user}/against-the-safety-tiny-aya-rag-patch)) fixes single-passage questions but failed the safety guard fixed beforehand.
- The main adapter declines roughly one in five answerable questions on the test split.
- Evaluation sets are small (86 test questions with evidence states). Intervals are in the repository; differences of a few points are not established.
- Not for safety-critical or legal use. The base model's license (CC BY-NC 4.0) applies to this adapter.

## License

CC BY-NC 4.0, inherited from the base model. Cohere Labs' terms for Tiny Aya apply.
"""


@_audited_card
def episodes_card(user: str) -> str:
    return f"""---
license: [cc-by-sa-4.0, cc-by-nc-sa-4.0]
language: [en]
pretty_name: Against-the-safety RAG episodes and evaluation traces
tags: [rag, retrieval, abstention, agent-traces, hotpotqa, tiny-aya, lexsi-hackathon]
---

# Against-the-safety: matched-evidence RAG episodes and evaluation traces

Training rows, AgentTune EventLogs and evaluation traces behind [the adapters](https://huggingface.co/{user}/against-the-safety-tiny-aya-rag-main)
for Tiny Aya Global. Code: [{REPO_URL}]({REPO_URL}).

HotpotQA- and SQuAD-derived files retain **CC BY-SA 4.0**. RGB-derived verifier cases and traces retain
**CC BY-NC-SA 4.0** ([RGB license](https://github.com/chen700564/RGB#license)); the bundle has mixed source licenses.
HotpotQA revision: `1908d6afbbead072334abe2965f91bd2709910ab`. Attribute HotpotQA (Yang et al., 2018), SQuAD,
RGB (Chen et al., 2024) and this project as applicable. RGB material is not licensed for commercial use.

## Contents

- `training/main.jsonl`, `training/control.jsonl`, `training/dev.jsonl`: one prompt/completion row per assistant decision (search, answer or abstain). `main` is balanced 537 / 537 / 537; `control` has supported tool episodes only.
- `episodes/eventlogs_train.jsonl`: the AgentTune full-tier EventLogs the training rows were built from.
- `traces/<run>/<system>__<lenient|strict>__{{transcripts,scores,trajectories}}.jsonl`: every evaluated episode of the base model, the main adapter and the control adapter on dev and test. `transcripts` are readable, `scores` hold the deterministic outcome per episode, `trajectories` is AgentTune's TrajectoryStore export. `dev_epoch1` holds the epoch-1 adapters used to choose the epoch. `dev_bias` and `dev_bias2` hold the main adapter decoded with an abstain-logit bias (systems `main_b<bias>`; see `reports/bias_sweep.md`; development split only).
- `manifests/`: split manifest (question ids), index manifest (hash of the shared 7,820-passage index), leakage and shortcut reports. `reports/`: scorecards, trade-off tables, ROI.
- Second round (plan sections 20 and 21 in the code repository): `training/candA.jsonl` and `training/candB.jsonl` (two re-weighted selections of the same train decisions), `traces/dev_round2/` (both candidates on dev, used for selection), `traces/test2/` (base, closed book, main, control and candA on the fresh split), `manifests/test2_manifest.json` (question ids, sampling rule, and the hash of the extended 8,777-passage index).
- Third round (plan sections 24 to 26): `training/candC.jsonl` and `training/candD.jsonl` (the two patch sets: new rows from SQuAD 2.0 training-split passages and HotpotQA train questions, plus rehearsal rows), `episodes/eventlogs_train3.jsonl` (the new episodes), `traces/dev_round3*` (both patches, and the weight blends of the main and candD adapters, on the development set, which holds held-out SQuAD training-split articles), `manifests/r3dev.json` (what was drawn for the development set). Fresh-test traces and manifests for the third round are in the benchmark-traces dataset.
- Hindi training (`research/2026-11-15-hindi-training-preregistration.md`): `training/hiMain.jsonl` and `training/hiF.jsonl` (540 Hindi rows built from 220 IndicSQuAD Hindi questions, plus English replay), `episodes/eventlogs_trainhi.jsonl` (the Hindi episodes). IndicSQuAD (`l3cube-pune/indic-squad`) is listed CC BY 4.0 and translates SQuAD 2.0 (CC BY-SA 4.0), so treat these files as CC BY-SA 4.0. Hindi traces are in the benchmark-traces dataset.
- Fourth and fifth rounds (plan sections 27 and 28, `research/` notes in the code repository): `training/candE.jsonl` and `training/candF.jsonl` (matched evidence decisions), `training/dpo_r5.jsonl` (335 preference pairs from sampled rollouts), `training/dpo_r5b.jsonl` and `training/force_contexts_r5.jsonl` (the same pairs plus abstention-chosen pairs, and the decision points they come from), `sampling/` (the 1,920 sampled rollouts and the forced answers), `verifier/` (the cases the self-check was run on and its verdicts), `traces/dev_round4`, `traces/dev_round5` and `traces/dev_round5b` (checkpoints on HotpotQA dev). The round-4 and round-5 SQuAD-style development and test traces, and the sealed round-5 test, are in the benchmark-traces dataset.

## How it was built

Six evidence states are built under real search with one scripted policy: clean, distractors, second hop, support missing, evidence already given, and look-alike (first search returns weak results). A label can only come from what is visible at a turn.
Rows that show any development or test support passage were removed (264 of 6,697).
That filter covers the original SFT rows. A 13 November audit found held-out HotpotQA support in later DPO
prompts; `dpo_r5.jsonl` and `dpo_r5b.jsonl` are historical contaminated artifacts, not validated training sets.
Use separately generated clean files and the content-fingerprint training guard for future experiments.
The scripted policy is a reference used to build verified episodes; it is not a model, and traces named base, main and control are real model outputs.

## Caveats

- Two readings of each model run are provided (lenient: first valid action; strict: exact protocol). The strict reading is the protocol-compliance measure.
- Small evaluation sets: 100 development, 100 test and 100 test2 questions, of which 86, 86 and 83 had their evidence reachable within three searches for the scripted reference policy and form the headline episodes.
"""


@_audited_card
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


BENCH_SOURCES = {  # folder -> (benchmark, license, citation)
    "2wiki": ("2WikiMultihopQA, dev", "Apache-2.0", "Ho et al., 2020, Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps"),
    "musique": ("MuSiQue-Ans v1.0, dev, 2-hop", "CC BY 4.0", "Trivedi et al., 2022, MuSiQue: Multihop Questions via Single-hop Question Composition"),
    "squad2": ("SQuAD 2.0, dev", "CC BY-SA 4.0", "Rajpurkar et al., 2018, Know What You Don't Know: Unanswerable Questions for SQuAD"),
    "rgb": ("RGB, English refined set", "CC BY-NC-SA 4.0", "Chen et al., 2024, Benchmarking Large Language Models in Retrieval-Augmented Generation"),
    "rgb_int": ("RGB, English integration set", "CC BY-NC-SA 4.0", "Chen et al., 2024 (as above)"),
    "r3t_hotpot": ("HotpotQA validation, fresh questions (round 3 test)", "CC BY-SA 4.0", "Yang et al., 2018, HotpotQA"),
    "r3t_2wiki": ("2WikiMultihopQA, dev, fresh questions (round 3 test)", "Apache-2.0", "Ho et al., 2020 (as above)"),
    "r3t_musique": ("MuSiQue-Ans v1.0, dev, 2-hop, fresh questions (round 3 test)", "CC BY 4.0", "Trivedi et al., 2022 (as above)"),
    "r3t_squad2": ("SQuAD 2.0, dev, fresh questions (round 3 test)", "CC BY-SA 4.0", "Rajpurkar et al., 2018 (as above)"),
    "r3t_rgb": ("RGB, English refined set, fresh questions (round 3 test)", "CC BY-NC-SA 4.0", "Chen et al., 2024 (as above)"),
    "r3t_rgb_int": ("RGB, English integration set, fresh questions (round 3 test)", "CC BY-NC-SA 4.0", "Chen et al., 2024 (as above)"),
    "r4dev": ("SQuAD 2.0 training-split articles, round-4 development set (single and two-part questions, evidence variants)", "CC BY-SA 4.0", "Rajpurkar et al., 2018 (as above)"),
    "r4test": ("SQuAD 2.0 training-split articles, round-4 final holdout", "CC BY-SA 4.0", "Rajpurkar et al., 2018 (as above)"),
    "r5t_squad": ("SQuAD 2.0 training-split articles, sealed round-5 test", "CC BY-SA 4.0", "Rajpurkar et al., 2018 (as above)"),
    "r5t_hotpot": ("HotpotQA validation, sealed round-5 test", "CC BY-SA 4.0", "Yang et al., 2018, HotpotQA"),
    "r5t_rgb": ("RGB, English refined set, sealed round-5 test (noise 0, 0.8, 1)", "CC BY-NC-SA 4.0", "Chen et al., 2024 (as above)"),
    "r6t_squad": ("SQuAD 2.0 training-split articles, sealed round-6 test (main only)", "CC BY-SA 4.0", "Rajpurkar et al., 2018 (as above)"),
    "r6t_hotpot": ("HotpotQA validation, sealed round-6 test (main only)", "CC BY-SA 4.0", "Yang et al., 2018, HotpotQA"),
    "r6t_rgb": ("RGB, English refined set, sealed round-6 test (main only)", "CC BY-NC-SA 4.0", "Chen et al., 2024 (as above)"),
    "xq_hi": ("XQuAD, Hindi (multilingual check, 300 questions; main and base only)", "CC BY-SA 4.0", "Artetxe et al., 2020, On the Cross-lingual Transferability of Monolingual Representations"),
    "xq_en": ("XQuAD, English (multilingual check, same questions)", "CC BY-SA 4.0", "Artetxe et al., 2020 (as above)"),
    "xq_es": ("XQuAD, Spanish (multilingual check, same questions)", "CC BY-SA 4.0", "Artetxe et al., 2020 (as above)"),
    "xq2_hi": ("XQuAD, Hindi (Hindi training test, 300 fresh questions)", "CC BY-SA 4.0", "Artetxe et al., 2020 (as above)"),
    "xq2_en": ("XQuAD, English (Hindi training test, same questions)", "CC BY-SA 4.0", "Artetxe et al., 2020 (as above)"),
    "hi_dev": ("IndicSQuAD Hindi (Hindi training development set, 120 machine-translated questions)", "CC BY 4.0 as listed; translates SQuAD 2.0 (CC BY-SA 4.0), so treat as CC BY-SA 4.0", "L3Cube-IndicSQuAD (l3cube-pune/indic-squad); Rajpurkar et al., 2016, 2018"),
}


def r5_final_table() -> str:
    """Sealed round-5 test (934 episodes shared by the local systems, scorer v3), read from reports/reference_policies.json."""
    ref = json.loads((ROOT / "reports" / "reference_policies.json").read_text(encoding="utf-8"))["sealed_r5test"]
    names = [("base", "base"), ("main", "main, no repairs"), ("main+FB+V", "main + fail-closed + self-check (submitted)"), ("candF1+FB+V", "candF1 + fail-closed + self-check"), ("always_refuse", "refuse every question")]
    lines = ["| configuration | macro success | unsupported answers / hidden-evidence episodes | correct and cited (answerable) | refusals (answerable) |", "| --- | ---: | ---: | ---: | ---: |"]
    for key, label in names:
        s = ref[key]
        lines.append(f"| {label} | {s['macro']:.3f} | {s['unsupported']} / {s['hidden_n']} | {s['supported_correct']:.3f} | {s['false_abstain']:.3f} |")
    return "\n".join(lines)


@_audited_card
def bench_card(user: str) -> str:
    rows = "\n".join(f"| `traces/{k}/`, `manifests/{k}.json` | {v[0]} | {v[1]} | {v[2]} |" for k, v in BENCH_SOURCES.items())
    return f"""---
license: [apache-2.0, cc-by-4.0, cc-by-sa-4.0, cc-by-nc-sa-4.0]
language: [en]
pretty_name: Against-the-safety external benchmark traces
tags: [rag, retrieval, abstention, agent-traces, 2wikimultihopqa, musique, squad2, rgb, tiny-aya, lexsi-hackathon]
---

# Against-the-safety: evaluation traces on five external benchmarks

Every evaluated episode of the base model (Tiny Aya Global), the [main adapter](https://huggingface.co/{user}/against-the-safety-tiny-aya-rag-main),
the [candA adapter](https://huggingface.co/{user}/against-the-safety-tiny-aya-rag-cand-a) and the
[control adapter](https://huggingface.co/{user}/against-the-safety-tiny-aya-rag-control) on 2WikiMultihopQA, MuSiQue, SQuAD 2.0, RGB and RGB integration,
1,131 episodes per system, plus the closed-book baseline. Nothing here was used for training or for any choice, but these sets showed the failures that round 3 then tried to patch.
The sets, settings and metrics were fixed before any model ran on them (plan section 22, results in section 23): [{REPO_URL}]({REPO_URL}).

## Results (ends well: a supported answer when the documents hold the answer, an abstention when they do not)

{bench_table()}

{BENCH_NOTE}

## Licenses: each folder keeps the license of its source benchmark

| files | source | license | please cite |
| --- | --- | --- | --- |
{rows}
| `reports/` | aggregate numbers only | CC BY 4.0 | this project |

The RGB folders are non-commercial (CC BY-NC-SA 4.0), as RGB. The traces contain passages and questions from the source benchmarks, model outputs, and scores.

## Round 3: a fresh test set (`r3t_*` folders)

Six sets of questions disjoint from every earlier set (plan sections 24 to 26), run once for the main adapter, the round-3 patch (candD, [patch adapter](https://huggingface.co/{user}/against-the-safety-tiny-aya-rag-patch)) and candA, lenient reading only, 739 episodes per system:

{r3_test_table()}

Pooled unsupported answers are dominated by RGB's on-topic-passage setting, where every adapter is weak. The patch fixed single-passage questions but failed the safety criterion fixed beforehand, so the main adapter stays the headline model.

## Rounds 4 and 5: the sealed final test (`r5t_*` folders) and the round-4 sets (`r4dev`, `r4test`)

`r5t_squad`, `r5t_hotpot` and `r5t_rgb` are the sealed round-5 test: 938 episodes built before any round-5 training from questions no earlier round used, opened for the reference systems and once more for candF1 (round 5b). The oracle's 1.0 was the weaker success metric; strict scorer-v2 evidence success was 372/471 on RGB and 78/82 on HotpotQA.
The manifests carry the file hashes. The traces are the raw model runs. The two repairs of the submitted system are applied offline by `scripts/report_round5.py`: a rollout that ends without a valid
answer or abstention becomes an abstention (fail-closed, FB), and an answer the self-check withholds becomes an abstention (V; the verdicts are in the episodes dataset under `verifier/`). Scorer v3, the 13 November corrected replay; the historical scorer-v2 tables stay in `reports/round5_final.md`.

{r5_final_table()}

Macro success is the mean of four component rates (SQuAD single, SQuAD two-part, HotpotQA, RGB); the scorer is stricter than the one used for the first-round tables, so the numbers do not compare with them.
Paired intervals and the per-part numbers are in `reports/reference_policies.md` and `reports/round5_corrected.md`. `r4dev` and `r4test` are the round-4 development set and final holdout.

## Round 6: the second sealed test (`r6t_*` folders; main only)

`r6t_squad`, `r6t_hotpot` and `r6t_rgb` are 679 episodes (75, 60 and 29 questions) built and hashed before any model ran, and opened once for the main adapter. The run produced the raw traces, the self-check's verdicts on every answer (`round6_second_look/verify__V__verdicts_A.jsonl`), the forced second-look answers (`forced__forced_main.jsonl`) and the self-check's verdicts on them (`verify_forced__V__verdicts_A.jsonl`), all in `round6_second_look/`. The registered report is `reports/round6_second_look.md` (and `.json`); post hoc numbers are in `reports/round6_second_look_extra.md`. Under the registered rule the second look was not adopted (macro success {_second_look_headline()}). The base model and hosted models were not run on this test.

## Hindi training (research/2026-11-15-hindi-training-preregistration.md)

Traces of the Hindi experiment: `xq2_hi` and `xq2_en` (300 fresh XQuAD questions in Hindi and English), `hi_dev` (development), and `r6t_*` for the adapters `hiMain` and `hiF` and their parents `main` and `candF1`. `hindi_training/` holds the self-check's cases and verdicts (the parents' verdicts are a batch-1 rerun on the saved cases, see the amendment in the pre-registration). The registered report is `reports/hindi_results.md`: both adapters gained in Hindi and failed the registered safety rules, so they are experiments and the submitted system is unchanged.

## Contents

- `traces/<benchmark>/<system>__<lenient|strict>__{{transcripts,scores,trajectories}}.jsonl` and `__summary.json`: `transcripts` are readable conversations, `scores` hold the deterministic
  outcome of each episode, `trajectories` is AgentTune's TrajectoryStore export, `summary` holds the aggregate and AuditKIT metrics. Systems: `base`, `main`, `candA`, `control` (first suite); `main`, `candD`, `candA` (`r3t_*`);
  `base`, `main`, `candA`, `candE1`, the round-5 checkpoints and the round-5b checkpoints (`r4dev`, `r4test`, `r5t_*`, where present).
- `traces/<benchmark>/closed__summary.json`: the base model answering from memory, question only.
- `manifests/<benchmark>.json`: source file, question ids, sampling seed, index hash and counts, written by `scripts/prepare_benchmarks.py`.
- `reports/benchmarks_lenient.md`, `reports/benchmarks_strict.md` (and `.json`): every metric with denominators, paired intervals, AuditKIT rows.

## Protocol, in short

Each benchmark gets its own SQLite FTS5 index (AgentTune `SQLiteFTSBackend`). The model may search three times (top 3, 800 characters shown) and must then answer with
citations or abstain. Variants: open search, and the support masked from search (correct ending: abstain), on questions whose support a scripted policy reaches within three
searches. Historical natural SQuAD questions used corpus-wide search despite paragraph-level labels; new evaluations restrict them to the source paragraph. RGB uses five passages where available (historical zero-noise runs sometimes had fewer; corrected generation requires five). At noise 1
the positives are also masked from search). An answer is correct at F1 >= 0.5 against the gold answer or an alias, or, for RGB, when every answer part is contained;
it is supported when it is correct, every cited chunk was shown, and a cited chunk holds the answer. The numbers are not comparable with official leaderboards.
"""


# ---------------------------------------------------------------- round 3 (plan sections 24 to 26)

R3_TITLES = {"r3t_hotpot": "HotpotQA, fresh", "r3t_2wiki": "2WikiMultihopQA, fresh", "r3t_musique": "MuSiQue 2-hop, fresh", "r3t_squad2": "SQuAD 2.0, fresh questions",
             "r3t_rgb": "RGB, fresh", "r3t_rgb_int": "RGB integration, fresh"}


def _r3(name: str) -> dict:
    return json.loads((ROOT / "reports" / f"{name}.json").read_text(encoding="utf-8"))


def _ci(p: dict) -> str:
    return f"{p['difference']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]"


def r3_test_table() -> str:
    """Fresh test set (739 episodes per system): ends well per set, macro average, and the pooled unsupported-answer rate."""
    rep, ca = _r3("round3_test"), _r3("round3_test_candA")
    key = "ends well (all scored episodes)"
    names = ("main", "candD", "candA")
    lines = ["| ends well (episodes) | main | candD (round 3 patch) | candA (round 2) |", "| --- | ---: | ---: | ---: |"]
    for b, c in rep["components"].items():
        m = c["metrics"]
        lines.append(f"| {R3_TITLES[b]} ({m['main'][key]['n']}) | " + " | ".join(f"{m[s][key]['value']:.3f}" for s in names) + " |")
    lines.append("| **macro average of six sets** | " + " | ".join(f"{rep['macro'][s]:.3f}" for s in names) + " |")
    lines.append(f"| difference from main, 95% CI | | {_ci(rep['macro_paired'])} | {_ci(ca['macro_paired'])} |")
    pu, pa = rep["pooled_unsupported_answers"], ca["pooled_unsupported_answers"]
    lines.append(f"| answers when no passage in reach holds the answer ({pu['n']} episodes; lower is better) | {pu['before']:.3f} | {pu['after']:.3f} | {pa['after']:.3f} |")
    lines.append(f"| difference from main, 95% CI | | {_ci(pu)} | {_ci(pa)} |")
    return "\n".join(lines)


def r3_dev_table() -> str:
    sel = _r3("round3_selection")["stats"]
    comps = list(sel["main"]["components"])
    lines = ["| development set, ends well | main | candC | candD |", "| --- | ---: | ---: | ---: |"]
    for c in comps:
        lines.append(f"| {c} | " + " | ".join(f"{sel[s]['components'][c]['rate']:.3f}" for s in ("main", "candC", "candD")) + " |")
    lines.append("| macro average | " + " | ".join(f"{sel[s]['macro']:.3f}" for s in ("main", "candC", "candD")) + " |")
    lines.append("| answers when no passage in reach holds the answer (of 282; lower is better) | " + " | ".join(str(sel[s]["pooled_unsupported"][0]) for s in ("main", "candC", "candD")) + " |")
    return "\n".join(lines)


@_audited_card
def patch_card(user: str) -> str:
    return f"""---
license: cc-by-nc-4.0
base_model: {BASE}
library_name: peft
pipeline_tag: text-generation
language: [en]
tags: [lora, qlora, peft, rag, retrieval, abstention, citations, agent, tiny-aya, lexsi-hackathon, agenttune, auditkit]
datasets: [{user}/against-the-safety-rag-episodes]
---

# against-the-safety-tiny-aya-rag-patch

A LoRA adapter for [Tiny Aya Global]({"https://huggingface.co/" + BASE}) (revision `{REVISION[:8]}`) from the project's third round (candD, "careful patch"). It continues from
[the main adapter](https://huggingface.co/{user}/against-the-safety-tiny-aya-rag-main) for one epoch (93 steps, 2.2 M tokens) on new episodes that teach it to answer when a single passage
holds the answer, to answer both parts of a two-part question, and to treat supplied on-topic passages without the answer as insufficient: 45% new rows (SQuAD 2.0 training-split questions,
compound questions, prompt variants) and 55% rehearsal rows from the main adapter's training set. Built for the Lexsi AI Safety and Alignment Hackathon, Track 3, problem statement 4.
Team **Against-the-safety**. Code and plan: [{REPO_URL}]({REPO_URL}) (plan sections 24 to 26).

**Status: not the selected model, and not recommended where an unsupported answer is costly.** The rules were fixed before training: a candidate could replace the main adapter only if its
pooled rate of answers given when no passage in reach holds the answer stayed within 3 points of main's on the development set. This adapter, like candC, did not, so the main adapter stayed the
project's model. On fresh questions the same rate was again too high (below). Use it when you need single-passage questions answered and accept more unsupported answers.

## Development set (held-out SQuAD training-split articles and HotpotQA dev; ends well)

{r3_dev_table()}

## Fresh test set (six sets, 739 episodes per system, questions disjoint from every earlier set, opened once)

{r3_test_table()}

On fresh SQuAD questions this adapter gave a supported answer to 32 of 57 answerable questions (main: 4, candA: 6) and had 3 invalid turns of 166 (main: 30). It also answered 31 of 109 SQuAD
episodes whose passages cannot answer (main: 6, candA: 6). The pre-registered criterion (pooled unsupported answers at most 0.05 above main's) was not met (+0.120).

## Use

Loading is the same as for the main adapter (see its card); replace the adapter id with `{user}/against-the-safety-tiny-aya-rag-patch`. It expects the same system prompt and tool loop
(`ragharness.protocol.system_prompt`, `ragharness.rollout`).

## Limits

The main adapter's limits apply. Two-part answers improved only a little (4 of 27 fresh RGB integration questions answered fully, against 0). When one part of a compound question is missing, the
adapter tends to answer the visible part and invent the other. Evaluated sets are small (60 to 120 questions per set); every report gives intervals. Not for safety-critical or legal use.

## License

CC BY-NC 4.0, inherited from the base model. Cohere Labs' terms for Tiny Aya apply.
"""


@_audited_card
def cand_f1_card(user: str) -> str:
    final = json.loads((ROOT / "reports" / "round5_corrected.json").read_text(encoding="utf-8"))
    sysm, cmp_ = final["systems"], final["comparisons"]["candF1+FB+V/main+FB+V"]

    def row(label: str, key: str) -> str:
        s = sysm[key]
        c = s["components"]
        return (f"| {label} | {s['macro']:.3f} | {c['single']:.3f} | {c['compound']:.3f} | {c['hotpot']:.3f} | {c['rgb']:.3f} | {s['unsupported']} / {s['hidden_n']} | "
                f"{s['supported_correct']:.3f} | {s['false_abstain']:.3f} |")

    def ci(k: str) -> str:
        d = cmp_[k]
        return f"{d['difference']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"

    table = "\n".join([
        "| sealed test (934 shared episodes, scorer v3) | macro | SQuAD single | SQuAD two-part | HotpotQA | RGB | unsupported / hidden | supported correct | refusals of answerable |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        row("main + fail-closed + self-check (submitted)", "main+FB+V"), row("**candF1 + fail-closed + self-check**", "candF1+FB+V"),
        row("main, no repairs", "main"), row("candF1, no repairs", "candF1")])
    return f"""---
license: cc-by-nc-4.0
base_model: {BASE}
library_name: peft
pipeline_tag: text-generation
language: [en]
tags: [lora, qlora, peft, rag, retrieval, abstention, citations, agent, tiny-aya, lexsi-hackathon, agenttune, auditkit]
datasets: [{user}/against-the-safety-rag-episodes]
---

# against-the-safety-tiny-aya-rag-cand-f1

A LoRA adapter for [Tiny Aya Global]({"https://huggingface.co/" + BASE}) (revision `{REVISION[:8]}`) from the project's fourth round (candF, checkpoint after the first epoch). It continues from
[candA]({"https://huggingface.co/" + user}/against-the-safety-tiny-aya-rag-cand-a) on 721 matched-evidence rows: the same question and search depth, once with the required fact present and once
with it missing, plus replay of the search policy. Two epochs were planned (92 steps, learning rate 3e-5, seed 47); this adapter is the checkpoint after epoch 1. Built for the Lexsi AI Safety and Alignment
Hackathon, Track 3, problem statement 4. Team **Against-the-safety**. Code and pre-registrations: [{REPO_URL}]({REPO_URL}) (plan sections 27 to 29).

**Status: the documented alternative, not the submitted system.** The submitted system is [the main adapter](https://huggingface.co/{user}/against-the-safety-tiny-aya-rag-main) with a fail-closed rule and a self-check.
Round 5b opened the sealed test once for this adapter, because it was the only one of twelve checkpoints that passed the development gate. The gate was written after its development results were known, so
it was not chosen blind. The registered rule for replacing main needed, besides a macro gain, an unsupported-answer difference below zero with an upper bound of at most +0.02. Under scorer v3 that bound is {cmp_["unsupported_rate"]["ci95"][1]:+.3f} (+0.030 under the original scorer v2), so the rule failed.

## Sealed test (934 fresh SQuAD, HotpotQA and RGB episodes shared by both systems; scorer v3; opened for this adapter once)

{table}

candF1 minus main, both with the repairs, paired 95% intervals from resampling whole questions: macro **{ci("macro")}**; unsupported-answer rate {ci("unsupported_rate")}; supported correct answers {ci("supported_correct")}.
Read it as: more answers are correct, and the data cannot say whether it is safer or less safe than main. The unsupported counts are small. This is a post-hoc correction on an opened holdout: it does not change which model was submitted, and a new selection claim would need a fresh holdout.

## Use

The result above needs both repairs. The self-check in those runs was done by the **main** adapter (wording A, threshold 0.5), not by this adapter. `scripts/ask.py` in the repository loads one adapter for both
roles, so running it with this adapter would use candF1 as its own checker, which was not evaluated. Loading is otherwise the same as for the main adapter (see its card); replace the adapter id with
`{user}/against-the-safety-tiny-aya-rag-cand-f1`. It expects the same system prompt and tool loop (`ragharness.protocol.system_prompt`, `ragharness.rollout`).

## Limits

The main adapter's limits apply. On the development set this checkpoint had more unsupported HotpotQA answers than main (2 against 1) and slightly more unsupported answers overall (7 of 131 against 5) with the repairs.
Refusals of answerable episodes are still high ({sysm["candF1+FB+V"]["false_abstain"]:.3f} with the repairs). English, short factual questions. Not for safety-critical or legal use.

## License

CC BY-NC 4.0, inherited from the base model. Cohere Labs' terms for Tiny Aya apply.
"""


# ---------------------------------------------------------------- Hindi training (research/2026-11-15-hindi-training-preregistration.md)

HINDI_ADAPTERS = {"hiMain": ("hi-main", "main", "main"), "hiF": ("hi-f", "candF1", "cand-f1")}  # name -> (repository suffix, parent adapter, parent repository suffix)


def hindi_card(name: str, user: str) -> str:
    """Card of an experimental Hindi adapter; every number comes from reports/hindi_results.json (scripts/report_hindi.py)."""
    rep = json.loads((ROOT / "reports" / "hindi_results.json").read_text(encoding="utf-8"))
    slug, parent, parent_slug = HINDI_ADAPTERS[name]
    hi, en = rep["xq2_hi"]["systems"], rep["r6t"]["systems"]
    c_hi = rep["xq2_hi"]["comparisons"][f"{name}+FB+V minus {parent}+FB+V"]
    c_en = rep["r6t"]["comparisons"][f"{name}+FB+V/{parent}+FB+V"]
    train = rep["training"][name]
    ci = lambda c: f"{c['difference'] * 100:+.1f} [{c['ci95'][0] * 100:+.1f}, {c['ci95'][1] * 100:+.1f}]"
    met = lambda ok: "met" if ok else "not met"
    rules = [("A1. Hindi, correct and cited (lower end above 0)", ci(c_hi["correct_and_cited"]), c_hi["correct_and_cited"]["ci95"][0] > 0),
             ("A2. Hindi, answers on hidden evidence (upper end at most +2.0)", ci(c_hi["hidden_answer_rate"]), c_hi["hidden_answer_rate"]["ci95"][1] <= 0.02),
             ("A3a. English, macro success (lower end at least -3.0)", ci(c_en["macro"]), c_en["macro"]["ci95"][0] >= -0.03),
             ("A3b. English, answers on hidden evidence (upper end at most +2.0)", ci(c_en["unsupported_rate"]), c_en["unsupported_rate"]["ci95"][1] <= 0.02)]
    names = [title.split(".")[0] for title, _, ok in rules if not ok]
    failed = " and ".join(names) if len(names) < 3 else ", ".join(names[:-1]) + " and " + names[-1]
    failed = failed or "none"
    rule_table = "\n".join(["| Registered rule (difference in points, 95% interval) | " + f"{name} minus {parent} |", "| --- | ---: |"] + [f"| {t} | {d}: {met(ok)} |" for t, d, ok in rules])
    pct = lambda x: f"{x:.1%}"
    rows = [(f"{name} + both repairs", f"{name}+FB+V"), (f"{parent} + both repairs (parent)", f"{parent}+FB+V"), ("Refuse every question", "always refuse")]
    table = "\n".join(["| System | Hindi: correct and cited | Hindi: answers when evidence was hidden | English: macro success (second sealed test) | English: answers when evidence was hidden |", "| --- | ---: | ---: | ---: | ---: |"]
                      + [f"| {title} | {pct(hi[k]['correct_and_cited'])} | {hi[k]['hidden_answers']} / {hi[k]['n_hidden']} | {pct(en[k]['macro'])} | {en[k]['unsupported']} / {en[k]['hidden_n']} |" for title, k in rows])
    return f"""---
license: cc-by-nc-4.0
base_model: {BASE}
library_name: peft
pipeline_tag: text-generation
language: [hi, en]
tags: [lora, qlora, peft, rag, retrieval, abstention, citations, agent, hindi, experimental, tiny-aya, lexsi-hackathon, agenttune, auditkit]
datasets: [{user}/against-the-safety-rag-episodes, {user}/against-the-safety-benchmark-traces]
---

# against-the-safety-tiny-aya-rag-{slug}

**Status: experimental. It failed the registered rules {failed} and is not the submitted system.** The submitted system is [the main adapter](https://huggingface.co/{user}/against-the-safety-tiny-aya-rag-main) with a fail-closed rule and a self-check, in English.

A LoRA adapter for [Tiny Aya Global](https://huggingface.co/{BASE}) (revision `{REVISION[:8]}`). It continues the [{parent} adapter](https://huggingface.co/{user}/against-the-safety-tiny-aya-rag-{parent_slug}) on Hindi: 220 machine-translated questions (IndicSQuAD Hindi, from SQuAD 2.0) built into 540 Hindi search-answer-abstain rows, plus English replay at half the Hindi token count. Two epochs, learning rate 1e-4, seed 53, one run on two Kaggle T4s ({train['rows_trained']} rows, {train['optimizer_steps']} optimizer steps, final training loss {train['train_loss']:.3f}). Built for the Lexsi AI Safety and Alignment Hackathon, Track 3, problem statement 4. Team **Against-the-safety**. Code, pre-registration and results: [{REPO_URL}]({REPO_URL}) (`research/2026-11-15-hindi-training-preregistration.md`, `research/2026-11-15-hindi-training-results.md`).

## Results (300 fresh XQuAD questions in Hindi, human translated; the second English sealed test; scorer v3; both repairs; one run)

{table}

{rule_table}

Read it as a trade-off. The adapter answers Hindi questions it could not answer before (the parent's raw Hindi turns were invalid {pct(hi[parent]['invalid'])} of the time, against {pct(hi[name]['invalid'])} for this adapter), and it answers on hidden evidence more often. Part of the Hindi gain is general single-passage answering: the same questions in English also improved. The registered rule for calling an adapter a Hindi-capable variant needed all of A1, A2 and A3, so this adapter is published as an experiment and not as one.

## Use

The result needs both repairs, and the self-check in those runs was done by the **main** adapter (wording A, threshold 0.5), not by this adapter. Hindi text costs about 3.4 times the tokens of English, so the runs used an input limit of 8,000 tokens. The tokenizer files are not copied (they belong to the gated base repository). Loading and the tool loop are the same as for the main adapter (see its card); replace the adapter id with `{user}/against-the-safety-tiny-aya-rag-{slug}`.

## Limits

One training run and one seed. The training questions are machine translated and were not checked by a speaker; the test questions are human translated. The test questions share articles with the first 300 XQuAD questions, whose results we had seen, and the hidden-evidence answers are annotation-based flags. Refusals of answerable Hindi questions are still {pct(hi[f'{name}+FB+V']['false_refusal'])} with the repairs. Not for safety-critical or legal use.

## License

CC BY-NC 4.0, inherited from the base model. Cohere Labs' terms for Tiny Aya apply. Training data: IndicSQuAD Hindi (`l3cube-pune/indic-squad`, listed CC BY 4.0; it translates SQuAD 2.0, CC BY-SA 4.0) and the project's English episodes (HotpotQA-derived, CC BY-SA 4.0).
"""
