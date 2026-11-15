"""The general-capability scorer: prompt format, letter tokens and bookkeeping, with a scripted model on CPU."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import general_capability as gc  # noqa: E402


class FakeTok:
    pad_token_id = 0

    def __init__(self):
        self.vocab = {}

    def encode(self, text, add_special_tokens=True):
        ids = [self.vocab.setdefault(w, len(self.vocab) + 1) for w in text.split()]
        return ([99999] if add_special_tokens else []) + ids


class FakeModel:
    """Always prefers the letter B; records the width of every batch it was given."""

    device = "cpu"

    def __init__(self, tok):
        self.tok, self.widths = tok, []

    def __call__(self, input_ids, attention_mask, position_ids, use_cache, logits_to_keep):
        import torch

        assert logits_to_keep == 1 and input_ids.shape == attention_mask.shape == position_ids.shape
        self.widths.append(input_ids.shape[1])
        out = torch.zeros((input_ids.shape[0], 1, 100000))
        out[:, :, self.tok.vocab["B"]] = 5.0

        class R:
            logits = out
        return R()


def row(i, gold, bench="mmlu", n=4):
    return {"id": f"q{i}", "bench": bench, "subject": "college_biology", "question": f"Question number {i} " + "word " * (i % 7), "choices": [f"c{k}" for k in range(n)], "gold": gold}


def test_prompt_has_the_usual_letter_format():
    p = gc.prompt(row(1, 0))
    assert p.startswith("The following are multiple choice questions (with answers) about college biology.")
    assert "\nA. c0\nB. c1\nC. c2\nD. c3\nAnswer:" in p and p.endswith("Answer:")
    assert gc.prompt(row(1, 0, bench="arc_c")).startswith("The following is a multiple choice science question (grade school).")


def test_letter_ids_come_from_the_tokenizers_own_split():
    tok = FakeTok()
    ids = gc.letter_ids(tok, 4)
    assert len(set(ids)) == 4 and ids[1] == tok.vocab["B"]


@pytest.mark.skipif(not (ROOT / "runs/kaggle/ats-train-both/out/main/adapter/tokenizer.json").exists(), reason="tokenizer files not available")
def test_real_tokenizer_gives_one_space_letter_token_per_choice():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(ROOT / "runs/kaggle/ats-train-both/out/main/adapter"))
    ids = gc.letter_ids(tok, 13)
    assert [tok.decode([i]) for i in ids] == [" " + c for c in "ABCDEFGHIJKLM"]


def test_scores_every_question_once_across_batches(tmp_path):
    tok = FakeTok()
    model = FakeModel(tok)
    rows = [row(i, i % 4) for i in range(40)]
    with (tmp_path / "out.jsonl").open("w", encoding="utf-8") as f:
        summary = gc.evaluate(model, tok, rows, "base", batch_tokens=120, max_input=2048, out_file=f)
    lines = [json.loads(l) for l in (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 40 and len({l["id"] for l in lines}) == 40 and len(model.widths) > 1
    assert all(l["pred"] == 1 for l in lines)  # the scripted model always says B
    assert summary["accuracy"] == pytest.approx(sum(r["gold"] == 1 for r in rows) / 40) and summary["questions"] == 40


def test_the_built_sets_have_the_registered_sizes_and_valid_gold():
    manifest = ROOT / "data/general/manifest.json"
    if not manifest.exists():
        pytest.skip("data/general not built")
    sizes = {k: v["questions"] for k, v in json.loads(manifest.read_text(encoding="utf-8")).items()}
    assert sizes == {"mmlu": 14042, "arc_c": 1172, "arc_e": 2376, "truthfulqa": 817, "gmmlu_en": 400, "gmmlu_hi": 400, "gmmlu_es": 400}
    for name in sizes:
        for line in (ROOT / "data/general" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()[:200]:
            r = json.loads(line)
            assert 0 <= r["gold"] < len(r["choices"]) and r["question"].strip() and all(c.strip() for c in r["choices"])
