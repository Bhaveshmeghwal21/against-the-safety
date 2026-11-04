"""The trainer's row selection (episodes + keys) must reproduce data/processed/training/{main,control}.jsonl exactly."""

import json
from pathlib import Path

import pytest

from ragharness import qlora, training

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "data" / "processed" / "training"
EPISODES = ROOT / "data" / "processed" / "episodes"

pytestmark = pytest.mark.skipif(not (TRAINING / "main.jsonl").exists(), reason="training sets not built")


@pytest.mark.parametrize("name", ["main", "control"])
def test_selected_rows_match_training_file(name):
    path = TRAINING / f"{name}.jsonl"
    expected = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    keys = training.read_keys(path)
    assert len(keys) == len({training.row_key(r["meta"]) for r in expected}), "row keys must be unique"

    logs, metas = training.load_episodes(EPISODES, "train")
    logs, metas = training.episodes_for_keys(logs, metas, keys)
    system = training.read_system(TRAINING / "main.jsonl")
    sft_rows = [row for log in logs for row in log.as_dataset_rows("sft")]
    got = qlora.to_hf_rows(sft_rows, system, metas, keys)

    assert len(got) == len(expected)
    as_pair = lambda r: json.dumps([r["prompt"], r["completion"]], sort_keys=True)
    assert sorted(map(as_pair, got)) == sorted(map(as_pair, expected))


def test_control_has_no_missing_evidence_or_abstentions():
    rows = [json.loads(l) for l in (TRAINING / "control.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(r["meta"]["answerable"] for r in rows)
    assert not any(r["meta"]["kind"] == "abstain" for r in rows)
    assert not any(r["meta"]["state"] == "S4" for r in rows)


def test_no_row_mentions_dev_or_test_support():
    man = json.loads((ROOT / "data" / "manifests" / "support_seen_in_sft.json").read_text(encoding="utf-8"))
    assert man["dev"]["unseen"] and len(man["dev"]["partly_seen"]) == 0 and len(man["test"]["partly_seen"]) == 0
