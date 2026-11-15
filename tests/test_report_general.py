"""The retention report: it needs every question from every system, and applies the registered margin to the paired interval."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import report_general as rg  # noqa: E402


def write_run(root: Path, drop: int = 0, main_loss: float = 0.0):
    (root / "data/general").mkdir(parents=True)
    run = root / "run"
    run.mkdir()
    manifest = {}
    for b, n in zip(rg.BENCHES, (600, 200, 200, 100, 80, 80, 80)):
        manifest[b] = {"questions": n, "license": "test", "source": "test"}
        with (run / f"{b}.jsonl").open("w", encoding="utf-8") as f:
            for system in rg.SYSTEMS:
                for i in range(n - (drop if system == "candF1" and b == "mmlu" else 0)):
                    base_right = i % 5 != 0  # 80% right
                    right = base_right and not (system == "main" and i % 100 < main_loss * 100)
                    f.write(json.dumps({"system": system, "bench": b, "id": f"{b}-{i}", "correct": bool(right)}) + "\n")
    (root / "data/general/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run / "summary_mmlu.json").write_text(json.dumps({"gpu": "T4", "torch": "x", "model_source": "m", "results": [{"bench": "mmlu", "system": "base", "seconds": 1.0}]}), encoding="utf-8")
    return run


def test_a_missing_question_stops_the_report(tmp_path, monkeypatch):
    run = write_run(tmp_path, drop=3)
    monkeypatch.setattr(rg, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["report_general.py", "--run", str(run), "--out", str(tmp_path / "out")])
    with pytest.raises(SystemExit, match="candF1 has 597 of 600"):
        rg.main()


def test_unchanged_systems_are_retained_and_a_ten_point_loss_is_not(tmp_path, monkeypatch):
    run = write_run(tmp_path)
    monkeypatch.setattr(rg, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["report_general.py", "--run", str(run), "--out", str(tmp_path / "out")])
    rg.main()
    rep = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert all(r["retained"] for a in rep["rules"].values() for r in a.values())
    assert rep["benchmarks"]["mmlu"]["difference_from_base"]["main"]["difference"] == 0.0

    other = tmp_path / "second"
    other.mkdir()
    run2 = write_run(other, main_loss=0.10)
    monkeypatch.setattr(rg, "ROOT", other)
    monkeypatch.setattr(sys, "argv", ["report_general.py", "--run", str(run2), "--out", str(other / "out")])
    rg.main()
    rep2 = json.loads((other / "out.json").read_text(encoding="utf-8"))
    assert not rep2["rules"]["main"]["mmlu"]["retained"] and rep2["rules"]["candF1"]["mmlu"]["retained"]


def test_bootstrap_is_seeded():
    import numpy as np

    arrays = {s: np.array([1.0, 0.0, 1.0, 1.0] * 50) for s in rg.SYSTEMS}
    assert rg.bootstrap(arrays) == rg.bootstrap(arrays)
