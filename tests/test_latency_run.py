"""The latency script's bookkeeping, run on CPU with a scripted generator in place of the model."""

import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import latency_run  # noqa: E402


def test_pick_is_deterministic_and_independent_of_the_arm():
    jobs = list(range(100))
    a, b = latency_run.pick(jobs, 10, 61), latency_run.pick(jobs, 10, 61)
    assert a == b and len(set(a)) == 10 and a == sorted(a)
    assert latency_run.pick(jobs, 10, 62) != a
    assert latency_run.pick(jobs[:4], 10, 61) == jobs[:4]


def test_meter_accumulates_and_resets():
    m = latency_run.Meter()
    f = m.wrap(lambda x: x + 1)
    assert f(1) == 2 and f(2) == 3 and m.calls == 2 and m.total >= 0
    m.reset()
    assert (m.total, m.calls) == (0.0, 0)


def test_summary_uses_auditkit_definitions():
    s = latency_run.summarize([100.0, 200.0, 300.0, 400.0, 1000.0], 2.0)
    assert s["latency_ms"]["p50"] == 300.0 and s["latency_ms"]["p95"] == 400.0 and s["latency_ms"]["max"] == 1000.0
    assert s["throughput"]["total_requests"] == 5 and s["throughput"]["rps"] == pytest.approx(5 / 2.0)


@pytest.mark.skipif(not (ROOT / "data/bench/r5t_hotpot/corpus.sqlite").exists(), reason="needs the built r5t benchmark folders")
def test_end_to_end_with_scripted_model(tmp_path, monkeypatch):
    from ragharness import checker, generate, protocol

    class FakeModel:
        class _Ctx:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def disable_adapter(self):
            return self._Ctx()

    class FakeGenerator:
        def __init__(self, *a, **k):
            self.output_tokens = 0

        def __call__(self, messages):
            self.output_tokens += 5
            return protocol.format_search("question") if len(messages) == 2 else protocol.format_answer("x", ["c00001"])

    ask_module = types.SimpleNamespace(load_model_for_system=lambda adapter: (FakeModel(), object()))
    monkeypatch.setitem(sys.modules, "ask", ask_module)
    monkeypatch.setattr(generate, "HFGenerator", FakeGenerator)
    monkeypatch.setattr(checker, "support_probability", lambda *a, **k: 0.9)
    monkeypatch.setattr(sys, "argv", ["latency_run.py", "--adapter", "unused", "--out", str(tmp_path), "--per-part", "2", "--warmup", "1"])
    latency_run.main()
    summary = json.loads((tmp_path / "latency_summary.json").read_text(encoding="utf-8"))
    assert set(summary["arms"]) == {"base", "submitted"}
    for arm, stats in summary["arms"].items():
        rows = [json.loads(l) for l in (tmp_path / f"requests_{arm}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert stats["requests"] == len(rows) == 6 and {r["part"] for r in rows} == set(latency_run.PARTS)
        assert all(r["output_tokens"] >= 5 and r["model_calls"] >= 1 and r["total_ms"] >= r["generation_ms"] for r in rows)
    base_rows = [json.loads(l) for l in (tmp_path / "requests_base.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(r["checks"] == 0 for r in base_rows)
