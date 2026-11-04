"""AgentTune training entry for our trajectories.

`build_trainer` calls AgentTune's `create_distill_trainer` (EventLog -> `Project.distill` -> `trainer_factory`).
The factory here is `RowCollector`, a GPU-free stand-in that turns AgentTune's SFT rows into Tiny Aya
prompt/completion rows and verifies each against the pinned chat template. The QLoRA factory used for the real
run plugs into the same `trainer_factory` hook and receives the same rows.
"""

from __future__ import annotations

from agenttune.agentic.distill_trainer import create_distill_trainer
from agenttune.agentic.events import EventLog

from . import rows


class _NoTrain:
    def __init__(self, owner: "RowCollector"):
        self.owner = owner

    def train(self):
        return {"n_rows": len(self.owner.rows), "by_kind": self.owner.by_kind()}


class RowCollector:
    def __init__(self, system: str):
        self.system = system
        self.rows: list[dict] = []
        self.checked: list[dict] = []

    def __call__(self, *, model, train_dataset, **kwargs):
        self.student = model
        self.agenttune_rows = list(train_dataset)
        self.rows = [d for r in self.agenttune_rows for d in rows.decisions(r, self.system)]
        self.checked = [rows.check_row(d) for d in self.rows]
        return _NoTrain(self)

    def by_kind(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.checked:
            out[c["kind"]] = out.get(c["kind"], 0) + 1
        return out


def build_trainer(logs: list[EventLog], factory, student: str = "CohereLabs/tiny-aya-global", output_dir: str = "runs/unused"):
    """AgentTune's own entry point; `factory` decides what happens to the rows."""
    return create_distill_trainer(student, trajectories=logs, trainer_factory=factory, output_dir=output_dir)
