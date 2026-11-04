"""Load built episodes as AgentTune EventLogs and select the exact rows of a training set.

The mixtures in mixture.py select ROWS (one per decision), but AgentTune consumes whole EventLogs. Selection is
therefore a set of (parent_qid, variant, step) keys; the factory keeps only decisions whose key is in the set.
"""

from __future__ import annotations

import json
from pathlib import Path

from agenttune.agentic.events import Event, EventKind, EventLog

Key = tuple[str, str, int]


def row_key(meta: dict) -> Key:
    return (meta["parent_qid"], meta["variant"], int(meta["step"]))


def read_keys(path: Path) -> set[Key]:
    return {row_key(json.loads(l)["meta"]) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


def read_system(path: Path) -> str:
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    return first["prompt"][0]["content"]


def load_episodes(episodes_dir: Path, split: str) -> tuple[list[EventLog], list[dict]]:
    """EventLogs and their metas, in the order the builder wrote them."""
    logs, metas = [], []
    log_lines = (episodes_dir / f"eventlogs_{split}.jsonl").read_text(encoding="utf-8").splitlines()
    meta_lines = (episodes_dir / f"episodes_{split}.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == len(meta_lines)
    for ll, ml in zip(log_lines, meta_lines):
        d = json.loads(ll)
        log = EventLog(tier=d["tier"], id=d["id"])
        for e in d["events"]:
            log.append(Event(EventKind(e["kind"]), e["payload"], scope=e["scope"]))
        logs.append(log)
        metas.append(json.loads(ml)["meta"])
    return logs, metas


def episodes_for_keys(logs: list[EventLog], metas: list[dict], keys: set[Key]) -> tuple[list[EventLog], list[dict]]:
    """Keep the episodes that own at least one selected row."""
    owners = {(k[0], k[1]) for k in keys}
    keep = [i for i, m in enumerate(metas) if (m["parent_qid"], m["variant"]) in owners]
    return [logs[i] for i in keep], [metas[i] for i in keep]
