"""JSONL manifests: the data contract between every stage of the pipeline.

Stages never pass Python objects to each other. They read a manifest, write a
manifest, and are therefore independently resumable and portable between the
laptop, Kaggle, and the Hub.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Iterator

__all__ = ["Utt", "read_jsonl", "write_jsonl", "append_jsonl", "total_hours"]


@dataclass(slots=True)
class Utt:
    """One utterance. `wav` is None for text-only manifests (e.g. MUCS train)."""

    utt_id: str
    text: str
    dur: float | None = None
    wav: str | None = None
    lang: str | None = None
    speaker: str | None = None
    sent_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Utt":
        known = {f.name for f in fields(cls)} - {"extra"}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(**{k: v for k, v in d.items() if k in known}, extra=extra)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        extra = d.pop("extra") or {}
        d = {k: v for k, v in d.items() if v is not None}
        d.update(extra)
        return d


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any] | Utt]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            if isinstance(row, Utt):
                row = row.to_dict()
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any] | Utt]) -> int:
    """Append mode. Used by resumable stages that checkpoint as they go."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            if isinstance(row, Utt):
                row = row.to_dict()
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def total_hours(rows: Iterable[dict[str, Any]]) -> float:
    return sum(r.get("dur") or 0.0 for r in rows) / 3600.0
