"""Dataset assembly for training.

Sources are either HF Hub datasets (the normal path, since Kaggle pulls
everything from `RohanRamesh/*`) or local JSONL manifests (smoke tests).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["load_manifest_dataset", "load_hub_dataset", "concat", "apply_subset"]

SR = 16_000


def load_manifest_dataset(path: str | Path):
    """Build a `datasets.Dataset` from a JSONL manifest with a `wav` column."""
    from datasets import Audio, Dataset

    from ..manifest import read_jsonl

    rows = [r for r in read_jsonl(path) if r.get("wav")]
    if not rows:
        raise SystemExit(f"{path}: no rows with a `wav` field")
    ds = Dataset.from_list(
        [{"audio": r["wav"], "text": r["text"], "utt_id": r["utt_id"]} for r in rows]
    )
    return ds.cast_column("audio", Audio(sampling_rate=SR))


def load_hub_dataset(repo: str, config: str | None = None, split: str = "train", token: str | None = None):
    from datasets import Audio, load_dataset

    ds = load_dataset(repo, config, split=split, token=token)
    if "audio" in ds.column_names:
        ds = ds.cast_column("audio", Audio(sampling_rate=SR))
    return ds


def apply_subset(ds, ids_path: str | Path, id_column: str = "utt_id"):
    """Filter a dataset down to the ids listed in a JSON array (Train_T1)."""
    keep = set(json.loads(Path(ids_path).read_text(encoding="utf-8")))
    before = len(ds)
    ds = ds.filter(lambda r: r[id_column] in keep, desc="subset")
    if len(ds) == 0:
        raise SystemExit(f"subset {ids_path} matched 0 of {before} rows on {id_column!r}")
    print(f"[subset] {len(ds):,} / {before:,} rows kept")
    return ds


def concat(datasets: list[Any], columns: tuple[str, ...] = ("audio", "text", "utt_id")):
    """Concatenate datasets after projecting to a common schema."""
    from datasets import concatenate_datasets

    projected = []
    for ds in datasets:
        drop = [c for c in ds.column_names if c not in columns]
        projected.append(ds.remove_columns(drop) if drop else ds)
    return concatenate_datasets(projected)
