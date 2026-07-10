"""Publish manifests to the Hub as Parquet shards with FLAC-encoded audio.

FLAC roughly halves the payload versus raw PCM: 22 h of 16 kHz/16-bit speech is
2.53 GB as WAV and ~1.4 GB as FLAC, with no loss.

Also provides `verify_round_trip`, which re-downloads a pushed config and checks
row count, total duration, and sampling rate against the source manifest. Run it
before committing a multi-hour training session to a corrupted upload.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

__all__ = ["push_manifest", "push_text_manifest", "verify_round_trip", "main"]

SR = 16_000


def _dataset_from_manifest(path: Path, *, with_audio: bool):
    from datasets import Audio, Dataset

    from ..manifest import read_jsonl

    rows = list(read_jsonl(path))
    if not rows:
        raise SystemExit(f"{path} is empty")

    if with_audio:
        missing = [r for r in rows if not r.get("wav")]
        if missing:
            raise SystemExit(f"{path}: {len(missing)} rows lack a `wav` field")
        cols = ("utt_id", "text", "dur", "speaker", "lang", "bigram", "matrix_lang")
        # Every row must carry every key, or pyarrow rejects the schema. Keep
        # None as an explicit null rather than dropping the field.
        recs = [
            {"audio": r["wav"], **{c: r.get(c) for c in cols}}
            for r in rows
        ]
        # Drop columns that are null for every row (e.g. `speaker` on real data).
        empty = [c for c in cols if all(r[c] is None for r in recs)]
        if empty:
            recs = [{k: v for k, v in r.items() if k not in empty} for r in recs]
        ds = Dataset.from_list(recs)
        return ds.cast_column("audio", Audio(sampling_rate=SR))

    # Text-only manifests are heterogeneous: bigrams_valid.jsonl has
    # {bigram, hi_word, en_word, checks}, sentences.jsonl has {sent_id, text,
    # bigram, matrix_lang}, mucs_train.jsonl has {utt_id, text, dur}. Pass every
    # key through, filling gaps with None so the schema is uniform.
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    return Dataset.from_list([{k: r.get(k) for k in keys} for r in rows])


def push_manifest(
    manifest: Path,
    repo_id: str,
    config: str,
    *,
    with_audio: bool = True,
    private: bool = True,
    token: str | None = None,
    max_shard_size: str = "500MB",
) -> int:
    ds = _dataset_from_manifest(manifest, with_audio=with_audio)
    print(f"[push] {manifest} -> {repo_id}:{config}  ({len(ds):,} rows, audio={with_audio})")
    ds.push_to_hub(
        repo_id,
        config_name=config,
        private=private,
        token=token,
        max_shard_size=max_shard_size,
    )
    return len(ds)


def push_text_manifest(manifest: Path, repo_id: str, config: str, **kw) -> int:
    return push_manifest(manifest, repo_id, config, with_audio=False, **kw)


def verify_round_trip(
    manifest: Path, repo_id: str, config: str, *, token: str | None = None, with_audio: bool = True
) -> dict:
    """Re-download and compare against the source manifest. Raises on mismatch."""
    from datasets import load_dataset

    from ..manifest import read_jsonl, total_hours

    src = list(read_jsonl(manifest))
    got = load_dataset(repo_id, config, split="train", token=token)

    if len(got) != len(src):
        raise SystemExit(f"row count mismatch: hub={len(got):,} manifest={len(src):,}")

    src_h = total_hours(src)
    if "dur" in got.column_names:
        got_h = sum(d or 0.0 for d in got["dur"]) / 3600
        if abs(got_h - src_h) > 0.01:
            raise SystemExit(f"duration mismatch: hub={got_h:.3f}h manifest={src_h:.3f}h")

    if with_audio:
        sr = got[0]["audio"]["sampling_rate"]
        if sr != SR:
            raise SystemExit(f"sampling rate is {sr}, expected {SR}")

    print(f"[verify] {repo_id}:{config} OK - {len(got):,} rows, {src_h:.2f} h, {SR} Hz")
    return {"rows": len(got), "hours": src_h}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--repo", required=True, help="e.g. RohanRamesh/hi-en-synth-cs")
    ap.add_argument("--config", required=True)
    ap.add_argument("--text-only", action="store_true")
    ap.add_argument("--public", action="store_true")
    # Never pass a token in argv: it lands in tracebacks and in `ps` output.
    ap.add_argument("--token", default=None, help="prefer the HF_TOKEN env var")
    ap.add_argument("--verify", action="store_true", help="round-trip check after push")
    args = ap.parse_args(argv)

    import os

    token = args.token or os.environ.get("HF_TOKEN")
    with_audio = not args.text_only
    push_manifest(
        args.manifest, args.repo, args.config,
        with_audio=with_audio, private=not args.public, token=token,
    )
    if args.verify:
        verify_round_trip(args.manifest, args.repo, args.config, token=token, with_audio=with_audio)
    return 0


if __name__ == "__main__":
    sys.exit(main())
