"""Pull ~15 h each of Common Voice Hindi and English (the paper's Train_H / Train_E).

Mozilla moved Common Voice to the Mozilla Data Collective in Oct 2025, and
`mozilla-foundation/common_voice_17_0` on the Hub is now an empty stub. We use
the ungated CC0 mirror `fsicoli/common_voice_17_0`.

WHY WE DO NOT USE `load_dataset`
--------------------------------
That mirror ships a loading script whose declared features do not match its own
TSVs (it announces a `variant` column the data lacks), so streaming dies with
"column names don't match". Loading scripts were also removed outright in
`datasets >= 4`. Instead we read the repo's actual layout:

    transcript/<lang>/<split>.tsv        path, sentence, client_id, ...
    audio/<lang>/<split>/<lang>_<split>_<i>.tar     <lang>_<split>_<i>/xxx.mp3

and stream the tars over HTTP with `tarfile` in stream mode, **aborting as soon
as the hour target is met**. English `train` alone is 45 GB across 28 shards; we
touch a few hundred MB of the first one.

SPLIT ORDER MATTERS
-------------------
We walk dev -> test -> train -> other, smallest first, because CV's `train`
metadata is enormous while `dev`/`test` are tiny and still huge in hours:

    en   dev  0.72 GB   ~50 h   tsv 4.9 MB      <- 15 h comes from here alone
    en   train 45 GB  ~3135 h   tsv 363 MB      <- never touched
    hi   dev+test+train         tsv 3.8 MB      <- 20.6 h available, enough for 15 h

These are Common Voice's own dev/test splits, not ours. Using them as auxiliary
monolingual *training* data is sound: they are out-of-domain with respect to
MUCS and are never evaluated on.

Because a streamed tar is consumed in order, this is a *prefix* sample, not a
uniform one. `--max-per-speaker` bounds how much any single `client_id` can
contribute, which is what actually matters for an out-of-domain auxiliary set.
Keep it generous: Hindi has only ~20.6 h available in total, so a tight cap can
starve the 15 h target. Pass 0 to disable the cap entirely.

Only needed for M8 (synthetic + monolingual).
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

from ..manifest import Utt, write_jsonl

__all__ = ["stream_subset", "main"]

REPO = "fsicoli/common_voice_17_0"
TARGET_SR = 16_000
MIN_DUR, MAX_DUR = 1.0, 30.0
# Smallest-metadata-first. `invalidated` is excluded (failed CV validation).
SPLIT_ORDER = ("dev", "test", "train", "other")


def _to_int16_16k(x: np.ndarray, sr: int) -> np.ndarray:
    import soxr

    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != TARGET_SR:
        x = soxr.resample(x, sr, TARGET_SR)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.0:
        x = x / peak
    return np.clip(x * 32767.0, -32768, 32767).astype(np.int16)


def _transcripts(lang: str, split: str, token: str | None) -> dict[str, tuple[str, str]]:
    """{basename.mp3: (sentence, client_id)} for one split."""
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(REPO, f"transcript/{lang}/{split}.tsv", repo_type="dataset", token=token)
    out: dict[str, tuple[str, str]] = {}
    with open(p, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            sent = (row.get("sentence") or "").strip()
            if sent:
                out[row["path"]] = (sent, row.get("client_id", ""))
    return out


def _repo_files(token: str | None) -> list[str]:
    global _FILES_CACHE
    if _FILES_CACHE is None:
        from huggingface_hub import HfApi

        _FILES_CACHE = HfApi().list_repo_files(REPO, repo_type="dataset", token=token)
    return _FILES_CACHE


_FILES_CACHE: list[str] | None = None


def _shards(lang: str, split: str, token: str | None) -> list[str]:
    files = _repo_files(token)  # 2,152 entries; fetch once, not per shard
    prefix = f"audio/{lang}/{split}/"
    tars = [f for f in files if f.startswith(prefix) and f.endswith(".tar")]

    def idx(f: str) -> int:
        m = re.search(r"_(\d+)\.tar$", f)
        return int(m.group(1)) if m else 0

    return sorted(tars, key=idx)


def _iter_tar(path_in_repo: str, token: str | None):
    """Yield (member_basename, bytes) while downloading only what we consume."""
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(REPO, path_in_repo, repo_type="dataset")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        with tarfile.open(fileobj=r.raw, mode="r|") as tf:
            for m in tf:
                if not m.isfile() or not m.name.endswith(".mp3"):
                    continue
                f = tf.extractfile(m)
                if f is not None:
                    yield Path(m.name).name, f.read()


def stream_subset(
    lang: str,
    hours: float,
    audio_dir: Path,
    *,
    token: str | None = None,
    splits: tuple[str, ...] = SPLIT_ORDER,
    max_per_speaker: int = 200,
) -> list[Utt]:
    import soundfile as sf

    audio_dir.mkdir(parents=True, exist_ok=True)
    target = hours * 3600.0
    acc = 0.0
    rows: list[Utt] = []
    per_speaker: Counter = Counter()
    skipped = 0

    bar = tqdm(total=int(target), desc=f"cv:{lang}", unit="s")
    for split in splits:
        if acc >= target:
            break
        try:
            trans = _transcripts(lang, split, token)
        except Exception as e:  # noqa: BLE001 - a missing split is not fatal
            print(f"[cv:{lang}] no transcript for split {split!r} ({type(e).__name__}); skipping")
            continue

        for shard in _shards(lang, split, token):
            if acc >= target:
                break
            for name, blob in _iter_tar(shard, token):
                if acc >= target:
                    break
                hit = trans.get(name)
                if hit is None:
                    skipped += 1
                    continue
                sentence, client = hit
                if max_per_speaker and client and per_speaker[client] >= max_per_speaker:
                    skipped += 1
                    continue
                try:
                    x, sr = sf.read(io.BytesIO(blob), dtype="float32")
                except Exception:  # noqa: BLE001 - a corrupt clip is not fatal
                    skipped += 1
                    continue

                pcm = _to_int16_16k(x, sr)
                dur = pcm.size / TARGET_SR
                if not (MIN_DUR <= dur <= MAX_DUR):
                    skipped += 1
                    continue

                utt_id = f"cv_{lang}_{Path(name).stem}"
                wav = audio_dir / f"{utt_id}.wav"
                sf.write(str(wav), pcm, TARGET_SR, subtype="PCM_16")
                rows.append(
                    Utt(utt_id=utt_id, text=sentence, dur=round(dur, 3),
                        wav=wav.as_posix(), lang=lang, speaker=client or None)
                )
                per_speaker[client] += 1
                acc += dur
                bar.update(int(dur))
    bar.close()

    print(f"[cv:{lang}] kept {len(rows):,} utts, {acc/3600:.2f} h "
          f"from {len(per_speaker):,} speakers (skipped {skipped:,})")
    if acc < target * 0.95:
        print(f"[cv:{lang}] WARNING: only {acc/3600:.2f} h of the {hours} h target; "
              f"splits {splits} exhausted")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--langs", nargs="+", default=["hi", "en"])
    ap.add_argument("--hours", type=float, default=15.0)
    ap.add_argument("--audio-root", type=Path, default=Path("data/cv"))
    ap.add_argument("--manifest-dir", type=Path, default=Path("manifests"))
    ap.add_argument("--splits", nargs="+", default=list(SPLIT_ORDER))
    ap.add_argument("--max-per-speaker", type=int, default=200, help="0 disables the cap")
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args(argv)

    for lang in args.langs:
        rows = stream_subset(
            lang, args.hours, args.audio_root / lang,
            token=args.hf_token, splits=tuple(args.splits),
            max_per_speaker=args.max_per_speaker,
        )
        out = args.manifest_dir / f"cv_{lang}.jsonl"
        write_jsonl(out, rows)
        print(f"[cv:{lang}] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
