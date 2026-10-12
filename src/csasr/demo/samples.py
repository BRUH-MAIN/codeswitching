"""Cut demo excerpts out of the MUCS test recordings.

The models were scored on whole recordings (`eval/grouping.py` explains why:
isolated 6-second clips lose the context that code-switching needs, and every
model looks far worse). An excerpt is therefore a run of *consecutive* segments
from one recording, about 25 s long -- one Whisper window -- with the matching
reference text. Excerpts are chosen with a seed, so a demo can be repeated.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np

from ..eval.grouping import recording_id, segment_index
from .registry import REPO

__all__ = ["Clip", "pick_excerpts", "load_test_excerpts"]

SR = 16_000
REAL_HF = "RohanRamesh/mucs-he-cs"


@dataclass
class Clip:
    clip_id: str
    audio: np.ndarray | None  # 16 kHz mono float32; None for text-only clips
    ref: str | None          # reference transcript, when there is one
    source: str = ""         # where it came from, for the report

    @property
    def seconds(self) -> float:
        return 0.0 if self.audio is None else len(self.audio) / SR


def pick_excerpts(utt_ids: list[str], durs: list[float], n: int, seconds: float,
                  seed: int = 0, recordings: list[str] | None = None) -> list[list[int]]:
    """Choose `n` runs of consecutive segments, each of at least `seconds` where possible.

    Returns lists of row indices into the inputs, one list per excerpt. Excerpts
    come from distinct recordings while there are enough (so a demo shows
    different speakers and topics), and from `recordings` if given.
    """
    by_reco: dict[str, list[int]] = {}
    for i, u in enumerate(utt_ids):
        by_reco.setdefault(recording_id(u), []).append(i)
    for idx in by_reco.values():
        idx.sort(key=lambda i: segment_index(utt_ids[i]))

    names = sorted(by_reco)
    if recordings:
        unknown = [r for r in recordings if r not in by_reco]
        if unknown:
            raise SystemExit(f"unknown recording id(s): {', '.join(unknown)}")
        names = list(recordings)
    rng = random.Random(seed)
    order = rng.sample(names, len(names))
    chosen = [order[i % len(order)] for i in range(n)]

    out = []
    for reco in chosen:
        idx = by_reco[reco]
        # latest start from which `seconds` of audio still follows
        last, acc = 0, 0.0
        for k in range(len(idx) - 1, -1, -1):
            acc += durs[idx[k]]
            if acc >= seconds:
                last = k
                break
        start = rng.randint(0, last)
        run, total = [], 0.0
        for i in idx[start:]:
            run.append(i)
            total += durs[i]
            if total >= seconds:
                break
        out.append(run)
    return out


def load_test_excerpts(n: int, seconds: float, seed: int = 0,
                       recordings: list[str] | None = None,
                       min_english: float = 0.0) -> list[Clip]:
    """Excerpts of the cached MUCS test split, with audio and reference text.

    `min_english` keeps only excerpts whose reference has at least that % of English
    words, for showing code-switching. Off by default: a random draw is the honest
    sample, and many MUCS transcripts write English terms in Devanagari.
    """
    os.environ.setdefault("HF_HOME", str(REPO / "csasr-work" / "hf"))
    from ..data.loaders import load_hub_dataset

    ds = load_hub_dataset(REAL_HF, "test", token=os.environ.get("HF_TOKEN"))
    utt_ids = ds["utt_id"]                 # column access does not decode audio
    durs = ds["dur"] if "dur" in ds.column_names else [6.0] * len(utt_ids)
    runs = pick_excerpts(utt_ids, durs, n if min_english <= 0 else 8 * n, seconds, seed,
                         recordings)
    if min_english > 0:
        from .render import analyse

        texts = ds["text"]
        runs = [r for r in runs
                if analyse(" ".join(texts[i] for i in r)).english_pct >= min_english][:n]
        if len(runs) < n:
            print(f"[demo] only {len(runs)} excerpt(s) have >= {min_english:g}% English words; "
                  "try another --seed or a lower --min-english")
    clips = []
    for run in runs:
        rows = [ds[i] for i in run]
        audio = np.concatenate([np.asarray(r["audio"]["array"], dtype=np.float32) for r in rows])
        first, last = segment_index(rows[0]["utt_id"]), segment_index(rows[-1]["utt_id"])
        reco = recording_id(rows[0]["utt_id"])
        clips.append(Clip(
            clip_id=f"{reco}_{first:04d}-{last:04d}", audio=audio,
            ref=" ".join(r["text"].strip() for r in rows),
            source=f"MUCS test, recording {reco}, segments {first}–{last}"))
    return clips
