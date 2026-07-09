"""Turn the MUCS 2021 Kaldi-style corpus into JSONL manifests (+ optional audio cuts).

MUCS ships long recordings plus a `segments` file giving per-sentence
timestamps. Layout varies between the train and test tarballs, so we *discover*
the Kaldi files rather than hardcoding paths.

Kaldi formats:
    text      <utt_id> <transcript...>
    segments  <utt_id> <reco_id> <start_sec> <end_sec>
    wav.scp   <reco_id> <path>            (paths are the packagers', not ours)

`--text-only` skips audio entirely, which is all Gate 0 needs. Track 2 never
trains on real code-switched audio, so the 89.8h train split is only ever
prepared text-only, plus a 4h dev slice that does get cut.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from tqdm import tqdm

from ..manifest import Utt, write_jsonl

__all__ = ["scan", "build_manifest", "main"]


def _read_kv(path: Path, *, maxsplit: int = 1) -> dict[str, str]:
    """Parse a Kaldi two-column file into {key: rest}."""
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(maxsplit=maxsplit)
            if len(parts) < 2:
                continue
            out[parts[0]] = parts[1]
    return out


def _find_one(root: Path, name: str) -> Path | None:
    """Locate a Kaldi file by name, preferring shallower paths."""
    hits = sorted(root.rglob(name), key=lambda p: len(p.parts))
    return hits[0] if hits else None


def scan(root: Path) -> dict:
    """Discover and parse the Kaldi files under `root`."""
    text_p = _find_one(root, "text")
    if text_p is None:
        raise SystemExit(f"no Kaldi `text` file found under {root}")

    seg_p = _find_one(root, "segments")
    scp_p = _find_one(root, "wav.scp")

    text = _read_kv(text_p)

    segments: dict[str, tuple[str, float, float]] = {}
    if seg_p is not None:
        with seg_p.open("r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 4:
                    utt, reco, start, end = parts[0], parts[1], float(parts[2]), float(parts[3])
                    segments[utt] = (reco, start, end)

    # wav.scp paths come from the packagers' machines. Remap by basename onto
    # whatever .wav files actually exist in this checkout.
    on_disk = {p.name: p for p in root.rglob("*.wav")}
    reco2path: dict[str, Path] = {}
    if scp_p is not None:
        for reco, raw in _read_kv(scp_p).items():
            cand = on_disk.get(Path(raw.strip()).name)
            if cand is not None:
                reco2path[reco] = cand
    for reco in {r for r, _, _ in segments.values()}:
        if reco not in reco2path and f"{reco}.wav" in on_disk:
            reco2path[reco] = on_disk[f"{reco}.wav"]

    return {
        "root": root,
        "text_path": text_p,
        "segments_path": seg_p,
        "wav_scp_path": scp_p,
        "text": text,
        "segments": segments,
        "reco2path": reco2path,
        "n_wav_on_disk": len(on_disk),
    }


def _cut(src: Path, start: float, end: float, dst: Path) -> float:
    """Write [start, end) of `src` to `dst` as 16 kHz mono 16-bit PCM."""
    import soundfile as sf

    info = sf.info(str(src))
    sr = info.samplerate
    s = max(0, int(round(start * sr)))
    e = min(info.frames, int(round(end * sr)))
    if e <= s:
        return 0.0

    data, _ = sf.read(str(src), start=s, stop=e, dtype="int16", always_2d=True)
    if data.shape[1] > 1:
        # Average in float; `mean(dtype="int16")` accumulates in int16 and overflows.
        data = data.mean(axis=1, keepdims=True).round().astype("int16")

    if sr != 16000:
        import numpy as np
        import soxr

        f = soxr.resample(data.astype("float32") / 32768.0, sr, 16000)
        data = np.clip(f * 32768.0, -32768, 32767).astype("int16").reshape(-1, 1)

    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), data, 16000, subtype="PCM_16")
    return data.shape[0] / 16000.0


def build_manifest(
    root: Path,
    out_jsonl: Path,
    *,
    text_only: bool = True,
    audio_dir: Path | None = None,
    keep_ids: set[str] | None = None,
) -> list[dict]:
    info = scan(root)
    text, segments, reco2path = info["text"], info["segments"], info["reco2path"]

    print(
        f"[prepare_mucs] {root.name}: {len(text):,} transcripts, "
        f"{len(segments):,} segments, {info['n_wav_on_disk']:,} wav files on disk"
    )
    if segments and not reco2path and not text_only:
        raise SystemExit("segments present but no recordings resolved; cannot cut audio")

    utt_ids = sorted(text)
    if keep_ids is not None:
        utt_ids = [u for u in utt_ids if u in keep_ids]

    rows: list[Utt] = []
    it = tqdm(utt_ids, desc=f"{root.name}:{'text' if text_only else 'cut'}", unit="utt")
    for utt in it:
        dur = None
        wav = None
        if utt in segments:
            reco, start, end = segments[utt]
            dur = round(end - start, 3)
            if not text_only:
                src = reco2path.get(reco)
                if src is None:
                    continue
                dst = (audio_dir or out_jsonl.parent / "audio") / f"{utt}.wav"
                if dst.exists():
                    import soundfile as sf

                    dur = sf.info(str(dst)).duration
                else:
                    dur = _cut(src, start, end, dst)
                if dur <= 0:
                    continue
                wav = str(dst.as_posix())
        rows.append(Utt(utt_id=utt, text=text[utt], dur=dur, wav=wav))

    n = write_jsonl(out_jsonl, rows)
    hours = sum(r.dur or 0 for r in rows) / 3600
    print(f"[prepare_mucs] wrote {n:,} rows -> {out_jsonl}  ({hours:.2f} h)")
    return [r.to_dict() for r in rows]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="extracted split dir")
    ap.add_argument("--out", type=Path, required=True, help="output manifest JSONL")
    ap.add_argument("--cut-audio", action="store_true", help="materialize 16kHz wav cuts")
    ap.add_argument("--audio-dir", type=Path, default=None)
    ap.add_argument(
        "--keep-ids",
        type=Path,
        default=None,
        help="restrict to utt_ids present in this JSONL (e.g. cut audio for the dev slice only)",
    )
    ap.add_argument(
        "--dev-hours",
        type=float,
        default=0.0,
        help="if >0, also carve a seeded dev slice of this many hours out of --out",
    )
    ap.add_argument("--dev-out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)

    keep_ids = None
    if args.keep_ids:
        # Read the id list up front: --keep-ids may point at --out (rebuilding the
        # dev manifest in place, this time with audio).
        from ..manifest import read_jsonl

        keep_ids = {r["utt_id"] for r in read_jsonl(args.keep_ids)}
        print(f"[prepare_mucs] restricting to {len(keep_ids):,} ids from {args.keep_ids}")

    rows = build_manifest(
        args.root,
        args.out,
        text_only=not args.cut_audio,
        audio_dir=args.audio_dir,
        keep_ids=keep_ids,
    )

    if args.dev_hours > 0:
        if args.dev_out is None:
            raise SystemExit("--dev-hours requires --dev-out")
        rng = random.Random(args.seed)
        shuffled = rows[:]
        rng.shuffle(shuffled)
        picked, acc = [], 0.0
        for r in shuffled:
            if acc >= args.dev_hours * 3600:
                break
            picked.append(r)
            acc += r.get("dur") or 0.0
        write_jsonl(args.dev_out, picked)
        print(f"[prepare_mucs] dev slice: {len(picked):,} utts, {acc/3600:.2f} h -> {args.dev_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
