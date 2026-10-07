"""Decode with a CTC model (MMS), at recording level, into `decode.py`'s format.

The Whisper path concatenates each recording and lets faster-whisper chunk it.
The same is done here so every system in the ablation is scored on identical
units: a recording's segments are concatenated, cut into overlapping windows,
and the logits of each window are stitched after discarding the overlap -- the
frames nearest a window edge are the ones with the least context on one side.
Greedy CTC, no language model: the encoder is the thing under test.

Output rows and the per-utterance reference file match `decode.py` exactly, so
`score.py` and `select.py` need no CTC-specific branch.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from ..manifest import write_jsonl
from .decode import SR, _concat_audio, _load_rows, _progress
from .grouping import group_by_recording

__all__ = ["windows", "main"]

#: wav2vec2's convolutional stack emits one frame per 320 samples (20 ms).
FRAME = 320


def windows(n: int, chunk: int, stride: int) -> list[tuple[int, int, int, int]]:
    """Cut `n` samples into overlapping windows.

    Returns (start, end, drop_left, drop_right) in samples: decode [start, end)
    and discard `drop_*` from each side. Kept regions tile [0, n) exactly.
    """
    if n <= chunk:
        return [(0, n, 0, 0)]
    step = chunk - 2 * stride
    out = []
    start = 0
    while True:
        end = min(start + chunk, n)
        last = end == n
        out.append((start, end, 0 if start == 0 else stride, 0 if last else stride))
        if last:
            return out
        start += step


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="a train_ctc.py checkpoint dir, or a hub id for zero-shot")
    ap.add_argument("--target-lang", default=None,
                    help="zero-shot only: the MMS adapter to load, e.g. hin")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--refs-out", type=Path, default=None)
    ap.add_argument("--test-hf", default=None)
    ap.add_argument("--test-config", default="test")
    ap.add_argument("--test-manifest", type=Path, default=None)
    ap.add_argument("--hf-token", default=None, help="prefer the HF_TOKEN env var")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--chunk-s", type=float, default=30.0)
    ap.add_argument("--stride-s", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    import torch

    rows = [dict(r) for r in _load_rows(args)]

    if Path(args.model).is_dir():
        from ..train.train_ctc import load_ctc_model

        model, processor = load_ctc_model(args.model)
    else:
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        kw = {"target_lang": args.target_lang, "ignore_mismatched_sizes": True} \
            if args.target_lang else {}
        model = Wav2Vec2ForCTC.from_pretrained(args.model, **kw)
        processor = AutoProcessor.from_pretrained(
            args.model, **({"target_lang": args.target_lang} if args.target_lang else {}))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = model.to(device, dtype).eval()
    print(f"[decode-ctc] {args.model} on {device} ({dtype})")

    chunk = int(args.chunk_s * SR) // FRAME * FRAME
    stride = int(args.stride_s * SR) // FRAME * FRAME
    groups = group_by_recording(rows)
    print(f"[decode-ctc] {len(groups)} recording(s), {args.chunk_s:g}s windows, "
          f"{args.stride_s:g}s overlap each side")

    out = []
    for reco, group in _progress(sorted(groups.items()), desc="decode", unit="rec"):
        audio = _concat_audio(group)
        wins = windows(len(audio), chunk, stride)
        ids: list[np.ndarray] = []
        for b in range(0, len(wins), args.batch_size):
            batch = wins[b : b + args.batch_size]
            feats = processor.feature_extractor(
                [audio[s:e] for s, e, _, _ in batch], sampling_rate=SR,
                return_tensors="pt", padding=True, return_attention_mask=True,
            )
            with torch.inference_mode():
                logits = model(feats.input_values.to(device, dtype),
                               attention_mask=feats.attention_mask.to(device)).logits
            pred = logits.argmax(-1).cpu().numpy()
            for (s, e, dl, dr), p in zip(batch, pred):
                n_frames = int(model._get_feat_extract_output_lengths(torch.tensor(e - s)))
                p = p[:n_frames]  # drop the frames that are batch padding
                ids.append(p[dl // FRAME : len(p) - dr // FRAME])
        # One decode over the stitched frames, so a character repeated across a
        # window join collapses the way it would inside a window.
        text = processor.tokenizer.decode(np.concatenate(ids)) if ids else ""
        out.append({"utt_id": reco, "hyp": text.strip(), "n_segments": len(group),
                    "dur": float(len(audio) / SR)})

    n = write_jsonl(args.out, out)
    print(f"[decode-ctc] wrote {n:,} hypotheses -> {args.out}")
    refs_path = args.refs_out or args.out.with_name("refs_utt.jsonl")
    write_jsonl(refs_path, [{"utt_id": r["utt_id"], "text": r["text"]} for r in rows])
    print(f"[decode-ctc] wrote {len(rows):,} per-utterance references -> {refs_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
