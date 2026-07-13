"""Stage 2b: synthesize code-mixed audio with `ai4bharat/indic-parler-tts`.

Runs under `transformers==4.46.1` (the version parler-tts pins).

Design notes:

*   **Sharded and resumable.** `--shard i --num-shards N` slices the sentence
    list. A completed clip is never regenerated. A Kaggle session that dies at
    the 12h cap costs at most the in-flight batch.
*   **Length-sorted batching.** Parler-TTS is autoregressive over audio codec
    frames; batching sentences of similar length keeps padding (and therefore
    wasted decode steps) down.
*   **16-bit PCM out, never float32.** float32 at 22.05 kHz would be ~7 GB for
    22 h. We resample to 16 kHz (Whisper's rate) and write int16: ~2.5 GB.
*   **Trailing-silence trim.** Batched generation pads the shorter clips with
    codec silence. Untrimmed, this inflates every duration and therefore the
    corpus hour count.
"""

from __future__ import annotations

import os

# MUST precede any `import transformers`. `parler_tts` imports
# `transformers.PreTrainedModel`, which pulls in `modeling_utils` ->
# `loss_deformable_detr` -> `image_transforms`, and THAT does
#
#     if is_tf_available():
#         import tensorflow as tf
#
# On Kaggle TensorFlow is installed but wants a newer `protobuf` than the one our
# pins leave behind, so the import dies with
#
#     ImportError: cannot import name 'runtime_version' from 'google.protobuf'
#
# We never use TensorFlow. `USE_TF=0` makes `is_tf_available()` return False and
# the import never happens.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..manifest import Utt, read_jsonl, write_jsonl
from .speakers import description_for

__all__ = ["trim_silence", "to_int16_16k", "main"]

TARGET_SR = 16_000
MAX_DUR = 30.0  # Whisper's receptive window
MIN_DUR = 0.5  # anything shorter is a TTS collapse


def trim_silence(x: np.ndarray, *, thresh: float = 1e-3, pad_ms: int = 30, sr: int = 22_050) -> np.ndarray:
    """Strip leading/trailing near-silence, keeping a short pad."""
    if x.size == 0:
        return x
    loud = np.flatnonzero(np.abs(x) > thresh)
    if loud.size == 0:
        return x[:0]
    pad = int(sr * pad_ms / 1000)
    lo = max(0, loud[0] - pad)
    hi = min(x.size, loud[-1] + pad + 1)
    return x[lo:hi]


def to_int16_16k(x: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16 kHz and quantize to int16 with clipping."""
    import soxr

    if sr != TARGET_SR:
        x = soxr.resample(x, sr, TARGET_SR)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.0:  # parler occasionally overshoots
        x = x / peak
    return np.clip(x * 32767.0, -32768, 32767).astype(np.int16)


def _load_model(model_id: str, device: str, dtype: str):
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    model = ParlerTTSForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=getattr(torch, dtype)
    ).to(device)
    model.eval()
    prompt_tok = AutoTokenizer.from_pretrained(model_id)
    desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    return model, prompt_tok, desc_tok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sentences", type=Path, required=True)
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="shard manifest JSONL")
    ap.add_argument("--model", default="ai4bharat/indic-parler-tts")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test cap")
    args = ap.parse_args(argv)

    import soundfile as sf
    import torch
    from transformers import set_seed

    rows = list(read_jsonl(args.sentences))
    rows = rows[args.shard :: args.num_shards]
    if args.limit:
        rows = rows[: args.limit]

    # Skip clips this shard already produced (resume after a session timeout).
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if not (args.audio_dir / f"{r['sent_id']}.wav").exists()]
    done = len(rows) - len(todo)
    print(f"[tts] shard {args.shard}/{args.num_shards}: {len(rows):,} sentences, "
          f"{done:,} already synthesized, {len(todo):,} to go")

    # Sort by text length to minimize padding within a batch.
    todo.sort(key=lambda r: len(r["text"]))

    model, prompt_tok, desc_tok = _load_model(args.model, args.device, args.dtype)
    sr = int(model.config.sampling_rate)
    print(f"[tts] model sampling rate {sr} Hz -> resampling to {TARGET_SR} Hz")

    set_seed(args.seed)
    dropped = 0

    for start in tqdm(range(0, len(todo), args.batch_size), desc="tts", unit="batch"):
        batch = todo[start : start + args.batch_size]
        descs = [description_for(r["sent_id"])[1] for r in batch]
        prompts = [r["text"] for r in batch]

        d = desc_tok(descs, return_tensors="pt", padding=True).to(args.device)
        p = prompt_tok(prompts, return_tensors="pt", padding=True).to(args.device)

        with torch.inference_mode():
            gen = model.generate(
                input_ids=d.input_ids,
                attention_mask=d.attention_mask,
                prompt_input_ids=p.input_ids,
                prompt_attention_mask=p.attention_mask,
            )

        audio = gen.to(torch.float32).cpu().numpy()
        if audio.ndim == 1:
            audio = audio[None, :]

        for r, wav in zip(batch, audio):
            wav = trim_silence(np.asarray(wav).squeeze(), sr=sr)
            pcm = to_int16_16k(wav, sr)
            dur = pcm.size / TARGET_SR
            if not (MIN_DUR <= dur <= MAX_DUR):
                dropped += 1
                continue
            sf.write(str(args.audio_dir / f"{r['sent_id']}.wav"), pcm, TARGET_SR, subtype="PCM_16")

    # Rebuild the shard manifest from what is actually on disk, so a resumed run
    # emits a manifest covering both old and new clips.
    out_rows = []
    for r in rows:
        wav = args.audio_dir / f"{r['sent_id']}.wav"
        if not wav.exists():
            continue
        sp, _ = description_for(r["sent_id"])
        out_rows.append(
            Utt(
                utt_id=r["sent_id"],
                text=r["text"],
                dur=round(sf.info(str(wav)).duration, 3),
                wav=wav.as_posix(),
                speaker=sp.name,
                sent_id=r["sent_id"],
                lang="cs",
                extra={"bigram": r.get("bigram"), "matrix_lang": r.get("matrix_lang")},
            )
        )

    n = write_jsonl(args.out, out_rows)
    hours = sum(r.dur or 0 for r in out_rows) / 3600
    print(f"[tts] shard {args.shard}: wrote {n:,} clips ({hours:.2f} h), dropped {dropped:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
