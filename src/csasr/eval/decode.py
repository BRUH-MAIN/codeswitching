"""Stage 4: decode the test set and write a hypotheses JSONL.

WHY RECORDING-LEVEL, NOT PER-UTTERANCE
--------------------------------------
The paper decodes with WhisperX, which transcribes a whole recording: it detects
the language once, feeds 30-second chunks with real surrounding context, and
applies temperature fallback plus a compression-ratio check that aborts
repetition loops.

Decoding the 3,136 isolated 6-second clips instead is a different and much
harder task, and it silently destroys what CBA measures. Without context Whisper
renders English loanwords in Devanagari, so the hypothesis has no script
boundary and *no switch bigram can match*. Measured on real test audio with an
identical model:

    reference          21.6% Latin
    per-segment         0.0% Latin   MER 182.6   CBA-HE 0.0
    recording-level    12.2% Latin   MER 103.1   CBA-HE 1.6

Per-utterance decoding also flipped language detection to Urdu on some clips and
fell into `तो तो तो ...` repetition loops. Hence `--mode recording` is the
default. `--mode segment` is kept for the training-time dev metric, where only
relative ranking of checkpoints matters.

`--language none` reproduces WhisperX's "None" option: detect once, per
recording. That is the point of the evaluation -- a model that has not learned
code-switching mis-detects and then transliterates or deletes.

Scoring happens on CPU in `score.py`, never here.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..manifest import write_jsonl
from .grouping import concat_refs, group_by_recording, recording_id

__all__ = ["main"]

SR = 16_000

# OpenAI's decoding heuristics, as faster-whisper / WhisperX apply them.
TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
COMPRESSION_RATIO_THRESHOLD = 2.4  # abort repetition loops
LOGPROB_THRESHOLD = -1.0
NO_SPEECH_THRESHOLD = 0.6


def _load_rows(args) -> list[dict]:
    from ..data.loaders import load_hub_dataset, load_manifest_dataset

    token = args.hf_token or os.environ.get("HF_TOKEN")
    if args.test_hf:
        ds = load_hub_dataset(args.test_hf, args.test_config, token=token)
    elif args.test_manifest:
        ds = load_manifest_dataset(args.test_manifest)
    else:
        raise SystemExit("need --test-hf or --test-manifest")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    return ds


def _concat_audio(group: list[dict]) -> np.ndarray:
    return np.concatenate([np.asarray(r["audio"]["array"], dtype=np.float32) for r in group])


# --------------------------------------------------------------------------
# faster-whisper: the engine WhisperX wraps
# --------------------------------------------------------------------------
def _decode_faster_whisper(rows, args) -> list[dict]:
    import torch
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    from .ct2 import resolve_ct2

    model_path = resolve_ct2(
        args.model, cache_dir=args.ct2_cache, quantization=args.compute_type,
        token=args.hf_token or os.environ.get("HF_TOKEN"),
    )

    n_gpu = torch.cuda.device_count()
    device = "cuda" if n_gpu else "cpu"
    compute = args.compute_type if n_gpu else "int8"

    # CTranslate2 replicates the model across every listed device and runs
    # batches on them in parallel. Kaggle hands out 2x T4; using one is a 2x
    # waste.
    if args.device_index:
        device_index = args.device_index
    else:
        device_index = list(range(n_gpu)) if n_gpu else [0]

    model = WhisperModel(
        model_path, device=device, device_index=device_index,
        compute_type=compute, num_workers=max(1, len(device_index)),
    )
    print(f"[decode] faster-whisper {model_path} on {device}{device_index} ({compute})")

    # BatchedInferencePipeline is what makes WhisperX fast: VAD carves the audio
    # into speech chunks, and those chunks are decoded as a BATCH instead of one
    # at a time. Sequential decoding pins one GPU at ~4 GB of 15 GB and leaves
    # the second idle; batching fills both.
    engine = model
    batched = args.batch_size > 1 and args.vad
    if batched:
        engine = BatchedInferencePipeline(model=model)
        print(f"[decode] batched pipeline, batch_size={args.batch_size}")
    else:
        print("[decode] sequential (batch_size=1 or --no-vad); this is SLOW")

    language = None if args.language.lower() in ("none", "auto", "") else args.language

    common = dict(
        language=language,
        task="transcribe",
        beam_size=args.num_beams,
        temperature=list(TEMPERATURES),
        compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
        log_prob_threshold=LOGPROB_THRESHOLD,
        no_speech_threshold=NO_SPEECH_THRESHOLD,
        vad_filter=args.vad,
        # Whisper confuses Hindi and Urdu constantly -- they are the same spoken
        # language in different scripts. faster-whisper's default detects the
        # language from ONE 30s window, and a single bad window sends the whole
        # recording into Perso-Arabic script, where no Hindi/English switch
        # bigram can possibly match and CBA collapses. Vote over several windows.
        language_detection_segments=args.lang_detect_segments,
        language_detection_threshold=args.lang_detect_threshold,
    )
    if batched:
        common["batch_size"] = args.batch_size
    else:
        common["condition_on_previous_text"] = False

    groups = group_by_recording(rows)
    how = (f"auto-detect per recording, voting over {args.lang_detect_segments} windows"
           if language is None else "forced")
    print(f"[decode] {len(groups)} recording(s), language={language!r} ({how})")

    out = []
    for reco, group in tqdm(sorted(groups.items()), desc="decode", unit="rec"):
        audio = _concat_audio(group)
        segments, info = engine.transcribe(audio, **common)
        text = " ".join(s.text.strip() for s in segments).strip()
        out.append({
            "utt_id": reco,
            "hyp": text,
            "detected_language": info.language,
            "language_probability": round(float(info.language_probability or 0.0), 3),
            "n_segments": len(group),
            "dur": float(len(audio) / SR),
        })
    return out


# --------------------------------------------------------------------------
# transformers: fallback engine, long-form or per-segment
# --------------------------------------------------------------------------
def _decode_transformers(rows, args) -> list[dict]:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()
    model.generation_config.forced_decoder_ids = None
    # A fine-tuned checkpoint bakes language="hi" into its generation config;
    # clear it so `--language none` really does auto-detect.
    if args.language.lower() in ("none", "auto", ""):
        model.generation_config.language = None
    language = None if args.language.lower() in ("none", "auto", "") else args.language

    gen_kw = dict(
        language=language, task="transcribe", num_beams=args.num_beams,
        temperature=TEMPERATURES, compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
        logprob_threshold=LOGPROB_THRESHOLD, no_speech_threshold=NO_SPEECH_THRESHOLD,
        condition_on_prev_tokens=False,
    )

    if args.mode == "segment":
        print(f"[decode] transformers per-segment on {len(rows)} utts (dev metric only)")
        out = []
        for start in tqdm(range(0, len(rows), args.batch_size), desc="decode", unit="batch"):
            batch = rows[start : start + args.batch_size]
            arrays = [r["audio"]["array"] for r in batch]
            feats = processor.feature_extractor(
                arrays, sampling_rate=SR, return_tensors="pt"
            ).input_features.to(device, dtype)
            with torch.inference_mode():
                ids = model.generate(feats, max_new_tokens=args.max_new_tokens,
                                     language=language, task="transcribe", num_beams=args.num_beams)
            for r, h in zip(batch, processor.batch_decode(ids, skip_special_tokens=True)):
                out.append({"utt_id": r["utt_id"], "hyp": h.strip()})
        return out

    groups = group_by_recording(rows)
    print(f"[decode] transformers long-form, {len(groups)} recording(s), language={language!r}")
    out = []
    for reco, group in tqdm(sorted(groups.items()), desc="decode", unit="rec"):
        audio = _concat_audio(group)
        inputs = processor(audio, sampling_rate=SR, return_tensors="pt",
                           truncation=False, padding="longest", return_attention_mask=True)
        inputs = {k: (v.to(device, dtype) if v.dtype.is_floating_point else v.to(device))
                  for k, v in inputs.items()}
        with torch.inference_mode():
            ids = model.generate(**inputs, return_timestamps=True, **gen_kw)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        out.append({"utt_id": reco, "hyp": text, "n_segments": len(group),
                    "dur": float(len(audio) / SR)})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="hub id or local checkpoint dir")
    ap.add_argument("--out", type=Path, required=True, help="hypotheses JSONL")
    ap.add_argument("--engine", default="faster-whisper", choices=["faster-whisper", "transformers"])
    ap.add_argument("--mode", default="recording", choices=["recording", "segment"])
    ap.add_argument("--test-hf", default=None)
    ap.add_argument("--test-config", default="test")
    ap.add_argument("--test-manifest", type=Path, default=None)
    ap.add_argument("--refs-out", type=Path, default=None,
                    help="also write recording-level references here")
    # Read the token from the environment. Passing it in argv leaks it into
    # every traceback and into `ps` output.
    ap.add_argument("--hf-token", default=None, help="prefer the HF_TOKEN env var")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="faster-whisper: VAD chunks decoded per batch (1 = sequential)")
    ap.add_argument("--device-index", type=int, nargs="*", default=None,
                    help="GPU ids to replicate onto; default = all visible")
    ap.add_argument("--language", default="none",
                    help="'none' => detect per recording (the paper's setting)")
    ap.add_argument("--lang-detect-segments", type=int, default=8,
                    help="windows to vote over when detecting language "
                         "(faster-whisper default is 1, which mis-detects hi as ur)")
    ap.add_argument("--lang-detect-threshold", type=float, default=0.5)
    ap.add_argument("--num-beams", type=int, default=5, help="WhisperX default")
    ap.add_argument("--vad", action="store_true", default=True, help="faster-whisper VAD filter")
    ap.add_argument("--no-vad", dest="vad", action="store_false")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--ct2-cache", default="ct2_models")
    ap.add_argument("--max-new-tokens", type=int, default=440)
    ap.add_argument("--limit", type=int, default=0)
    # Shard whole recordings across processes. Kaggle gives 2x T4, so
    #   CUDA_VISIBLE_DEVICES=0 ... --shard 0 --num-shards 2  &
    #   CUDA_VISIBLE_DEVICES=1 ... --shard 1 --num-shards 2  &
    # halves wall clock even when a single call cannot fill both GPUs.
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args(argv)

    ds = _load_rows(args)
    rows = [dict(r) for r in ds] if args.mode == "recording" or args.engine == "faster-whisper" else ds

    if args.num_shards > 1:
        if args.mode != "recording":
            raise SystemExit("--num-shards requires --mode recording")
        # Shard by recording, never by utterance: a split recording would be
        # decoded without its own context, which is the bug this mode exists to
        # avoid.
        keep = sorted(group_by_recording(rows))[args.shard :: args.num_shards]
        rows = [r for r in rows if recording_id(r["utt_id"]) in set(keep)]
        print(f"[decode] shard {args.shard}/{args.num_shards}: "
              f"{len(keep)} recording(s), {len(rows):,} segments")

    if args.engine == "faster-whisper":
        if args.mode == "segment":
            raise SystemExit("faster-whisper is only wired for --mode recording")
        hyps = _decode_faster_whisper(rows, args)
    else:
        hyps = _decode_transformers(rows, args)

    n = write_jsonl(args.out, hyps)
    print(f"[decode] wrote {n:,} hypotheses -> {args.out}")

    if args.mode == "recording":
        refs = concat_refs([{"utt_id": r["utt_id"], "text": r["text"]} for r in rows])
        refs_path = args.refs_out or args.out.with_name("refs_recording.jsonl")
        write_jsonl(refs_path, [{"utt_id": k, "text": v} for k, v in sorted(refs.items())])
        print(f"[decode] wrote {len(refs):,} recording-level references -> {refs_path}")
        print(f"\n  score on CPU:\n"
              f"    python -m csasr.eval.score --refs {refs_path} --hyps {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
