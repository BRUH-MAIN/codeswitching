"""Stage 4: decode the test set and write a hypotheses JSONL.

`--language None` (the default) reproduces WhisperX's "None" language-detection
option used in the paper. This is the whole point of the evaluation: a model that
has not learned code-switching mis-detects the language and then either
transliterates Hindi into Latin or deletes it outright. Forcing `<|hi|>` at
decode time would hide exactly the failure being measured.

Scoring is deliberately NOT done here -- `score.py` runs on CPU from this file.
That keeps GPU sessions short and makes metric changes free to re-run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from ..manifest import write_jsonl

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="hub id or local checkpoint dir")
    ap.add_argument("--out", type=Path, required=True, help="hypotheses JSONL")
    ap.add_argument("--test-hf", default=None)
    ap.add_argument("--test-config", default="test")
    ap.add_argument("--test-manifest", type=Path, default=None)
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--language",
        default="none",
        help="'none' => Whisper auto-detects (the paper's setting). Or 'hi'/'en'.",
    )
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=225)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from ..data.loaders import load_hub_dataset, load_manifest_dataset

    if args.test_hf:
        ds = load_hub_dataset(args.test_hf, args.test_config, token=args.hf_token)
    elif args.test_manifest:
        ds = load_manifest_dataset(args.test_manifest)
    else:
        raise SystemExit("need --test-hf or --test-manifest")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()
    model.generation_config.forced_decoder_ids = None

    language = None if args.language.lower() in ("none", "auto", "") else args.language
    print(f"[decode] {args.model} on {len(ds):,} utts, language={language!r} "
          f"({'auto-detect' if language is None else 'forced'})")

    rows = []
    for start in tqdm(range(0, len(ds), args.batch_size), desc="decode", unit="batch"):
        batch = ds[start : start + args.batch_size]
        arrays = [a["array"] for a in batch["audio"]]
        feats = processor.feature_extractor(
            arrays, sampling_rate=16_000, return_tensors="pt"
        ).input_features.to(device, dtype)

        with torch.inference_mode():
            ids = model.generate(
                feats,
                language=language,
                task="transcribe",
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
            )
        hyps = processor.batch_decode(ids, skip_special_tokens=True)

        for utt_id, ref, hyp in zip(batch["utt_id"], batch["text"], hyps):
            rows.append({"utt_id": utt_id, "ref": ref, "hyp": hyp.strip()})

    n = write_jsonl(args.out, rows)
    print(f"[decode] wrote {n:,} hypotheses -> {args.out}")
    print(f"[decode] now score on CPU:\n"
          f"  python -m csasr.eval.score --refs <ref.jsonl> --hyps {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
