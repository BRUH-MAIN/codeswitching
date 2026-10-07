"""Resolve a Whisper model to something `faster-whisper` can load.

faster-whisper runs CTranslate2 models, not HF ones. OpenAI's checkpoints have
prebuilt conversions on the Hub; our fine-tuned ones do not, so they get
converted once and cached.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["PREBUILT_CT2", "resolve_ct2"]


def _compat_converter(model_id: str, **kwargs):
    """ctranslate2's TransformersConverter, patched for our pinned transformers.

    ctranslate2>=4.8's converter calls `model_class.from_pretrained(path,
    dtype=...)`. transformers>=4.49 (our floor, and what train_a100.sh's `train`
    extra installs into this same env) only pops `torch_dtype` in
    from_pretrained -- `dtype` falls through unrecognized into the model's
    own __init__ and crashes with `TypeError: ... unexpected keyword argument
    'dtype'`. Translate it here rather than pinning either package tighter
    than training actually needs.
    """
    from ctranslate2.converters import TransformersConverter

    class _Converter(TransformersConverter):
        def load_model(self, model_class, model_name_or_path, **kw):
            if "dtype" in kw:
                kw["torch_dtype"] = kw.pop("dtype")
            return super().load_model(model_class, model_name_or_path, **kw)

    return _Converter(model_id, **kwargs)

# Systran maintains the official CTranslate2 conversions used by faster-whisper.
PREBUILT_CT2: dict[str, str] = {
    "openai/whisper-tiny": "Systran/faster-whisper-tiny",
    "openai/whisper-base": "Systran/faster-whisper-base",
    "openai/whisper-small": "Systran/faster-whisper-small",
    "openai/whisper-medium": "Systran/faster-whisper-medium",
    "openai/whisper-large-v2": "Systran/faster-whisper-large-v2",
    "openai/whisper-large-v3": "Systran/faster-whisper-large-v3",
}


def _source_mtime(p: Path) -> float:
    """Newest mtime among a local checkpoint's files; 0 for a hub id (immutable)."""
    if not p.is_dir():
        return 0.0
    return max((f.stat().st_mtime for f in p.glob("*") if f.is_file()), default=0.0)


def resolve_ct2(
    model_id: str,
    *,
    cache_dir: str | Path = "ct2_models",
    quantization: str = "float16",
    token: str | None = None,
) -> str:
    """Return a model name/path loadable by `faster_whisper.WhisperModel`.

    Prebuilt OpenAI conversions pass straight through. Anything else -- our
    fine-tuned whisper-small checkpoints -- is converted with CTranslate2 and
    cached on disk, so a second decode is free.
    """
    if model_id in PREBUILT_CT2:
        return PREBUILT_CT2[model_id]

    # Already a local CTranslate2 directory?
    p = Path(model_id)
    if p.is_dir() and (p / "model.bin").exists():
        return str(p)

    out = Path(cache_dir) / model_id.replace("/", "__")
    cached = out / "model.bin"
    if cached.exists():
        # Keyed on the checkpoint PATH, which is stable across runs -- so a
        # retrain writing new weights to the same --out silently kept scoring the
        # previous run's conversion (job 11812: identical test numbers from
        # visibly different models). Reconverting when the source is newer costs
        # a few minutes; not reconverting costs a whole invalid eval.
        if _source_mtime(p) > cached.stat().st_mtime:
            print(f"[ct2] cached conversion is older than {p} -- reconverting")
        else:
            print(f"[ct2] reusing cached conversion: {out}")
            return str(out)

    # train_whisper.py's WhisperProcessor.from_pretrained(..., use_fast=True)
    # silently returns a SLOW tokenizer on this transformers version (confirmed:
    # WhisperTokenizerFast loaded directly works fine, so this is a
    # WhisperProcessor auto-dispatch quirk, not a missing/broken `tokenizers`
    # install) -- so a locally fine-tuned checkpoint saves vocab.json/merges.txt
    # but no tokenizer.json, which faster-whisper needs at inference time.
    # Backfill it here instead of requiring a full retrain: WhisperTokenizerFast
    # auto-converts from the slow-tokenizer files just fine, same vocab either way.
    if p.is_dir() and not (p / "tokenizer.json").exists() \
            and not (p / "adapter_config.json").exists():
        from transformers import WhisperTokenizerFast

        print(f"[ct2] {p} has no tokenizer.json (slow-tokenizer save) -- generating one")
        WhisperTokenizerFast.from_pretrained(str(p)).save_pretrained(str(p))

    def _convert(src: str) -> None:
        print(f"[ct2] converting {src} -> {out} ({quantization})")
        out.parent.mkdir(parents=True, exist_ok=True)
        _compat_converter(
            src,
            copy_files=["tokenizer.json", "preprocessor_config.json"],
            load_as_float16=quantization.startswith("float16"),
        ).convert(str(out), quantization=quantization, force=True)

    # A LoRA/QLoRA checkpoint holds only adapter weights. Merge it into its base
    # in a scratch directory, convert that, and drop the scratch copy -- the
    # CTranslate2 model is the only artifact decoding needs.
    from ..train.export import is_adapter_dir

    if p.is_dir() and is_adapter_dir(p):
        import shutil
        import tempfile

        from ..train.export import merge_adapter

        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix="merged-", dir=out.parent))
        try:
            merge_adapter(p, tmp)
            if not (tmp / "tokenizer.json").exists():
                from transformers import WhisperTokenizerFast

                WhisperTokenizerFast.from_pretrained(str(tmp)).save_pretrained(str(tmp))
            _convert(str(tmp))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return str(out)

    _convert(model_id)
    return str(out)
