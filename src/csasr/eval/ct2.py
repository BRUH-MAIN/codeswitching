"""Resolve a Whisper model to something `faster-whisper` can load.

faster-whisper runs CTranslate2 models, not HF ones. OpenAI's checkpoints have
prebuilt conversions on the Hub; our fine-tuned ones do not, so they get
converted once and cached.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["PREBUILT_CT2", "resolve_ct2"]

# Systran maintains the official CTranslate2 conversions used by faster-whisper.
PREBUILT_CT2: dict[str, str] = {
    "openai/whisper-tiny": "Systran/faster-whisper-tiny",
    "openai/whisper-base": "Systran/faster-whisper-base",
    "openai/whisper-small": "Systran/faster-whisper-small",
    "openai/whisper-medium": "Systran/faster-whisper-medium",
    "openai/whisper-large-v2": "Systran/faster-whisper-large-v2",
    "openai/whisper-large-v3": "Systran/faster-whisper-large-v3",
}


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
    if (out / "model.bin").exists():
        print(f"[ct2] reusing cached conversion: {out}")
        return str(out)

    from ctranslate2.converters import TransformersConverter

    print(f"[ct2] converting {model_id} -> {out} ({quantization})")
    out.parent.mkdir(parents=True, exist_ok=True)
    TransformersConverter(
        model_id,
        copy_files=["tokenizer.json", "preprocessor_config.json"],
        load_as_float16=quantization.startswith("float16"),
    ).convert(str(out), quantization=quantization, force=True)
    return str(out)
