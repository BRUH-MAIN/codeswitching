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


def _copy_preprocessor_config(model_id: str, out: Path, *, token: str | None) -> None:
    """faster-whisper reads exactly `preprocessor_config.json` from the
    converted directory to size its feature extractor (n_mels etc.); absent,
    it silently falls back to defaults, which happen to be correct for
    whisper-{tiny,base,small,medium} (80 mels) but would be silently WRONG for
    large-v3 (128 mels).

    `transformers` has renamed this file across versions -- older releases
    wrote `preprocessor_config.json`, current ones write `processor_config.json`
    -- so a fine-tuned checkpoint may only have one or the other depending on
    which version trained it. Try both under their real names; write whichever
    is found under the name faster-whisper actually looks for.
    """
    p = Path(model_id)
    for name in ("preprocessor_config.json", "processor_config.json"):
        if p.is_dir():
            src = p / name
            if not src.exists():
                continue
            (out / "preprocessor_config.json").write_bytes(src.read_bytes())
            return
        else:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

            try:
                downloaded = hf_hub_download(model_id, name, token=token)
            except (EntryNotFoundError, RepositoryNotFoundError):
                continue
            (out / "preprocessor_config.json").write_bytes(Path(downloaded).read_bytes())
            return
    print(f"[ct2] {model_id}: no preprocessor/processor config found; "
          f"faster-whisper will fall back to feature-extractor defaults")


def resolve_ct2(
    model_id: str,
    *,
    cache_dir: str | Path = "ct2_models",
    quantization: str = "float16",
    token: str | None = None,
) -> str:
    """Return a model name/path loadable by `faster_whisper.WhisperModel`.

    Prebuilt OpenAI conversions pass straight through. Anything else -- our
    fine-tuned checkpoints -- is converted with CTranslate2 and cached on
    disk, so a second decode is free.
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
    # preprocessor_config.json is handled separately below (name varies by
    # transformers version); asking TransformersConverter for a hardcoded name
    # here is exactly what broke on checkpoints saved by newer transformers.
    TransformersConverter(
        model_id,
        copy_files=["tokenizer.json"],
        load_as_float16=quantization.startswith("float16"),
    ).convert(str(out), quantization=quantization, force=True)
    _copy_preprocessor_config(model_id, out, token=token)
    return str(out)
