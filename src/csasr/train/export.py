"""Turn a LoRA / QLoRA adapter checkpoint into a plain Whisper checkpoint.

faster-whisper runs CTranslate2 models, and CTranslate2 converts ordinary HF
checkpoints -- it knows nothing about PEFT. So an adapter is merged into its
base first. For QLoRA the adapter is merged into the FULL-PRECISION base, not
the 4-bit one it was trained against: that is the standard practice, and it is
what makes the LoRA and QLoRA rows decode through an identical path.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["is_adapter_dir", "adapter_base", "merge_adapter"]


def is_adapter_dir(path: str | Path) -> bool:
    return (Path(path) / "adapter_config.json").exists()


def adapter_base(path: str | Path) -> str:
    cfg = json.loads((Path(path) / "adapter_config.json").read_text(encoding="utf-8"))
    return cfg["base_model_name_or_path"]


def merge_adapter(
    adapter_dir: str | Path,
    out_dir: str | Path,
    *,
    base: str | None = None,
    verify: bool = True,
    save_processor: bool = True,
) -> Path:
    """Merge `adapter_dir` into its base model and save a standalone checkpoint."""
    import torch
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    adapter_dir, out_dir = Path(adapter_dir), Path(out_dir)
    base = base or adapter_base(adapter_dir)

    model = WhisperForConditionalGeneration.from_pretrained(
        base, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    peft_model = PeftModel.from_pretrained(model, str(adapter_dir))
    peft_model.eval()

    if verify:
        # A merge that silently changes the function (a mismatched base, a
        # scaling bug) would score as a "bad adapter" rather than as a bug.
        cfg = model.config
        gen = torch.Generator().manual_seed(0)
        feats = torch.randn(1, cfg.num_mel_bins, 2 * cfg.max_source_positions, generator=gen)
        dec = torch.tensor([[cfg.decoder_start_token_id]])
        with torch.inference_mode():
            before = peft_model(input_features=feats, decoder_input_ids=dec).logits

    merged = peft_model.merge_and_unload()

    if verify:
        with torch.inference_mode():
            after = merged(input_features=feats, decoder_input_ids=dec).logits
        diff = (before - after).abs().max().item()
        scale = before.abs().max().item()
        if diff > 1e-3 * max(scale, 1.0):
            raise SystemExit(f"[export] merged logits differ from adapter logits by "
                             f"{diff:.4g} (scale {scale:.4g}) -- refusing to export")
        print(f"[export] merge verified: max |logit diff| {diff:.2e}")

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir))
    if save_processor:
        # Checkpoints written mid-training carry the processor; else use the base's.
        src = adapter_dir if (adapter_dir / "preprocessor_config.json").exists() else base
        WhisperProcessor.from_pretrained(str(src)).save_pretrained(str(out_dir))
    print(f"[export] {adapter_dir} + {base} -> {out_dir}")
    return out_dir
