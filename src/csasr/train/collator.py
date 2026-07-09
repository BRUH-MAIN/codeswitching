"""Batch collation for Whisper seq2seq fine-tuning.

Two details that silently corrupt training if missed:

1.  **Label padding must be -100**, not the tokenizer pad id, or the model is
    trained to emit pad tokens.

2.  **Strip the leading decoder_start token from labels.** The tokenizer emits
    `<|startoftranscript|><|hi|><|transcribe|><|notimestamps|> ...` but the model
    prepends `decoder_start_token_id` itself when building `decoder_input_ids`.
    Leaving it in shifts the sequence by one and the model learns to predict its
    own BOS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["WhisperCollator"]


@dataclass
class WhisperCollator:
    processor: Any
    decoder_start_token_id: int
    max_label_len: int = 448  # Whisper's decoder context

    def __call__(self, features: list[dict[str, Any]]) -> dict:
        import torch

        # Log-mel features are always 3000 frames, so this is a stack, not a pad.
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # The model re-adds decoder_start_token_id when shifting labels right.
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        if labels.shape[1] > self.max_label_len:
            labels = labels[:, : self.max_label_len]

        batch["labels"] = labels
        return batch
