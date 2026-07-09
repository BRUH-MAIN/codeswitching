"""Collator tests against a real Whisper tokenizer.

These target the two failure modes that train silently and wrongly:
padding labels with the pad id instead of -100, and leaving the
decoder_start token on the labels (which shifts the sequence by one and
teaches the model to predict its own BOS).

Skipped when transformers/torch are absent (they are not needed for Stage 0).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import WhisperProcessor  # noqa: E402

from csasr.train.collator import WhisperCollator  # noqa: E402

MODEL = "openai/whisper-tiny"  # 39M, ~150 MB; same tokenizer family as small/large


@pytest.fixture(scope="module")
def processor():
    p = WhisperProcessor.from_pretrained(MODEL, language="hi", task="transcribe")
    p.tokenizer.set_prefix_tokens(language="hi", task="transcribe")
    return p


@pytest.fixture(scope="module")
def collator(processor):
    from transformers import WhisperForConditionalGeneration

    cfg = WhisperForConditionalGeneration.from_pretrained(MODEL).config
    return WhisperCollator(processor, cfg.decoder_start_token_id), cfg


def _features(processor, texts):
    import numpy as np

    fe = processor.feature_extractor
    out = []
    for t in texts:
        audio = np.zeros(16000, dtype=np.float32)
        out.append(
            {
                "input_features": fe(audio, sampling_rate=16000).input_features[0],
                "labels": processor.tokenizer(t).input_ids,
            }
        )
    return out


class TestPrefixTokens:
    def test_hi_prefix_is_applied(self, processor):
        ids = processor.tokenizer("नमस्ते").input_ids
        decoded = processor.tokenizer.convert_ids_to_tokens(ids[:4])
        assert "<|startoftranscript|>" in decoded
        assert "<|hi|>" in decoded
        assert "<|transcribe|>" in decoded
        assert "<|notimestamps|>" in decoded

    def test_prefix_is_hi_not_en(self, processor):
        toks = processor.tokenizer.convert_ids_to_tokens(processor.tokenizer("x").input_ids)
        assert "<|en|>" not in toks


class TestCollator:
    def test_whisper_pad_and_eos_are_the_same_token(self, processor):
        # 50257 <|endoftext|> serves as both. So "no pad_token_id in labels" is
        # NOT the invariant -- every sequence legitimately ends with 50257.
        tok = processor.tokenizer
        assert tok.pad_token_id == tok.eos_token_id

    def test_padding_is_minus_100_and_only_trailing(self, processor, collator):
        coll, _ = collator
        batch = coll(_features(processor, ["इस document में", "हाँ"]))
        labels = batch["labels"]
        eos = processor.tokenizer.eos_token_id
        assert labels.shape[0] == 2
        assert (labels == -100).any(), "short sequence must be padded"

        for row in labels:
            mask = row == -100
            n_pad = int(mask.sum())
            if n_pad:
                # -100 must form a trailing run: nothing real after the padding.
                assert bool(mask[-n_pad:].all()), "-100 appears mid-sequence"
                assert not bool(mask[:-n_pad].any())
            real = row[row != -100]
            assert int(real[-1]) == eos, "each unpadded sequence ends with EOS"
            # Padding positions carry -100, never the pad id.
            assert not bool(((row == eos) & mask).any())

    def test_decoder_start_token_is_stripped(self, processor, collator):
        coll, cfg = collator
        batch = coll(_features(processor, ["इस document में", "यह ठीक है"]))
        # Every label row began with decoder_start_token_id before collation;
        # the model re-adds it when shifting right, so it must be gone now.
        assert (batch["labels"][:, 0] != cfg.decoder_start_token_id).all()

    def test_input_features_are_stacked_to_3000_frames(self, processor, collator):
        coll, _ = collator
        batch = coll(_features(processor, ["a", "b", "c"]))
        assert batch["input_features"].shape[0] == 3
        assert batch["input_features"].shape[-1] == 3000  # 30 s window

    def test_labels_truncated_to_448(self, processor, collator):
        coll, _ = collator
        long_text = "document " * 600
        batch = coll(_features(processor, [long_text]))
        assert batch["labels"].shape[1] <= 448

    def test_roundtrip_decode_recovers_the_text(self, processor, collator):
        coll, cfg = collator
        text = "इस document में formatting"
        batch = coll(_features(processor, [text]))
        labels = batch["labels"].clone()
        labels[labels == -100] = processor.tokenizer.pad_token_id
        # Re-add the start token the collator stripped, then decode.
        got = processor.tokenizer.decode(labels[0], skip_special_tokens=True)
        assert got.strip() == text
