"""Dataset assembly for the Hub. Exercises the local half of push_to_hub
(schema construction), which is where the failures actually live -- the upload
itself is HF's problem.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("datasets")

import soundfile as sf  # noqa: E402

from csasr.data.push_to_hub import _dataset_from_manifest  # noqa: E402
from csasr.manifest import write_jsonl  # noqa: E402


def _wav(path, seconds=0.5, sr=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(int(sr * seconds), dtype=np.int16), sr, subtype="PCM_16")
    return path.as_posix()


class TestTextOnly:
    def test_bigrams_schema_has_no_utt_id(self, tmp_path):
        """bigrams_valid.jsonl carries {bigram, hi_word, en_word, checks}."""
        p = tmp_path / "bigrams.jsonl"
        write_jsonl(p, [
            {"bigram": "इस spoken", "hi_word": "इस", "en_word": "spoken",
             "checks": {"script": True, "not_translation": True}},
        ])
        ds = _dataset_from_manifest(p, with_audio=False)
        assert set(ds.column_names) == {"bigram", "hi_word", "en_word", "checks"}
        assert ds[0]["hi_word"] == "इस"

    def test_sentences_schema(self, tmp_path):
        p = tmp_path / "sentences.jsonl"
        write_jsonl(p, [
            {"sent_id": "a1", "text": "इस document में", "bigram": "इस document", "matrix_lang": "hi"},
        ])
        ds = _dataset_from_manifest(p, with_audio=False)
        assert set(ds.column_names) == {"sent_id", "text", "bigram", "matrix_lang"}

    def test_ragged_keys_are_filled_with_none(self, tmp_path):
        """A missing key in one row must not break the pyarrow schema."""
        p = tmp_path / "ragged.jsonl"
        write_jsonl(p, [
            {"utt_id": "a", "text": "x", "dur": 1.0},
            {"utt_id": "b", "text": "y"},            # no dur
        ])
        ds = _dataset_from_manifest(p, with_audio=False)
        assert ds[1]["dur"] is None
        assert len(ds) == 2


class TestWithAudio:
    def test_synthetic_manifest_keeps_speaker_and_bigram(self, tmp_path):
        p = tmp_path / "t2.jsonl"
        write_jsonl(p, [
            {"utt_id": "s1", "text": "इस document", "dur": 0.5,
             "wav": _wav(tmp_path / "a" / "s1.wav"), "speaker": "Rohit",
             "lang": "cs", "bigram": "इस document", "matrix_lang": "hi"},
            {"utt_id": "s2", "text": "यह file", "dur": 0.5,
             "wav": _wav(tmp_path / "a" / "s2.wav"), "speaker": "Divya",
             "lang": "cs", "bigram": "यह file", "matrix_lang": "hi"},
        ])
        ds = _dataset_from_manifest(p, with_audio=True)
        assert "audio" in ds.column_names
        assert ds[0]["audio"]["sampling_rate"] == 16000
        assert {r["speaker"] for r in ds} == {"Rohit", "Divya"}

    def test_all_null_columns_are_dropped(self, tmp_path):
        """Real MUCS rows have no speaker/bigram; those columns should vanish."""
        p = tmp_path / "test.jsonl"
        write_jsonl(p, [
            {"utt_id": "u1", "text": "इस document", "dur": 0.5, "wav": _wav(tmp_path / "b" / "u1.wav")},
        ])
        ds = _dataset_from_manifest(p, with_audio=True)
        assert set(ds.column_names) == {"audio", "utt_id", "text", "dur"}
        assert "speaker" not in ds.column_names
        assert "bigram" not in ds.column_names

    def test_missing_wav_field_is_rejected_loudly(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        write_jsonl(p, [{"utt_id": "u1", "text": "x", "dur": 1.0}])
        with pytest.raises(SystemExit, match="lack a `wav` field"):
            _dataset_from_manifest(p, with_audio=True)
