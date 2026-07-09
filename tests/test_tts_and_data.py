import numpy as np
import pytest

from csasr.manifest import Utt, read_jsonl, total_hours, write_jsonl
from csasr.tts.make_subset import select_subset
from csasr.tts.speakers import SPEAKERS, assign_speaker, description_for
from csasr.tts.synthesize import to_int16_16k, trim_silence


class TestSpeakers:
    def test_two_voices_one_male_one_female(self):
        assert {s.gender for s in SPEAKERS} == {"male", "female"}
        assert {s.name for s in SPEAKERS} == {"Rohit", "Divya"}

    def test_description_contains_the_quality_phrase(self):
        _, d = description_for("abc")
        assert "very clear audio" in d

    def test_assignment_is_stable_across_processes(self):
        # Python's builtin hash() is salted per process; a resumed Kaggle run
        # would otherwise reassign voices mid-corpus.
        assert assign_speaker("sent-0001").name == assign_speaker("sent-0001").name
        import subprocess, sys

        code = (
            "from csasr.tts.speakers import assign_speaker;"
            "print(assign_speaker('sent-0001').name)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            env={"PYTHONHASHSEED": "42", "PATH": ""},
        )
        if out.returncode == 0:
            assert out.stdout.strip() == assign_speaker("sent-0001").name

    def test_roughly_balanced_split(self):
        names = [assign_speaker(f"s{i:05d}").name for i in range(2000)]
        frac = names.count("Rohit") / len(names)
        assert 0.45 < frac < 0.55


class TestAudioHelpers:
    def test_trim_silence_removes_leading_and_trailing(self):
        x = np.concatenate([np.zeros(1000), np.ones(500) * 0.5, np.zeros(1000)])
        out = trim_silence(x, sr=22050, pad_ms=0)
        assert out.size == 500

    def test_trim_silence_on_all_silence_returns_empty(self):
        assert trim_silence(np.zeros(1000), sr=22050).size == 0

    def test_to_int16_16k_resamples_and_clips(self):
        sr = 22050
        x = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
        pcm = to_int16_16k(x, sr)
        assert pcm.dtype == np.int16
        assert abs(pcm.size - 16000) < 50  # 1 second at 16 kHz

    def test_to_int16_normalizes_overshoot(self):
        pcm = to_int16_16k(np.array([2.0, -2.0, 0.0], dtype=np.float32), 16000)
        assert pcm.max() <= 32767 and pcm.min() >= -32768


class TestSubset:
    def _rows(self, n=100, dur=10.0, bigrams=10):
        return [
            {"utt_id": f"u{i}", "dur": dur, "bigram": f"bg{i % bigrams}"}
            for i in range(n)
        ]

    def test_subset_hits_duration_target(self):
        rows = self._rows(n=100, dur=10.0)  # 1000 s total
        ids = select_subset(rows, target_hours=500 / 3600, seed=1)
        picked = sum(10.0 for _ in ids)
        assert 500 <= picked <= 510  # stops at or just past target

    def test_subset_is_strict_and_unique(self):
        rows = self._rows()
        ids = select_subset(rows, target_hours=0.1, seed=1)
        assert len(ids) == len(set(ids))
        assert set(ids) < {r["utt_id"] for r in rows}

    def test_subset_is_deterministic(self):
        rows = self._rows()
        assert select_subset(rows, 0.1, seed=7) == select_subset(rows, 0.1, seed=7)

    def test_subset_spreads_across_bigrams(self):
        rows = self._rows(n=100, dur=10.0, bigrams=10)
        ids = select_subset(rows, target_hours=100 / 3600, seed=1)  # ~10 clips
        by = {i: f"bg{int(i[1:]) % 10}" for i in ids}
        # Round-robin should touch ~10 distinct bigrams, not 1.
        assert len(set(by.values())) >= 8

    def test_subset_stops_when_corpus_exhausted(self):
        rows = self._rows(n=5, dur=1.0)  # 5 s total
        ids = select_subset(rows, target_hours=10.0, seed=1)
        assert len(ids) == 5


class TestManifest:
    def test_round_trip_preserves_extra_fields(self, tmp_path):
        p = tmp_path / "m.jsonl"
        write_jsonl(p, [Utt(utt_id="a", text="hi", dur=1.5, extra={"bigram": "x y"})])
        rows = list(read_jsonl(p))
        assert rows[0]["bigram"] == "x y"
        assert rows[0]["dur"] == 1.5
        assert "wav" not in rows[0]  # None fields are dropped

    def test_total_hours(self):
        assert total_hours([{"dur": 1800}, {"dur": 1800}]) == pytest.approx(1.0)

    def test_from_dict_captures_unknown_keys_into_extra(self):
        u = Utt.from_dict({"utt_id": "a", "text": "t", "bigram": "x y"})
        assert u.extra == {"bigram": "x y"}
