"""Tests for `_copy_preprocessor_config`, the fix for a real production failure:
fine-tuned checkpoints saved by current `transformers` carry `processor_config.json`,
not the `preprocessor_config.json` name CTranslate2's converter and faster-whisper
expect. See ct2.py's docstring for the full story.
"""

from __future__ import annotations

import pytest

from csasr.eval.ct2 import _copy_preprocessor_config, resolve_ct2


class TestLocalDir:
    def test_preprocessor_config_present_is_copied_verbatim(self, tmp_path):
        src = tmp_path / "model"
        src.mkdir()
        (src / "preprocessor_config.json").write_text('{"n_mels": 80}', encoding="utf-8")
        out = tmp_path / "out"
        out.mkdir()

        _copy_preprocessor_config(str(src), out, token=None)

        assert (out / "preprocessor_config.json").read_text(encoding="utf-8") == '{"n_mels": 80}'

    def test_processor_config_is_renamed_to_preprocessor_config(self, tmp_path):
        """The actual bug: newer transformers only writes processor_config.json."""
        src = tmp_path / "model"
        src.mkdir()
        (src / "processor_config.json").write_text('{"n_mels": 80}', encoding="utf-8")
        out = tmp_path / "out"
        out.mkdir()

        _copy_preprocessor_config(str(src), out, token=None)

        assert (out / "preprocessor_config.json").read_text(encoding="utf-8") == '{"n_mels": 80}'
        assert not (out / "processor_config.json").exists()

    def test_preprocessor_config_preferred_over_processor_config(self, tmp_path):
        src = tmp_path / "model"
        src.mkdir()
        (src / "preprocessor_config.json").write_text("OLD", encoding="utf-8")
        (src / "processor_config.json").write_text("NEW", encoding="utf-8")
        out = tmp_path / "out"
        out.mkdir()

        _copy_preprocessor_config(str(src), out, token=None)

        assert (out / "preprocessor_config.json").read_text(encoding="utf-8") == "OLD"

    def test_neither_present_does_not_raise(self, tmp_path, capsys):
        src = tmp_path / "model"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()

        _copy_preprocessor_config(str(src), out, token=None)  # must not raise

        assert not (out / "preprocessor_config.json").exists()
        assert "feature-extractor defaults" in capsys.readouterr().out


class TestHubRepo:
    def test_downloads_and_renames_processor_config(self, tmp_path, monkeypatch):
        import huggingface_hub

        downloaded = tmp_path / "cache" / "processor_config.json"
        downloaded.parent.mkdir(parents=True)
        downloaded.write_text('{"n_mels": 80}', encoding="utf-8")

        from huggingface_hub.errors import EntryNotFoundError

        def fake_download(repo_id, filename, token=None):
            if filename == "preprocessor_config.json":
                raise EntryNotFoundError("not found")
            assert filename == "processor_config.json"
            return str(downloaded)

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

        out = tmp_path / "out"
        out.mkdir()
        _copy_preprocessor_config("user/repo", out, token="tok")

        assert (out / "preprocessor_config.json").read_text(encoding="utf-8") == '{"n_mels": 80}'

    def test_neither_file_in_repo_does_not_raise(self, tmp_path, monkeypatch, capsys):
        import huggingface_hub
        from huggingface_hub.errors import EntryNotFoundError

        def fake_download(repo_id, filename, token=None):
            raise EntryNotFoundError("not found")

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

        out = tmp_path / "out"
        out.mkdir()
        _copy_preprocessor_config("user/repo", out, token="tok")  # must not raise

        assert not (out / "preprocessor_config.json").exists()
        assert "feature-extractor defaults" in capsys.readouterr().out


class TestResolveCt2Passthrough:
    def test_prebuilt_models_bypass_conversion_entirely(self):
        # No ctranslate2 call, no filesystem/network touch -- these are direct
        # string lookups against PREBUILT_CT2.
        assert resolve_ct2("openai/whisper-base") == "Systran/faster-whisper-base"
        assert resolve_ct2("openai/whisper-large-v2") == "Systran/faster-whisper-large-v2"

    def test_existing_local_ct2_dir_passes_through(self, tmp_path):
        p = tmp_path / "already_converted"
        p.mkdir()
        (p / "model.bin").write_bytes(b"fake")
        assert resolve_ct2(str(p)) == str(p)
