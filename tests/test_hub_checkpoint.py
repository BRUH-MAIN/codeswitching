"""Tests for resumable, Hub-backed training checkpoints.

The pure path/parsing logic is tested directly. The Hub-talking pieces
(`build_hub_checkpoint_callback`, `clear_checkpoint`, `download_checkpoint`) are
tested against a fake `HfApi`/`snapshot_download`, never the real network --
same style as `test_backend.py`'s stub for LLM backends.
"""

from __future__ import annotations

import pytest

from csasr.train import hub_checkpoint as hc


class TestCheckpointPathInRepo:
    def test_default_kind_is_last(self):
        assert hc.checkpoint_path_in_repo("m6") == "m6/last-checkpoint"

    def test_best_kind(self):
        assert hc.checkpoint_path_in_repo("m6", kind="best") == "m6/best-checkpoint"

    def test_strips_slashes(self):
        assert hc.checkpoint_path_in_repo("m6/") == "m6/last-checkpoint"
        assert hc.checkpoint_path_in_repo("/m6") == "m6/last-checkpoint"

    def test_rejects_unknown_kind(self):
        with pytest.raises(ValueError):
            hc.checkpoint_path_in_repo("m6", kind="bogus")  # type: ignore[arg-type]


class TestDownloadCheckpoint:
    def test_missing_repo_returns_none(self, monkeypatch, tmp_path):
        import huggingface_hub
        from huggingface_hub.errors import RepositoryNotFoundError

        # The real exception needs a live HTTP `response` object to construct;
        # a bare subclass is enough to exercise the isinstance() catch in
        # download_checkpoint without faking an HTTP round trip.
        class _Fake404(RepositoryNotFoundError):
            def __init__(self):
                pass

        def fake_snapshot_download(*a, **kw):
            raise _Fake404()

        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
        assert hc.download_checkpoint("user/repo", "m6", token=None, local_dir=tmp_path) is None

    def test_incomplete_upload_returns_none(self, monkeypatch, tmp_path):
        """A prefix dir with no trainer_state.json is not a resumable checkpoint."""
        import huggingface_hub

        def fake_snapshot_download(repo_id, **kw):
            (tmp_path / "m6" / "last-checkpoint").mkdir(parents=True, exist_ok=True)
            return str(tmp_path)

        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
        assert hc.download_checkpoint("user/repo", "m6", token=None, local_dir=tmp_path) is None

    def test_valid_checkpoint_returns_its_path(self, monkeypatch, tmp_path):
        import huggingface_hub

        def fake_snapshot_download(repo_id, **kw):
            d = tmp_path / "m6" / "best-checkpoint"
            d.mkdir(parents=True, exist_ok=True)
            (d / "trainer_state.json").write_text("{}", encoding="utf-8")
            return str(tmp_path)

        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
        got = hc.download_checkpoint("user/repo", "m6", token=None, local_dir=tmp_path, kind="best")
        assert got == tmp_path / "m6" / "best-checkpoint"

    def test_unrelated_error_propagates(self, monkeypatch, tmp_path):
        import huggingface_hub

        def fake_snapshot_download(*a, **kw):
            raise RuntimeError("network is on fire")

        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
        with pytest.raises(RuntimeError):
            hc.download_checkpoint("user/repo", "m6", token=None, local_dir=tmp_path)


class _FakeApi:
    """Records every call instead of touching the network."""

    def __init__(self, calls, token=None):
        self._calls = calls

    def create_repo(self, repo_id, **kw):
        self._calls.append(("create_repo", repo_id))

    def upload_folder(self, *, folder_path, repo_id, path_in_repo, token, commit_message=None):
        self._calls.append(("upload_folder", path_in_repo))

    def super_squash_history(self, *, repo_id, token=None):
        self._calls.append(("squash", repo_id))

    def delete_folder(self, *, path_in_repo, repo_id, token=None):
        self._calls.append(("delete_folder", path_in_repo))


class TestClearCheckpoint:
    def test_deletes_the_run_folder(self, monkeypatch):
        import huggingface_hub

        calls = []
        monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: _FakeApi(calls))

        hc.clear_checkpoint("user/repo", "m6/", token="tok")
        assert calls == [("delete_folder", "m6")]

    def test_swallows_errors_rather_than_failing_a_finished_run(self, monkeypatch, capsys):
        import huggingface_hub

        class BoomApi:
            def __init__(self, token=None):
                pass

            def delete_folder(self, **kw):
                raise RuntimeError("hub is down")

        monkeypatch.setattr(huggingface_hub, "HfApi", BoomApi)
        hc.clear_checkpoint("user/repo", "m6", token="tok")  # must not raise
        assert "cleanup skipped" in capsys.readouterr().out


class TestHubCheckpointCallback:
    """Requires transformers for TrainerCallback/TrainerState/TrainingArguments."""

    @pytest.fixture(autouse=True)
    def _require_transformers(self):
        pytest.importorskip("transformers")

    def test_pushes_last_always_and_best_only_when_best(self, monkeypatch, tmp_path):
        import huggingface_hub
        from transformers import TrainerControl, TrainerState, TrainingArguments

        calls = []
        monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: _FakeApi(calls))

        cb = hc.build_hub_checkpoint_callback("user/repo", "m6", token="tok")
        assert ("create_repo", "user/repo") in calls

        out_dir = tmp_path / "out"
        ckpt = out_dir / "checkpoint-100"
        ckpt.mkdir(parents=True)
        (ckpt / "trainer_state.json").write_text("{}", encoding="utf-8")

        args = TrainingArguments(output_dir=str(out_dir))
        state = TrainerState()
        state.global_step = 100
        state.best_model_checkpoint = None
        control = TrainerControl()

        calls.clear()
        cb.on_save(args, state, control)
        assert ("upload_folder", "m6/last-checkpoint") in calls
        assert ("upload_folder", "m6/best-checkpoint") not in calls
        assert ("squash", "user/repo") in calls

        calls.clear()
        state.best_model_checkpoint = str(ckpt)  # this save just became the best
        cb.on_save(args, state, control)
        assert ("upload_folder", "m6/last-checkpoint") in calls
        assert ("upload_folder", "m6/best-checkpoint") in calls

    def test_missing_local_checkpoint_dir_is_a_silent_noop(self, monkeypatch, tmp_path):
        """on_save may fire for events that are not step-checkpoint saves; a
        directory that doesn't exist must not be treated as an error."""
        import huggingface_hub
        from transformers import TrainerControl, TrainerState, TrainingArguments

        calls = []
        monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: _FakeApi(calls))
        cb = hc.build_hub_checkpoint_callback("user/repo", "m6", token="tok")

        args = TrainingArguments(output_dir=str(tmp_path / "out"))
        state = TrainerState()
        state.global_step = 999
        control = TrainerControl()

        calls.clear()
        cb.on_save(args, state, control)  # checkpoint-999 was never created
        assert calls == []

    def test_push_failure_does_not_raise(self, monkeypatch, tmp_path):
        import huggingface_hub
        from transformers import TrainerControl, TrainerState, TrainingArguments

        class BoomApi:
            def __init__(self, token=None):
                pass

            def create_repo(self, *a, **kw):
                pass

            def upload_folder(self, **kw):
                raise RuntimeError("hub is down")

        monkeypatch.setattr(huggingface_hub, "HfApi", BoomApi)
        cb = hc.build_hub_checkpoint_callback("user/repo", "m6", token="tok")

        out_dir = tmp_path / "out"
        ckpt = out_dir / "checkpoint-1"
        ckpt.mkdir(parents=True)
        (ckpt / "trainer_state.json").write_text("{}", encoding="utf-8")

        args = TrainingArguments(output_dir=str(out_dir))
        state = TrainerState()
        state.global_step = 1
        control = TrainerControl()

        cb.on_save(args, state, control)  # must not raise; training must survive a bad push
