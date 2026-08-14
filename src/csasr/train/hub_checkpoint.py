"""Resumable training state, stored on the Hugging Face Hub instead of Kaggle's
ephemeral disk.

Kaggle can kill a session (walltime cap, disconnect, OOM) with zero warning and
zero chance to run cleanup code. Whatever sits on `/kaggle/working` at that
instant is gone, and `train_whisper.py` had no way to pick back up -- a killed
M7 meant re-running M7 from step 0. This module makes a run resumable across
that failure mode by pushing the Trainer's full on-disk checkpoint (model
weights + optimizer + scheduler + RNG + `trainer_state.json` -- everything
`Trainer._save_checkpoint` writes, not just the model) to a private Hub repo
every `save_steps`, and downloading it back at the start of the next attempt.

TWO checkpoints are tracked per run, not one:

* `last-checkpoint` -- pushed on every save. This is what a resumed run
  continues training FROM: correct optimizer momentum, LR schedule position,
  and global_step.
* `best-checkpoint` -- pushed only when a save becomes the new
  `state.best_model_checkpoint` (the one `load_best_model_at_end` reloads when
  training finishes). This exists because "last" and "best" are frequently
  DIFFERENT checkpoints -- e.g. a run that started overfitting before it died.
  `trainer_state.json` records the best checkpoint's path as a path on the
  DEAD session's disk, which does not exist on a fresh machine, so recovering
  it requires its own copy on the Hub, independent of "last".

One shared repo holds every run's state, keyed by a `run_name` subfolder
(`m6/last-checkpoint`, `m6/best-checkpoint`, `m7/...`), so a single
`--hub-checkpoint-repo` covers M6/M7/M8 without three extra repos.

Both are deleted from the Hub by an explicit `clear_checkpoint()` call, made
from `train_whisper.py` only after the FINAL model has been saved and pushed
-- deliberately not from a `TrainerCallback.on_train_end` hook, which fires
before that final save even starts. Cleaning up that early would delete the
one thing a crash during the final save could still be recovered from. If the
process is killed at any point before that explicit call, whatever was last
pushed stays put for the next attempt -- that asymmetry is the whole mechanism.

Every push also calls `super_squash_history`: `upload_folder` creates a new
Hub commit each time, and old commits keep their LFS blobs around (optimizer
state for a whisper-small full fine-tune is a few GB). Unsquashed, a run with
many saves would accumulate tens of GB of dead history for a repo that only
ever needs to hold its two LATEST states.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

__all__ = [
    "checkpoint_path_in_repo",
    "download_checkpoint",
    "build_hub_checkpoint_callback",
    "clear_checkpoint",
]

Kind = Literal["last", "best"]


def checkpoint_path_in_repo(run_name: str, kind: Kind = "last") -> str:
    """The fixed path a run's `kind` state lives at -- always overwritten in
    place, never accumulated as `checkpoint-1000/`, `checkpoint-2000/`, ..."""
    if kind not in ("last", "best"):
        raise ValueError(f"kind must be 'last' or 'best', got {kind!r}")
    return f"{run_name.strip('/')}/{kind}-checkpoint"


def download_checkpoint(
    repo_id: str,
    run_name: str,
    *,
    token: str | None,
    local_dir: str | Path,
    kind: Kind = "last",
) -> Optional[Path]:
    """Return a local path to `run_name`'s pushed `kind` checkpoint, or None.

    None covers three unremarkable cases, all meaning "nothing to recover
    here": the checkpoint repo does not exist yet, it exists but this run has
    never pushed this `kind`, or a previous push left an incomplete upload.
    Only a genuinely unexpected error propagates.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    prefix = checkpoint_path_in_repo(run_name, kind)
    try:
        downloaded = snapshot_download(
            repo_id,
            token=token,
            repo_type="model",
            allow_patterns=f"{prefix}/*",
            local_dir=str(local_dir),
        )
    except (RepositoryNotFoundError, EntryNotFoundError):
        return None
    except Exception as e:  # noqa: BLE001 - an empty/missing prefix raises assorted HTTP errors
        if "404" in str(e):
            return None
        raise

    ckpt_dir = Path(downloaded) / prefix
    if not (ckpt_dir / "trainer_state.json").exists():
        return None
    return ckpt_dir


def clear_checkpoint(repo_id: str, run_name: str, *, token: str | None) -> None:
    """Delete `run_name`'s resumable state (both kinds) from the Hub.

    Call this only after the final model is safely saved and pushed elsewhere
    -- see the module docstring for why this is a separate, explicit call
    rather than a Trainer callback hook.
    """
    from huggingface_hub import HfApi

    run_name = run_name.strip("/")
    api = HfApi(token=token)
    try:
        api.delete_folder(path_in_repo=run_name, repo_id=repo_id, token=token)
        print(f"[hub-ckpt] {run_name}: finished cleanly, cleared resume state from the Hub")
    except Exception as e:  # noqa: BLE001 - cleanup failure must not fail an otherwise-finished run
        print(f"[hub-ckpt] cleanup skipped ({type(e).__name__}: {e})")


def build_hub_checkpoint_callback(repo_id: str, run_name: str, *, token: str | None):
    """Construct the `TrainerCallback` that pushes/clears state for `run_name`.

    `transformers` is imported lazily here (not at module import time) so the
    pure functions above stay importable -- and unit-testable -- without torch
    or transformers installed, matching the rest of this package's convention
    of keeping heavy ML deps out of Stage-0-reachable import paths.
    """
    from huggingface_hub import HfApi
    from transformers import TrainerCallback

    run_name = run_name.strip("/")
    api = HfApi(token=token)
    api.create_repo(repo_id, exist_ok=True, private=True, repo_type="model", token=token)

    def _push(local_dir: Path, kind: Kind, step: int) -> None:
        dest = checkpoint_path_in_repo(run_name, kind)
        print(f"[hub-ckpt] {run_name}: pushing {kind} (step {step:,}) -> "
              f"{repo_id}:{dest}", flush=True)
        api.upload_folder(
            folder_path=str(local_dir),
            repo_id=repo_id,
            path_in_repo=dest,
            token=token,
            commit_message=f"{run_name} {kind} checkpoint @ step {step}",
        )
        api.super_squash_history(repo_id=repo_id, token=token)

    class HubCheckpointCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if not ckpt_dir.is_dir():
                return control
            try:
                _push(ckpt_dir, "last", state.global_step)
                is_best = (
                    state.best_model_checkpoint is not None
                    and Path(state.best_model_checkpoint).resolve() == ckpt_dir.resolve()
                )
                if is_best:
                    _push(ckpt_dir, "best", state.global_step)
            except Exception as e:  # noqa: BLE001 - a failed push must never crash training
                print(f"[hub-ckpt] push failed ({type(e).__name__}: {e}); "
                      f"local checkpoint is intact, will retry at the next save")
            return control

    return HubCheckpointCallback()
