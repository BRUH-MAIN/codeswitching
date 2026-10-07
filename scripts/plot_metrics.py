#!/usr/bin/env python
"""Plot train/eval curves from a train_whisper.py run, without needing a
TensorBoard server (the training scripts already log to TensorBoard too --
report_to="auto" in train_whisper.py -- but standing one up over a SLURM
compute node without port access is its own hassle; this just reads the same
data train_whisper.py's trainer.save_state() writes and renders a PNG).

    python scripts/plot_metrics.py --out csasr-work/exp/m6_large
    python scripts/plot_metrics.py --out csasr-work/exp/m6_large csasr-work/exp/m7_large \
        --save comparison.png

Reads <out>/trainer_state.json (Trainer's log_history: one dict per logged
step, train steps carrying 'loss'/'learning_rate', eval steps carrying
'eval_loss'/'eval_mer'/'eval_cba_he'/'eval_cba_eh'). If training is still
running, trainer_state.json won't exist yet at the top level -- pass the
latest checkpoint directory instead (checkpoint-N/trainer_state.json), which
Trainer keeps up to date throughout the run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: SLURM compute nodes have no display
import matplotlib.pyplot as plt

# (log_history key, subplot title, y-axis label)
PANELS = [
    ("loss", "Train loss", "loss"),
    ("eval_loss", "Val loss", "loss"),
    ("learning_rate", "Learning rate", "lr"),
    ("eval_mer", "Val MER", "%"),
    ("eval_cba_he", "Val CBA (hi->en)", "%"),
    ("eval_cba_eh", "Val CBA (en->hi)", "%"),
]


def _find_trainer_state(path: Path) -> Path:
    """Accept either a run's --out dir or a direct trainer_state.json path."""
    if path.is_file():
        return path
    direct = path / "trainer_state.json"
    if direct.exists():
        return direct
    # Training still in progress: fall back to the latest checkpoint's copy.
    checkpoints = sorted(
        path.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    for ckpt in reversed(checkpoints):
        candidate = ckpt / "trainer_state.json"
        if candidate.exists():
            return candidate
    raise SystemExit(
        f"No trainer_state.json under {path} (top-level or checkpoint-*/). "
        "Has training started yet?"
    )


def _series(log_history: list[dict], key: str) -> tuple[list[float], list[float]]:
    steps, values = [], []
    for entry in log_history:
        if key in entry:
            steps.append(entry["step"])
            values.append(entry[key])
    return steps, values


def plot_run(ax_row, label: str, log_history: list[dict]) -> None:
    for ax, (key, title, ylabel) in zip(ax_row, PANELS):
        steps, values = _series(log_history, key)
        if steps:
            ax.plot(steps, values, marker=".", markersize=3, label=label)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, nargs="+", required=True,
                    help="one or more train_whisper.py --out dirs (or trainer_state.json paths) to plot")
    ap.add_argument("--save", type=Path, default=None,
                    help="default: metrics.png next to the first --out, or ./metrics.png "
                         "if --out is a bare trainer_state.json")
    args = ap.parse_args(argv)

    runs = []
    for out in args.out:
        state_path = _find_trainer_state(out)
        log_history = json.loads(state_path.read_text())["log_history"]
        label = out.name if out.is_dir() else out.parent.name
        runs.append((label, log_history))
        print(f"[plot] {label}: {len(log_history)} logged entries from {state_path}")

    fig, axes = plt.subplots(len(runs), len(PANELS), figsize=(4 * len(PANELS), 3.2 * len(runs)), squeeze=False)
    for row, (label, log_history) in zip(axes, runs):
        plot_run(row, label, log_history)
    for ax in axes[:, 0]:
        ax.legend(loc="best", fontsize="small")
    fig.suptitle(" / ".join(label for label, _ in runs))
    fig.tight_layout()

    save = args.save
    if save is None:
        first = args.out[0]
        save = (first if first.is_dir() else first.parent) / "metrics.png"
    fig.savefig(save, dpi=150)
    print(f"[plot] saved -> {save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
