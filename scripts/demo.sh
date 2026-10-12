#!/bin/bash
# Live demo of the trained models on one A100: transcribes MUCS test excerpts with the
# story set (zero-shot large-v2 against the fine-tuned ones) and writes an HTML report
# with audio players. Large-v2 is slow on CPU; this is the way to demo all of it.
#
#   cd /dist_home/d_palmani/c-08/codeswitching && sbatch scripts/demo.sh
#   MODELS="zeroshot-medium medium-qlora" N=5 sbatch scripts/demo.sh
#   MODE=transcribe AUDIO="/path/a.wav /path/b.flac" sbatch scripts/demo.sh
#
# Knobs: MODE (samples|transcribe), MODELS, N, SECONDS_PER_CLIP, SEED, AUDIO, OUT.
# The first load of each fine-tuned model converts it (a few minutes, large-v2 more),
# cached in ct2_models/. Report: results/demo/<MODE>.html unless OUT is set.
#SBATCH --job-name=csasr-demo
#SBATCH --partition=workq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-demo-%j.out

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO=$SLURM_SUBMIT_DIR
else
    REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
if [[ ! -f "$REPO/src/csasr/demo/cli.py" ]]; then
    echo "!! $REPO is not the codeswitching repo. Run:" >&2
    echo "   cd /dist_home/d_palmani/c-08/codeswitching && sbatch scripts/demo.sh" >&2
    exit 1
fi
cd "$REPO"

PY=${PYTHON:-$HOME/.venvs/csasr-abl/bin/python3}
if [[ ! -x "$PY" ]]; then
    echo "!! $PY not found: this is the ablation venv (peft, faster-whisper). Build it with" >&2
    echo "   sbatch scripts/ablate.sh (its environment stage), or set PYTHON=." >&2
    exit 1
fi

export PYTHONUNBUFFERED=1 USE_TF=0 USE_FLAX=0 HF_DATASETS_DISABLE_PROGRESS_BARS=1
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

: "${MODE:=samples}"
: "${MODELS:=story}"
: "${N:=4}"
: "${SECONDS_PER_CLIP:=25}"
: "${SEED:=0}"
ARGS=(--models $MODELS --device cuda)
[[ -n "${OUT:-}" ]] && ARGS+=(--html "$OUT")

echo "== $(date)  on $(hostname)  job ${SLURM_JOB_ID:-local}  mode=$MODE models=$MODELS =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

case "$MODE" in
    samples)
        "$PY" -m csasr.demo samples "${ARGS[@]}" --n "$N" --seconds "$SECONDS_PER_CLIP" --seed "$SEED" ;;
    transcribe)
        [[ -n "${AUDIO:-}" ]] || { echo "!! MODE=transcribe needs AUDIO=\"file1 file2\"" >&2; exit 1; }
        "$PY" -m csasr.demo transcribe "${ARGS[@]}" $AUDIO ;;
    *)
        echo "!! unknown MODE=$MODE (samples|transcribe)" >&2; exit 1 ;;
esac
echo "== finished $(date) =="
