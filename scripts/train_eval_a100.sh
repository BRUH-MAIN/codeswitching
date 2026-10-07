#!/bin/bash
# Train M6/M7/M8 AND decode+score them, as ONE sbatch job on ONE A100.
#
# train_a100.sh and eval_a100.sh still exist standalone (e.g. to re-score after
# a decode.py fix without retraining), but the normal path -- train, then eval
# the checkpoints you just made -- used to mean two separate `sbatch` submissions
# with a manual "next: sbatch scripts/eval_a100.sh" step in between, plus a second
# queue wait for the second job. This script holds the GPU allocation across both
# phases instead.
#
#   cd /path/to/codeswitching               # sbatch FROM THE REPO ROOT
#   sbatch scripts/preflight_cluster.sh     # DO THIS FIRST (~1 min)
#   sbatch scripts/train_eval_a100.sh
#   sbatch --export=ALL,MODELS="m6 m7" scripts/train_eval_a100.sh
#
# `MODELS` (default "m6 m7 m8") drives BOTH phases: whatever you train here is
# what gets decoded+scored here. See train_a100.sh for M7/M8's sequential
# chaining (each continues from the prior stage's checkpoint, per the paper's
# described procedure) and eval_a100.sh for the decode/score protocol
# (recording-level, language=none, matching the paper's WhisperX-style setup).
#
# `--gres=gpu:1` does not say WHICH card you get -- nothing below assumes an
# A100; precision/TF32/gradient-checkpointing are detected at runtime by
# _plan_hardware() in src/csasr/train/train_whisper.py.

#SBATCH --job-name=csasr-t2-train-eval
#SBATCH --partition=workq
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm-train-eval-%j.out

set -euo pipefail

# The site's system python3 is Ubuntu's, which ships without pip (ensurepip is
# disabled for the system interpreter on Debian/Ubuntu, so `python3 -m pip
# install ...` just dies with "No module named pip"). The `conda` module's
# python already has a working pip.
CONDA_PY=/dist_home/common-apps/conda/bin/python3
if [[ -n "${PYTHON:-}" ]]; then
    PY=$PYTHON
elif [[ -x "$CONDA_PY" ]]; then
    PY=$CONDA_PY
else
    PY=python3
fi

# Pure PyTorch codebase -- csasr never imports TensorFlow or Flax. But
# `transformers` defaults to USE_TF=AUTO and probes for TF at import time
# regardless, and the conda base env's TensorFlow (Keras 3) is incompatible
# with transformers' TF integration code, so importing Seq2SeqTrainer crashes
# on an unrelated backend we never asked for.
export USE_TF=0
export USE_FLAX=0

# sbatch fully buffers stdout (not line-buffered like a tty), so log lines pile
# up in memory instead of landing in the .out file as they happen.
export PYTHONUNBUFFERED=1

# datasets' own load/map progress bars use \r updates same as tqdm -- in a flat
# .out file those pile into one unreadable multi-KB line instead of overwriting
# in place. The decode loop has its own tty-aware periodic progress printer.
export HF_DATASETS_DISABLE_PROGRESS_BARS=1

# sbatch COPIES this script to /var/spool/slurmd/job*/slurm_script and runs it
# from there, so "${BASH_SOURCE[0]}" points at the spool copy, not the checkout.
# $SLURM_SUBMIT_DIR (the directory you ran sbatch from) is the only reliable
# anchor under sbatch.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO=$SLURM_SUBMIT_DIR
else
    REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
: "${REPO:=$PWD}"

if [[ ! -f "$REPO/configs/train_m7_large.yaml" ]]; then
    echo "Cannot find configs/ under REPO=$REPO." >&2
    echo "  Under sbatch, REPO is \$SLURM_SUBMIT_DIR -- so submit from the repo root:" >&2
    echo "      cd /path/to/codeswitching && sbatch scripts/train_eval_a100.sh" >&2
    echo "  Or override:  REPO=/path/to/codeswitching sbatch scripts/train_eval_a100.sh" >&2
    exit 1
fi
cd "$REPO"
echo "[repo] $REPO"

# ---- where the big files go ------------------------------------------------
: "${WORKDIR:=${SLURM_SUBMIT_DIR:-$PWD}/csasr-work}"
: "${HF_HOME:=$WORKDIR/hf}"
: "${OUT_ROOT:=$WORKDIR/exp}"
: "${RESULTS:=${SLURM_SUBMIT_DIR:-$PWD}/results}"
: "${MODELS:=m6 m7 m8}"
: "${ZERO_SHOT:=1}"
export HF_HOME
mkdir -p "$HF_HOME" "$OUT_ROOT" "$RESULTS"

: "${FROM_SCRATCH:=0}"
FROM_SCRATCH_FLAG=()
[[ "$FROM_SCRATCH" == "1" ]] && FROM_SCRATCH_FLAG=(--from-scratch)

# Fail now, not 40 minutes in, if the filesystem cannot hold the run.
avail_gb=$(df -BG --output=avail "$WORKDIR" | tail -1 | tr -dc '0-9')
echo "[disk] $WORKDIR has ${avail_gb} GB free"
if (( avail_gb < 60 )); then
    echo "!! Only ${avail_gb} GB free; M7 needs ~40 GB and M8 ~60 GB." >&2
    echo "   Set WORKDIR to a bigger filesystem (preflight lists candidates)," >&2
    echo "   or run MODELS=\"m6\" first." >&2
    (( avail_gb < 25 )) && exit 1
fi

# Token from the environment, never from argv -- argv leaks into every traceback
# and into `ps` output for every other user on the node.
# Prefer an already-exported HF_TOKEN; otherwise fall back to the shared .env
# one level above every project in this workspace (HF_ENV_FILE overrides the
# path). Job 11779 died at this exact check because HF_TOKEN had not been
# re-exported in the submitting shell.
: "${HF_ENV_FILE:=$(dirname "$REPO")/.env}"
if [[ -z "${HF_TOKEN:-}" && -f "$HF_ENV_FILE" ]]; then
    HF_TOKEN=$(grep -m1 -E '^[[:space:]]*HF_TOKEN[[:space:]]*=' "$HF_ENV_FILE" \
        | sed -E 's/^[[:space:]]*HF_TOKEN[[:space:]]*=[[:space:]]*//; s/[[:space:]]*$//; s/^"(.*)"$/\1/')
    [[ -n "$HF_TOKEN" ]] && export HF_TOKEN
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is unset and not found in $HF_ENV_FILE. export it, or add a" >&2
    echo "line there:" >&2
    echo "    HF_TOKEN = <token>" >&2
    exit 1
fi

echo "== $(date) on $(hostname -s) =="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
echo "cpus=${SLURM_CPUS_PER_TASK:-$(nproc)}  mem=${SLURM_MEM_PER_NODE:-?}MB"

# =============================================================================
# Phase 1: train
# =============================================================================

echo ""
echo "########################### TRAIN ###########################"
echo "[install] $PY -m pip install -e .[train]"
$PY -m pip install -e ".[train]"
$PY -c "import csasr; print('csasr', csasr.__version__)"

# Train_T1 is a by-reference subset, so M6 needs the id list. Fetch it once.
T1_IDS=$($PY - <<'PY'
import os
from huggingface_hub import hf_hub_download
print(hf_hub_download("RohanRamesh/hi-en-synth-cs", "t1_ids.json",
                      repo_type="dataset", token=os.environ["HF_TOKEN"]))
PY
)
echo "t1_ids: $T1_IDS"

free_map_cache () {
    local before after
    before=$(du -sk "$HF_HOME" 2>/dev/null | cut -f1)
    find "$HF_HOME" -name 'cache-*.arrow' -delete 2>/dev/null || true
    after=$(du -sk "$HF_HOME" 2>/dev/null | cut -f1)
    echo "[cache] freed $(( (before - after) / 1024 )) MB of featurized Arrow"
    df -h "$HF_HOME" | tail -1
}

run_model () {
    local m=$1; shift
    echo ""
    echo "===================== $m ====================="
    date
    free_map_cache
    # _plan_hardware()'s auto gradient-checkpointing heuristic only weighs
    # weight+grad+Adam state against VRAM, never activation memory, and left
    # checkpointing off + OOM'd on the very first step at batch-size 16 on a
    # run that "should" have fit (job 11745). Force it on explicitly.
    $PY -m csasr.train.train_whisper \
        --config "configs/train_${m}_large.yaml" \
        --out "$OUT_ROOT/${m}_large" \
        --gradient-checkpointing on \
        "${FROM_SCRATCH_FLAG[@]}" \
        "$@"
    df -h "$WORKDIR" | tail -1
}

# The paper trains M6/M7/M8 sequentially -- M7 continues from M6's checkpoint,
# and M8 (described as testing whether OOD monolingual data "could enhance a
# model previously fine-tuned with synthetic in-domain audio") continues from
# M7's -- rather than each stage independently full-fine-tuning openai/whisper-
# large-v2 from scratch. configs/train_m7_large.yaml / train_m8_large.yaml still
# say `model: openai/whisper-large-v2` as documentation of "what to use if
# running this stage standalone"; the --model override below implements the
# actual chaining.
require_prev_stage () {
    local prev_out=$1 stage=$2
    if [[ ! -d "$prev_out" ]]; then
        echo "!! $stage is chained from $prev_out (sequential fine-tuning, per the" >&2
        echo "   paper's procedure) but that directory does not exist. Run its prior" >&2
        echo "   stage first, or pass an explicit --model to fall back to a" >&2
        echo "   from-scratch start." >&2
        exit 1
    fi
}

for m in $MODELS; do
    case "$m" in
        m6) run_model m6 --subset-ids "$T1_IDS" ;;   # CLI overrides the yaml path
        m7) require_prev_stage "$OUT_ROOT/m6_large" m7
            run_model m7 --model "$OUT_ROOT/m6_large" ;;
        # M8 needs cv_en on the Hub. It fails at load time if absent, before any
        # GPU work -- so a missing cv_en costs nothing but must not kill m6/m7.
        m8) require_prev_stage "$OUT_ROOT/m7_large" m8
            run_model m8 --model "$OUT_ROOT/m7_large" || echo "!! m8 failed (cv_en not on the Hub yet?)" ;;
        *)  echo "unknown model '$m'" >&2; exit 2 ;;
    esac
done

if [[ "${PUSH:-0}" == "1" ]]; then
    for m in $MODELS; do
        [[ -d "$OUT_ROOT/${m}_large" ]] || continue
        $PY - "$m" "$OUT_ROOT/${m}_large" <<'PY'
import os, sys
from huggingface_hub import HfApi
m, folder = sys.argv[1], sys.argv[2]
repo = f"RohanRamesh/whisper-large-v2-cs-{m}"
api = HfApi()
api.create_repo(repo, exist_ok=True, private=True, token=os.environ["HF_TOKEN"])
api.upload_folder(folder_path=folder, repo_id=repo, token=os.environ["HF_TOKEN"])
print("pushed", repo)
PY
    done
fi

echo ""
echo "== train phase done $(date) =="
ls -la "$OUT_ROOT"

# =============================================================================
# Phase 2: eval
# =============================================================================

echo ""
echo "########################### EVAL ############################"
echo "[install] $PY -m pip install -e .[decode]"
$PY -m pip install -e ".[decode]"

REAL=RohanRamesh/mucs-he-cs
REFS=$RESULTS/refs_test.jsonl
TABLE=$RESULTS/table2_large.jsonl

nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

# --refs-out writes PER-UTTERANCE references even though decoding is at
# recording level. CBA's denominator has to come from the reference
# utterances, or it is inflated by ~23% with bigrams that span an utterance
# boundary and were never spoken adjacently.
decode () {
    local model=$1 tag=$2
    echo ""
    echo "---- decode $tag ($model) ----"
    $PY -m csasr.eval.decode \
        --model "$model" \
        --test-hf "$REAL" --test-config test \
        --out "$RESULTS/hyp_${tag}.jsonl" \
        --refs-out "$REFS" \
        --language none --batch-size 16 --lang-detect-segments 8
}

# MER mode=hybrid is the definition Gate 3 settled empirically: it puts large-v2
# zero-shot at 51.9 against the paper's 52.0. Both CBA rules are reported
# because the paper's matching rule is genuinely ambiguous.
score () {
    local tag=$1
    for cba in adjacent lenient; do
        $PY -m csasr.eval.score \
            --refs "$REFS" --hyps "$RESULTS/hyp_${tag}.jsonl" \
            --mer-mode hybrid --group recording --cba-mode "$cba" \
            --name "${tag}-large-v2-cba_${cba}" --out "$TABLE"
    done
}

if [[ "$ZERO_SHOT" == "1" ]]; then
    decode openai/whisper-large-v2 largev2_zeroshot
    score largev2_zeroshot
fi

for m in $MODELS; do
    ckpt="$OUT_ROOT/${m}_large"
    if [[ ! -d "$ckpt" ]]; then
        echo "!! $ckpt missing -- skipping $m (did the train phase above finish it?)"
        continue
    fi
    decode "$ckpt" "$m"
    score "$m"
done

echo ""
echo "== done $(date) =="
echo "Table 2 rows -> $TABLE"
$PY - <<PY
import json, pathlib
p = pathlib.Path("$TABLE")
if not p.exists():
    raise SystemExit("no results written")
rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
w = max(len(r.get("system", "?")) for r in rows)
print(f"{'system':<{w}}  {'MER':>6} {'CBA-HE':>7} {'CBA-EH':>7}")
for r in rows:
    print(f"{r.get('system','?'):<{w}}  {r['mer']:6.1f} {r['cba_he']:7.1f} {r['cba_eh']:7.1f}")
print("\npaper (large-v2): zero-shot 52.0 | M6 48.2 | M7 40.8 | M8 39.2")
PY
