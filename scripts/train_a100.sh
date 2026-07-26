#!/bin/bash
# Train M6 / M7 / M8 on one GPU. No notebook involved -- csasr.train is a real
# CLI, which is the payoff of having kept the Kaggle notebooks thin.
#
#   cd /path/to/codeswitching               # sbatch FROM THE REPO ROOT
#   sbatch scripts/preflight_cluster.sh     # DO THIS FIRST (~1 min)
#   sbatch scripts/train_a100.sh
#   sbatch --export=ALL,MODELS="m6 m7" scripts/train_a100.sh
#
# `MODELS=... sbatch ...` also works, but only because sbatch defaults to
# --export=ALL. Some sites configure --export=NONE, and then the variable
# silently vanishes and you get all three models. --export=ALL,VAR=... is
# explicit and works either way. HF_TOKEN travels by the same mechanism, so if
# the token check below trips on a site you know exports it, that is why.
#
# SBATCH header follows the site template. `--gres=gpu:1` does not say WHICH
# card you get, so nothing below assumes an A100: precision, TF32 and gradient
# checkpointing are all detected at runtime by _plan_hardware() in
# src/csasr/train/train_whisper.py.

#SBATCH --job-name=csasr-t2
#SBATCH --partition=workq
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=asaicomputenode02
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm-train-%j.out

set -euo pipefail

PY=${PYTHON:-python3}          # the site template calls python3, not python

# ---- locate the repo -------------------------------------------------------
# sbatch COPIES this script to /var/spool/slurmd/job*/slurm_script and runs it
# from there, so "${BASH_SOURCE[0]}" points at the spool copy, not the checkout.
# $SLURM_SUBMIT_DIR (the directory you ran sbatch from) is the only reliable
# anchor under sbatch; BASH_SOURCE is the fallback for `bash scripts/...`.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO=$SLURM_SUBMIT_DIR
else
    REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
: "${REPO:=$PWD}"

if [[ ! -f "$REPO/configs/train_m7_large.yaml" ]]; then
    echo "Cannot find configs/ under REPO=$REPO." >&2
    echo "  Under sbatch, REPO is \$SLURM_SUBMIT_DIR -- so submit from the repo root:" >&2
    echo "      cd /path/to/codeswitching && sbatch scripts/train_a100.sh" >&2
    echo "  Or override:  REPO=/path/to/codeswitching sbatch scripts/train_a100.sh" >&2
    exit 1
fi
cd "$REPO"
echo "[repo] $REPO"

# ---- where the big files go ------------------------------------------------
# This needs ~60 GB and it must NOT be a quota'd home directory:
#   parquet download   ~5 GB
#   model weights      ~6 GB
#   featurized Arrow   5.8 GB (M6) / 17.3 GB (M7) / 35.5 GB (M8)
#   checkpoints        ~12 GB   (6.2 GB each x save_total_limit 2; the AdamW
#                                state is dropped -- see save_only_model)
# Override WORKDIR if preflight showed a better filesystem.
: "${WORKDIR:=${SLURM_SUBMIT_DIR:-$PWD}/csasr-work}"
: "${HF_HOME:=$WORKDIR/hf}"
: "${OUT_ROOT:=$WORKDIR/exp}"
: "${MODELS:=m6 m7 m8}"
export HF_HOME
mkdir -p "$HF_HOME" "$OUT_ROOT"

# Fail now, not 40 minutes in, if the filesystem cannot hold the run.
avail_gb=$(df -BG --output=avail "$WORKDIR" | tail -1 | tr -dc '0-9')
echo "[disk] $WORKDIR has ${avail_gb} GB free"
if (( avail_gb < 60 )); then
    echo "!! Only ${avail_gb} GB free; M7 needs ~40 GB and M8 ~60 GB." >&2
    echo "   Set WORKDIR to a bigger filesystem (preflight lists candidates)," >&2
    echo "   or run MODELS=\"m6\" first." >&2
    (( avail_gb < 25 )) && exit 1     # below this even M6 cannot finish
fi

# Token from the environment, never from argv -- argv leaks into every traceback
# and into `ps` output for every other user on the node.
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is unset. export it, or source a file that does:" >&2
    echo "    export HF_TOKEN=\$(cat ~/.hf_token)" >&2
    exit 1
fi

echo "== $(date) on $(hostname -s) =="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
echo "cpus=${SLURM_CPUS_PER_TASK:-$(nproc)}  mem=${SLURM_MEM_PER_NODE:-?}MB"
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

# ---- the featurized cache is what fills the disk --------------------------
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
    # Only the cache-*.arrow files go; the parquet download is the expensive part.
    free_map_cache
    $PY -m csasr.train.train_whisper \
        --config "configs/train_${m}_large.yaml" \
        --out "$OUT_ROOT/${m}_large" \
        "$@"
    df -h "$WORKDIR" | tail -1
}

for m in $MODELS; do
    case "$m" in
        m6) run_model m6 --subset-ids "$T1_IDS" ;;   # CLI overrides the yaml path
        m7) run_model m7 ;;
        # M8 needs cv_en on the Hub. It fails at load time if absent, before any
        # GPU work -- so a missing cv_en costs nothing but must not kill m6/m7.
        m8) run_model m8 || echo "!! m8 failed (cv_en not on the Hub yet?)" ;;
        *)  echo "unknown model '$m'" >&2; exit 2 ;;
    esac
done

# ---- optional: publish the checkpoints ------------------------------------
# Not needed for eval -- scripts/eval_a100.sh reads $OUT_ROOT directly. Only for
# sharing. Note the repo name says large-v2, NOT small: the Kaggle branch's
# whisper-small-cs-* repos are a different model and must not be overwritten.
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
echo "== done $(date) =="
ls -la "$OUT_ROOT"
echo ""
echo "next: WORKDIR=$WORKDIR sbatch scripts/eval_a100.sh"
