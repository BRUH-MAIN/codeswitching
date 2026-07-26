#!/usr/bin/env bash
# Train M6 / M7 / M8 on a single A100. No notebook involved -- csasr.train is a
# real CLI, which is the payoff of having kept the Kaggle notebooks thin.
#
#   sbatch scripts/train_a100.sh            # as a SLURM job
#   bash   scripts/train_a100.sh            # interactively on a GPU node
#   MODELS="m6 m7" bash scripts/train_a100.sh
#
# Precision, TF32 and gradient checkpointing are auto-detected from the card;
# see _plan_hardware in src/csasr/train/train_whisper.py. Nothing here hardcodes
# 40 GB vs 80 GB.
#
#SBATCH --job-name=csasr-t2
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

# ---- environment ----------------------------------------------------------
# HF_HOME must NOT be a home directory. The featurized Arrow cache is 0.96 MB
# per clip: ~5.8 GB for M6, ~17.3 GB for M7, ~35.5 GB for M8. Home quotas are
# typically 10-50 GB and the failure arrives an hour into the job.
: "${HF_HOME:=/scratch/$USER/hf}"
: "${OUT_ROOT:=/scratch/$USER/csasr/exp}"
: "${MODELS:=m6 m7 m8}"
export HF_HOME
mkdir -p "$HF_HOME" "$OUT_ROOT"

# Token from the environment, never from argv -- argv leaks into every traceback
# and into `ps` output for every other user on the node.
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is unset. export it, or source a file that does:" >&2
    echo "    export HF_TOKEN=\$(cat ~/.hf_token)" >&2
    exit 1
fi

echo "== $(date) =="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
df -h "$HF_HOME" | tail -1
python -c "import csasr; print('csasr', csasr.__version__)"

# Train_T1 is a by-reference subset, so M6 needs the id list. Fetch it once.
T1_IDS=$(python - <<'PY'
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
    python -m csasr.train.train_whisper \
        --config "configs/train_${m}_large.yaml" \
        --out "$OUT_ROOT/${m}_large" \
        "$@"
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
        python - "$m" "$OUT_ROOT/${m}_large" <<'PY'
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
echo "next: bash scripts/eval_a100.sh"
