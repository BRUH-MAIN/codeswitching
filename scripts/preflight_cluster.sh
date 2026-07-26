#!/bin/bash
# Answer, in about a minute, the questions the training run would otherwise
# answer by dying 40 minutes in.
#
#   sbatch scripts/preflight_cluster.sh
#   cat slurm-preflight-*.out
#
# The SBATCH header below is your site's template verbatim. Nothing here assumes
# the GPU is an A100 -- `--gres=gpu:1` does not say which card you get, and
# whisper-large-v2 full fine-tuning needs ~24.7 GB of VRAM for optimizer state
# alone. This tells you before you queue a 24-hour job.

#SBATCH --job-name=csasr-preflight
#SBATCH --partition=workq
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=asaicomputenode02
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm-preflight-%j.out

set -uo pipefail   # NOT -e: every check should run even if an earlier one fails

echo "=============================================================="
echo " host      $(hostname -s)"
echo " date      $(date)"
echo " user      $USER"
echo " home      $HOME"
echo " submitdir ${SLURM_SUBMIT_DIR:-?}"
echo " cpus      ${SLURM_CPUS_PER_TASK:-?}"
echo " mem       ${SLURM_MEM_PER_NODE:-?} MB"
echo "=============================================================="

echo ""
echo "---- GPU ----------------------------------------------------"
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version \
           --format=csv 2>/dev/null || echo "  nvidia-smi FAILED -- no GPU visible?"

echo ""
echo "---- CPU / RAM ----------------------------------------------"
echo "  cores visible to this job: $(nproc)"
free -g | awk 'NR<=2 {print "  " $0}'

echo ""
echo "---- disk: where can 60+ GB live? ---------------------------"
# HF cache (17-35 GB of featurized Arrow) + checkpoints (12-19 GB) + parquet
# (~5 GB) + model weights (~6 GB). Home quotas are usually far too small.
for d in "$HOME" /scratch/"$USER" /scratch /tmp /dist_home/"$USER" "${SLURM_SUBMIT_DIR:-.}"; do
    if [[ -d "$d" ]]; then
        printf "  %-28s %s\n" "$d" "$(df -h "$d" 2>/dev/null | awk 'NR==2 {print $4" free of "$2}')"
        [[ -w "$d" ]] && echo "        writable" || echo "        NOT WRITABLE"
    else
        printf "  %-28s (does not exist)\n" "$d"
    fi
done
command -v quota >/dev/null && { echo "  quota:"; quota -s 2>/dev/null | sed 's/^/    /'; }

echo ""
echo "---- python / package ---------------------------------------"
PY=${PYTHON:-python3}
echo "  interpreter: $(command -v $PY)  ($($PY --version 2>&1))"
$PY - <<'PY'
import importlib, sys

def probe(mod, attr="__version__"):
    try:
        m = importlib.import_module(mod)
        print(f"  ok   {mod:16s} {getattr(m, attr, '?')}")
        return m
    except Exception as e:
        print(f"  MISSING {mod:13s} ({type(e).__name__})")
        return None

probe("csasr"); t = probe("torch"); probe("transformers")
probe("datasets"); probe("accelerate"); probe("omegaconf")

if t is None:
    sys.exit(0)
print(f"  torch CUDA build: {t.version.cuda}   available: {t.cuda.is_available()}")
if not t.cuda.is_available():
    print("  !! torch cannot see the GPU -- wrong wheel for this driver?")
    sys.exit(0)

p = t.cuda.get_device_properties(0)
vram = p.total_memory / 1e9
cap = (p.major, p.minor)
print(f"  device: {p.name}  sm{cap[0]}{cap[1]}  {vram:.0f} GB")
print(f"  bf16 supported: {t.cuda.is_bf16_supported()}")

# The decision this whole script exists to make.
STATE = 1.54e9 * 16 / 1e9          # fp32 param+grad+Adam for whisper-large-v2
print("")
print(f"  whisper-large-v2 full FT needs ~{STATE:.1f} GB of optimizer state alone.")
if cap < (8, 0):
    print("  => sm80+ (Ampere) NOT present: no bf16, no TF32. fp16 only.")
if vram < STATE * 1.25:
    print(f"  => {vram:.0f} GB is TOO SMALL for large-v2 full fine-tuning.")
    print("     Options: --optim adamw_bnb_8bit (state ~15 GB), or fall back to")
    print("     whisper-small via the `main` branch configs.")
elif vram < STATE * 1.6:
    print(f"  => {vram:.0f} GB fits large-v2 WITH gradient checkpointing (auto-detected).")
else:
    print(f"  => {vram:.0f} GB fits large-v2 comfortably; checkpointing auto-off.")
PY

echo ""
echo "---- hub reachability (compute nodes are often air-gapped) ---"
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "  HF_TOKEN unset -- export it before the real run"
else
    $PY - <<'PY'
import os
try:
    from huggingface_hub import HfApi
    who = HfApi().whoami(token=os.environ["HF_TOKEN"])
    print(f"  ok   authenticated as {who.get('name')}")
    api = HfApi()
    api.dataset_info("RohanRamesh/hi-en-synth-cs", token=os.environ["HF_TOKEN"])
    print("  ok   hi-en-synth-cs reachable")
except Exception as e:
    print(f"  NO NETWORK / NO ACCESS: {type(e).__name__}: {e}")
    print("  -> run scripts/prefetch_hub.py on a LOGIN node, then set")
    print("     HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 for the training job.")
PY
fi

echo ""
echo "=============================================================="
echo " preflight done. Check: GPU model, free disk >= 60 GB, hub access."
echo "=============================================================="
