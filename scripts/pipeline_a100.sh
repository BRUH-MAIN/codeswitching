#!/bin/bash
# The FULL Track 2 pipeline as one sbatch job: data creation -> LLM text ->
# TTS audio -> train M6/M7/M8 -> decode+score, on this SLURM cluster.
#
# README's documented path keeps stages 1-2 (LLM text, TTS audio) on Kaggle
# because they need two GPUs (llama.cpp tensor_split, then TTS shards) and
# each pins a DIFFERENT, mutually incompatible transformers version -- and
# because the corpus this project actually uses is already generated and
# pushed to the Hub, so re-running 0-2 is not needed for day-to-day work. This
# script exists for reproducing that corpus from scratch on THIS cluster
# instead of Kaggle (e.g. if Kaggle access is lost, or to hand the whole
# recipe to someone else). It has the SAME staged-environment problem the
# Kaggle notebooks solve by being separate notebooks/kernels: each stage below
# reinstalls exactly the extras IT needs right before running, mirroring
# scripts/build_notebooks.py cell-for-cell (the source of truth for every flag
# below -- this does not re-derive them from configs/llm.yaml or tts.yaml,
# which are descriptive only and are not read by any of this code).
#
#   cd /path/to/codeswitching               # sbatch FROM THE REPO ROOT
#   sbatch scripts/preflight_cluster.sh     # DO THIS FIRST (~1 min)
#   sbatch scripts/pipeline_a100.sh                       # everything
#   sbatch --export=ALL,STAGES="3 4" scripts/pipeline_a100.sh   # train+eval only,
#                                                                 # data already on the Hub
#   sbatch --export=ALL,STAGES="0" scripts/pipeline_a100.sh     # just refresh stage 0
#
# STAGES (default "0 1 2 3 4"):
#   0  data prep (OpenSLR MUCS + Common Voice, CPU) + GATE 0, push train_text/dev/test/cv_*
#   1  LLM bigram+sentence generation (GPU, both cards via tensor_split) + GATE 1 / 1.5, push
#   2  TTS audio synthesis (GPU, sharded across both cards) + GATE 2, push synth_t2 + GATE 2b
#   3  train M6 -> M7 -> M8 (sequential, one GPU)
#   4  decode + score against the paper's Table 2 (one GPU)
#
# Every stage after the first is resumable in its own way: stage 0's downloads
# are resumable, stage 1/2's LLM calls are cached under data/llm_cache and TTS
# shards skip clips already on disk, and stage 3 resumes from the latest
# Trainer checkpoint under --out (or chains from the prior model's checkpoint,
# per train_a100.sh). If this job is killed by the walltime cap, resubmitting
# with the same WORKDIR and the same or narrower STAGES continues rather than
# restarting from zero -- EXCEPT stage 0/1/2's local manifests under
# manifests/, which live in $REPO, not $WORKDIR, and are NOT deleted between
# stages (each stage regenerates them from scratch if missing but otherwise
# reuses what's on disk).
#
# Runtime is dominated by stage 1 (LLM generation, ~8-12h on 2 GPUs per the
# paper's own recipe scale) and stage 2 (TTS synthesis, ~4-6h) ahead of
# training (~1 day for M6+M7+M8 sequentially) and a cheap eval pass. --time
# below is sized for all of it with headroom under this partition's 3-day cap;
# narrow STAGES to a subset for a tighter allocation.
#
# Two A100s are reserved (asaicomputenode02/03 each have gpu:a100:2) because
# stage 1's tensor_split and stage 2's shard parallelism both use both cards.
# Stages 3/4 are single-GPU by design (train_whisper.py's effective-batch
# assertion assumes it) -- CUDA_VISIBLE_DEVICES is pinned to 0 for those below
# so accelerate/Trainer never silently spreads across both and changes the
# paper's effective batch size out from under the recipe.

#SBATCH --job-name=csasr-t2-pipeline
#SBATCH --partition=workq
#SBATCH --time=68:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-pipeline-%j.out

set -euo pipefail

# See train_a100.sh: the system python3 has no pip (ensurepip is disabled for
# it on Debian/Ubuntu). The `conda` module's python has a working one.
CONDA_PY=/dist_home/common-apps/conda/bin/python3
if [[ -n "${PYTHON:-}" ]]; then
    PY=$PYTHON
elif [[ -x "$CONDA_PY" ]]; then
    PY=$CONDA_PY
else
    PY=python3
fi

export USE_TF=0
export USE_FLAX=0
export PYTHONUNBUFFERED=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=1

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO=$SLURM_SUBMIT_DIR
else
    REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
: "${REPO:=$PWD}"

if [[ ! -f "$REPO/configs/train_m7_large.yaml" ]]; then
    echo "Cannot find configs/ under REPO=$REPO." >&2
    echo "  Under sbatch, REPO is \$SLURM_SUBMIT_DIR -- so submit from the repo root:" >&2
    echo "      cd /path/to/codeswitching && sbatch scripts/pipeline_a100.sh" >&2
    exit 1
fi
cd "$REPO"
echo "[repo] $REPO"

: "${STAGES:=0 1 2 3 4}"
: "${WORKDIR:=${SLURM_SUBMIT_DIR:-$PWD}/csasr-work}"
: "${HF_HOME:=$WORKDIR/hf}"
: "${OUT_ROOT:=$WORKDIR/exp}"
: "${RESULTS:=${SLURM_SUBMIT_DIR:-$PWD}/results}"
: "${MODELS:=m6 m7 m8}"
: "${ZERO_SHOT:=1}"
: "${REAL_REPO:=RohanRamesh/mucs-he-cs}"
: "${SYNTH_REPO:=RohanRamesh/hi-en-synth-cs}"
: "${N_GPU:=2}"
export HF_HOME
mkdir -p "$HF_HOME" "$OUT_ROOT" "$RESULTS" manifests data/llm_cache data/synth/audio

# Prefer an already-exported HF_TOKEN; otherwise fall back to the shared .env
# one level above every project in this workspace (HF_ENV_FILE overrides the
# path).
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

stage_selected () {
    case " $STAGES " in
        *" $1 "*) return 0 ;;
        *)        return 1 ;;
    esac
}

echo "== $(date) on $(hostname -s) =="
echo "[stages] $STAGES"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

# =============================================================================
# Stage 0: data prep (CPU) -- GATE 0
# =============================================================================
if stage_selected 0; then
    echo ""
    echo "##################### STAGE 0: data prep #####################"
    $PY -m pip install -e ".[local]"

    echo "==> download MUCS (7.3 GB + 443 MB, resumable)"
    $PY -m csasr.data.download --splits test train

    echo "==> prepare test (cut audio) and train (TEXT ONLY) + 4h dev slice"
    $PY -m csasr.data.prepare_mucs \
        --root data/mucs/test --out manifests/mucs_test.jsonl \
        --cut-audio --audio-dir data/mucs_cut/test
    $PY -m csasr.data.prepare_mucs \
        --root data/mucs/train --out manifests/mucs_train.jsonl \
        --dev-hours 4 --dev-out manifests/mucs_dev.jsonl

    echo "==> cut audio for the 4h dev slice ONLY (never the other 86h)"
    $PY -m csasr.data.prepare_mucs \
        --root data/mucs/train --out manifests/mucs_dev.jsonl \
        --keep-ids manifests/mucs_dev.jsonl \
        --cut-audio --audio-dir data/mucs_cut/dev

    echo "==> GATE 0 -- reproduce Table 1"
    $PY scripts/verify_table1.py \
        --train manifests/mucs_train.jsonl --test manifests/mucs_test.jsonl

    echo "==> Common Voice: stream 15h each of hi and en"
    $PY -m csasr.data.prepare_cv --langs hi en --hours 15 --max-per-speaker 200

    echo "==> publish to $REAL_REPO"
    $PY -m csasr.data.push_to_hub --manifest manifests/mucs_train.jsonl \
        --repo "$REAL_REPO" --config train_text --text-only --verify
    for cfg in dev test; do
        $PY -m csasr.data.push_to_hub --manifest "manifests/mucs_${cfg}.jsonl" \
            --repo "$REAL_REPO" --config "$cfg" --verify
    done
    for lang in hi en; do
        $PY -m csasr.data.push_to_hub --manifest "manifests/cv_${lang}.jsonl" \
            --repo "$REAL_REPO" --config "cv_${lang}" --verify
    done
    echo "[stage 0] done $(date)"
fi

# =============================================================================
# Stage 1: LLM bigram + sentence generation (GPU, tensor_split across both
# cards -- ONE llama.cpp process uses both, so nothing here is per-GPU sharded)
# -- GATE 1 / GATE 1.5
# =============================================================================
if stage_selected 1; then
    echo ""
    echo "##################### STAGE 1: LLM text #####################"
    $PY -m pip install -e ".[llm]"
    # UNVERIFIED ON THIS CLUSTER: the notebook's Kaggle wheel URL is built
    # against Kaggle's specific torch/CUDA image and will very likely NOT match
    # this cluster's driver. Build from source against whatever CUDA module is
    # loaded instead -- both flag names are passed because llama.cpp renamed
    # LLAMA_CUBLAS to GGML_CUDA at some point in its history and the pin here
    # (llama-cpp-python>=0.3.0) does not fix which side of that rename you get.
    # The smoke test right below is what actually catches a bad build, in
    # minutes rather than 8-12 hours into the real run.
    CMAKE_ARGS="-DGGML_CUDA=on -DLLAMA_CUBLAS=on" \
        $PY -m pip install --force-reinstall --no-cache-dir llama-cpp-python
    $PY -c "
import llama_cpp
print('llama_cpp', llama_cpp.__version__)
assert llama_cpp.llama_supports_gpu_offload(), (
    'CPU-only wheel built -- the CUDA toolchain/module this cluster loaded is '
    'not what llama-cpp-python compiled against. Check module load cuda/... '
    'matches nvidia-smi, then re-run this stage.'
)
"

    LLM=unsloth/gemma-4-26B-A4B-it-GGUF
    GGUF=gemma-4-26B-A4B-it-UD-Q4_K_M.gguf

    if [[ ! -f manifests/mucs_train.jsonl ]]; then
        echo "==> pulling in-domain transcripts from $REAL_REPO (few-shot exemplars)"
        $PY - <<PY
import os
from datasets import load_dataset
from csasr.manifest import write_jsonl
ds = load_dataset("$REAL_REPO", "train_text", split="train", token=os.environ["HF_TOKEN"])
write_jsonl("manifests/mucs_train.jsonl", [dict(r) for r in ds])
print(f"{len(ds):,} in-domain sentences for few-shot prompting")
PY
    fi

    echo "==> SMOKE TEST -- 20 bigrams, before committing to the full ~8-12h run"
    $PY -m csasr.llm.smoke --backend llamacpp --model "$LLM" --gguf-file "$GGUF" \
        --train-manifest manifests/mucs_train.jsonl \
        --n-calls 2 --bigrams-per-call 10

    echo "==> generate bigrams (paper: 44,657 raw -> 5,932 unique, 13.3%)"
    $PY -m csasr.llm.gen_bigrams --backend llamacpp --model "$LLM" --gguf-file "$GGUF" \
        --train-manifest manifests/mucs_train.jsonl \
        --out manifests/bigrams_raw.jsonl \
        --cache data/llm_cache/bigrams.jsonl \
        --n-calls 15000

    echo "==> filter (script filter + LLM translation check, 3-sample vote)"
    $PY -m csasr.llm.filter_bigrams --backend llamacpp --model "$LLM" --gguf-file "$GGUF" \
        --raw manifests/bigrams_raw.jsonl --out manifests/bigrams_valid.jsonl \
        --cache data/llm_cache/transcheck.jsonl \
        --items-per-call 20 --n-samples 3

    echo "==> expand each bigram into 4 sentences (2 English-matrix, 2 Hindi-matrix)"
    $PY -m csasr.llm.gen_sentences --backend llamacpp --model "$LLM" --gguf-file "$GGUF" \
        --bigrams manifests/bigrams_valid.jsonl --out manifests/sentences.jsonl \
        --cache data/llm_cache/sentences.jsonl

    echo "==> GATE 1 -- yields must track the paper"
    $PY - <<'PY'
from csasr.manifest import read_jsonl
raw   = list(read_jsonl("manifests/bigrams_raw.jsonl"))
uniq  = {r["bigram"] for r in raw}
valid = list(read_jsonl("manifests/bigrams_valid.jsonl"))
sents = list(read_jsonl("manifests/sentences.jsonl"))
rows = [
    ("raw bigrams",    len(raw),   44_657, None),
    ("unique bigrams", len(uniq),   5_932, len(uniq) / max(len(raw), 1)),
    ("valid bigrams",  len(valid),  5_477, len(valid) / max(len(uniq), 1)),
    ("sentences",      len(sents), 16_000, None),
]
print(f"{'metric':<16}{'ours':>10}{'paper':>10}{'survival':>12}")
for name, got, want, surv in rows:
    s = f"{surv:.1%}" if surv else "-"
    print(f"{name:<16}{got:>10,}{want:>10,}{s:>12}")
print("\npaper survival: dedup 13.3%, filter 92.3%")
est_h = sum(len(r["text"].split()) for r in sents) / 2.4 / 3600
print(f"projected TTS audio: ~{est_h:.1f} h (Train_T1 needs 8h; paper's Train_T2 = 22h)")
if est_h < 16:
    print("WARNING: Train_T2 would be far short of 22h -- raise --n-calls above and "
          "re-run this stage; cached calls are free, only the new ones cost time.")
PY

    echo "==> strip matrix-language labels Gemma prefixes onto each sentence"
    $PY -m csasr.llm.fix_sentences --in manifests/sentences.jsonl \
        --bigrams manifests/bigrams_valid.jsonl --out manifests/sentences.jsonl

    echo "==> GATE 1.5 -- label leaks, script mix, switch density, domain overlap"
    $PY scripts/audit_sentences.py \
        --sentences manifests/sentences.jsonl --mucs manifests/mucs_train.jsonl

    echo "==> push text to the Hub"
    $PY -m csasr.data.push_to_hub --manifest manifests/bigrams_valid.jsonl \
        --repo "$SYNTH_REPO" --config bigrams --text-only
    $PY -m csasr.data.push_to_hub --manifest manifests/sentences.jsonl \
        --repo "$SYNTH_REPO" --config sentences --text-only
    echo "[stage 1] done $(date)"
fi

# =============================================================================
# Stage 2: TTS audio synthesis (GPU, shards spread across both cards) --
# GATE 2 / GATE 2b
# =============================================================================
if stage_selected 2; then
    echo ""
    echo "##################### STAGE 2: TTS audio #####################"
    # Cell order matters: the transformers==4.46.1 downgrade silently no-ops if
    # anything in this process already imported transformers. Nothing above
    # this point in stage 2 does.
    $PY -m pip install -e ".[tts]"
    $PY -m pip install git+https://github.com/huggingface/parler-tts.git
    $PY -c "
import csasr, transformers
assert transformers.__version__ == '4.46.1', (
    f'transformers is {transformers.__version__}, expected 4.46.1 -- something '
    'imported it before the downgrade took effect.'
)
print('csasr', csasr.__version__, '| transformers', transformers.__version__)
"

    TTS_MODEL=ai4bharat/indic-parler-tts
    NUM_SHARDS=4
    BATCH=16   # Parler is small (938M); 8 left the GPU underfed

    echo "==> PREFLIGHT -- Indic Parler-TTS is gated"
    $PY - <<PY
import os
from huggingface_hub import model_info
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
try:
    model_info("$TTS_MODEL", token=os.environ["HF_TOKEN"])
    print("OK", "$TTS_MODEL")
except GatedRepoError:
    raise SystemExit(
        "PREFLIGHT FAILED: $TTS_MODEL is gated.\n"
        "  Accept the licence at https://huggingface.co/$TTS_MODEL\n"
        "  (fine-grained tokens also need 'Read access to contents of all public "
        "gated repos you can access')."
    )
except RepositoryNotFoundError:
    raise SystemExit("PREFLIGHT FAILED: $TTS_MODEL not found, or your token cannot see it.")
PY

    if [[ ! -f manifests/sentences.jsonl ]]; then
        echo "==> pulling sentences from $SYNTH_REPO"
        $PY - <<PY
import os
from datasets import load_dataset
from csasr.manifest import write_jsonl
ds = load_dataset("$SYNTH_REPO", "sentences", split="train", token=os.environ["HF_TOKEN"])
write_jsonl("manifests/sentences.jsonl", [dict(r) for r in ds])
print(f"{len(ds):,} sentences to synthesize")
PY
    fi

    echo "==> synthesize -- $N_GPU shards at a time, one per GPU, resumable"
    for ((start=0; start<NUM_SHARDS; start+=N_GPU)); do
        pids=()
        for ((s=start; s<start+N_GPU && s<NUM_SHARDS; s++)); do
            gpu=$(( (s - start) % N_GPU ))
            echo "  GPU$gpu: shard $s/$NUM_SHARDS"
            CUDA_VISIBLE_DEVICES=$gpu $PY -m csasr.tts.synthesize \
                --sentences manifests/sentences.jsonl --audio-dir data/synth/audio \
                --out "manifests/train_t2.shard${s}.jsonl" \
                --shard "$s" --num-shards "$NUM_SHARDS" --batch-size "$BATCH" &
            pids+=($!)
        done
        for pid in "${pids[@]}"; do wait "$pid"; done
    done

    echo "==> merge shards, build Train_T1 by reference -- GATE 2"
    $PY - <<PY
from csasr.manifest import read_jsonl, write_jsonl
merged = []
for i in range($NUM_SHARDS):
    merged.extend(read_jsonl(f"manifests/train_t2.shard{i}.jsonl"))
write_jsonl("manifests/train_t2.jsonl", merged)
hours = sum(r["dur"] for r in merged) / 3600
print(f"Train_T2: {len(merged):,} clips, {hours:.2f} h (paper: 22 h)")
PY
    $PY -m csasr.tts.make_subset --t2 manifests/train_t2.jsonl \
        --out manifests/t1_ids.json --hours 8.0
    $PY - <<PY
import json
from csasr.manifest import read_jsonl
t2 = {r["utt_id"]: r for r in read_jsonl("manifests/train_t2.jsonl")}
t1_ids = set(json.load(open("manifests/t1_ids.json")))
assert t1_ids <= t2.keys(), "Train_T1 is NOT a subset of Train_T2"
t1_hours = sum(t2[i]["dur"] for i in t1_ids) / 3600
print(f"Train_T1: {len(t1_ids):,} clips, {t1_hours:.2f} h (paper: 8 h) -- subset OK")
PY

    echo "==> push synthetic corpus to the Hub"
    $PY -m csasr.data.push_to_hub --manifest manifests/train_t2.jsonl \
        --repo "$SYNTH_REPO" --config synth_t2 --verify
    $PY - <<PY
import os
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj="manifests/t1_ids.json",
                    path_in_repo="t1_ids.json", repo_id="$SYNTH_REPO",
                    repo_type="dataset", token=os.environ["HF_TOKEN"])
PY

    echo "==> GATE 2b -- t1_ids resolve against the PUBLISHED parquet"
    $PY scripts/verify_gate2.py --repo "$SYNTH_REPO" --config synth_t2
    echo "[stage 2] done $(date)"
fi

# =============================================================================
# Stage 3: train M6 -> M7 -> M8 (single GPU -- see the CUDA_VISIBLE_DEVICES
# note in the header comment)
# =============================================================================
if stage_selected 3; then
    echo ""
    echo "##################### STAGE 3: train #####################"
    export CUDA_VISIBLE_DEVICES=0
    $PY -m pip install -e ".[train]"
    $PY -c "import csasr; print('csasr', csasr.__version__)"

    T1_IDS=$($PY - <<'PY'
import os
from huggingface_hub import hf_hub_download
print(hf_hub_download("RohanRamesh/hi-en-synth-cs", "t1_ids.json",
                      repo_type="dataset", token=os.environ["HF_TOKEN"]))
PY
)
    echo "t1_ids: $T1_IDS"

    avail_gb=$(df -BG --output=avail "$WORKDIR" | tail -1 | tr -dc '0-9')
    echo "[disk] $WORKDIR has ${avail_gb} GB free"
    if (( avail_gb < 25 )); then
        echo "!! Only ${avail_gb} GB free; M7 needs ~40 GB, M8 ~60 GB." >&2
        exit 1
    fi

    free_map_cache () {
        local before after
        before=$(du -sk "$HF_HOME" 2>/dev/null | cut -f1)
        find "$HF_HOME" -name 'cache-*.arrow' -delete 2>/dev/null || true
        after=$(du -sk "$HF_HOME" 2>/dev/null | cut -f1)
        echo "[cache] freed $(( (before - after) / 1024 )) MB of featurized Arrow"
    }

    run_model () {
        local m=$1; shift
        echo ""
        echo "===================== $m ====================="
        date
        free_map_cache
        $PY -m csasr.train.train_whisper \
            --config "configs/train_${m}_large.yaml" \
            --out "$OUT_ROOT/${m}_large" \
            --gradient-checkpointing on \
            "$@"
        df -h "$WORKDIR" | tail -1
    }

    # See train_a100.sh for why M7/M8 chain from the prior stage's checkpoint
    # instead of independently full-fine-tuning openai/whisper-large-v2.
    require_prev_stage () {
        local prev_out=$1 stage=$2
        if [[ ! -d "$prev_out" ]]; then
            echo "!! $stage is chained from $prev_out but that directory does not exist." >&2
            exit 1
        fi
    }

    for m in $MODELS; do
        case "$m" in
            m6) run_model m6 --subset-ids "$T1_IDS" ;;
            m7) require_prev_stage "$OUT_ROOT/m6_large" m7
                run_model m7 --model "$OUT_ROOT/m6_large" ;;
            m8) require_prev_stage "$OUT_ROOT/m7_large" m8
                run_model m8 --model "$OUT_ROOT/m7_large" || echo "!! m8 failed (cv_en not on the Hub yet?)" ;;
            *)  echo "unknown model '$m'" >&2; exit 2 ;;
        esac
    done
    echo "[stage 3] done $(date)"
    ls -la "$OUT_ROOT"
fi

# =============================================================================
# Stage 4: decode + score against the paper's Table 2 (single GPU)
# =============================================================================
if stage_selected 4; then
    echo ""
    echo "##################### STAGE 4: eval #####################"
    export CUDA_VISIBLE_DEVICES=0
    $PY -m pip install -e ".[decode]"

    REFS=$RESULTS/refs_test.jsonl
    TABLE=$RESULTS/table2_large.jsonl

    decode () {
        local model=$1 tag=$2
        echo ""
        echo "---- decode $tag ($model) ----"
        $PY -m csasr.eval.decode \
            --model "$model" \
            --test-hf "$REAL_REPO" --test-config test \
            --out "$RESULTS/hyp_${tag}.jsonl" \
            --refs-out "$REFS" \
            --language none --batch-size 16 --lang-detect-segments 8
    }
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
            echo "!! $ckpt missing -- skipping $m (was stage 3 run?)"
            continue
        fi
        decode "$ckpt" "$m"
        score "$m"
    done

    echo ""
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
    echo "[stage 4] done $(date)"
fi

echo ""
echo "== pipeline done $(date) =="
