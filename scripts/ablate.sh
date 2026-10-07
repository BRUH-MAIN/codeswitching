#!/bin/bash
# The ablation study as ONE resumable sbatch job: zero-shot baselines, the
# ASR-size x fine-tuning-method grid, MMS-1B, text generation with three new
# LLMs (plus the domain-matched prompt), TTS, the LLM-axis training runs, and
# the report.
#
#   cd /path/to/codeswitching                 # sbatch FROM THE REPO ROOT
#   sbatch scripts/ablate.sh                  # everything that is not done yet
#   sbatch --export=ALL,STAGES="zeroshot grid mms llm report" scripts/ablate.sh
#   sbatch --export=ALL,STAGES="text tts llm report"          scripts/ablate.sh
#   RUNS="g26__whisper-small__lora g26__whisper-small__qlora" sbatch scripts/ablate.sh
#   SMOKE=1 sbatch --time=02:00:00 scripts/ablate.sh          # tiny end-to-end check
#
# (Variables set in front of `sbatch` reach the job because sbatch exports the
# submitting environment by default; --export=ALL,NAME=value is the explicit form.)
#
# GPU TIME. --time below is the partition's MaxTime (3 days); it cannot be set
# higher on this cluster. Two things stretch it further:
#
#   * Submit the script SEVERAL TIMES. Every unit of work (one training run, one
#     LLM's text, one corpus's audio) takes a lock and leaves a .done marker, so
#     concurrent jobs split the work across however many A100s the scheduler
#     grants instead of duplicating it. QOS `normal` has no per-user job cap.
#   * At 10 minutes before the walltime cap SLURM sends USR1; the job releases
#     its locks, RESUBMITS ITSELF (up to MAX_RESUBMITS times, default 2 -- so
#     up to 9 days of GPU time per submission) and exits. Training resumes from
#     its last checkpoint, LLM calls from the response cache, TTS from the clips
#     already on disk.
#
# Run names are <corpus>__<asr>__<ft>[__<tag>]:
#   corpus  g26 (existing Gemma-4-26B-A4B corpus, on the Hub) | g31 | s30 | s105
#           | <llm>m (that LLM's bigrams, domain-matched sentence prompt)
#   asr     whisper-small | whisper-medium | whisper-large-v2 | mms-1b
#   ft      full | lora | qlora
#   tag     lr<value> or seed<n>, optional
#
# STAGES (default: all, in this order):
#   zeroshot  decode-only rows for all four ASR models (also step 0 of dev curves)
#   grid      3 Whisper sizes x 3 methods on g26, plus one LoRA learning-rate check
#   mms       MMS-1B full + LoRA on g26
#   text      LLM text for g31, s30 and the matched-prompt corpus (s105 is opt-in via LLMS)
#   tts       Indic Parler-TTS for every corpus whose text is finished
#   llm       whisper-large-v2 full fine-tune on every corpus whose audio is finished
#   report    tables and figures (CPU)

#SBATCH --job-name=csasr-abl
#SBATCH --partition=workq
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --signal=B:USR1@600
#SBATCH --output=slurm-ablate-%j.out

set -euo pipefail

export USE_TF=0
export USE_FLAX=0
export PYTHONUNBUFFERED=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO=$SLURM_SUBMIT_DIR
else
    REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
if [[ ! -f "$REPO/src/csasr/eval/select.py" ]]; then
    echo "Cannot find src/csasr/eval/select.py under REPO=$REPO." >&2
    echo "  Under sbatch, REPO is \$SLURM_SUBMIT_DIR -- submit from the repo root:" >&2
    echo "      cd /path/to/codeswitching && sbatch scripts/ablate.sh" >&2
    exit 1
fi
cd "$REPO"

# ---- knobs (all overridable via --export=ALL,NAME=value) --------------------
: "${RUNS:=}"                       # explicit run names, run before any stage
if [[ -n "$RUNS" ]]; then
    : "${STAGES:=}"                 # RUNS alone means "just these"
else
    : "${STAGES:=zeroshot grid mms text tts llm report}"
fi
: "${SMOKE:=0}"                     # 1 = tiny end-to-end check into separate dirs
: "${WORKDIR:=$REPO/csasr-work}"
: "${HF_HOME:=$WORKDIR/hf}"
: "${REAL:=RohanRamesh/mucs-he-cs}"
: "${SYNTH:=RohanRamesh/hi-en-synth-cs}"
: "${HOURS:=22}"                    # every corpus is cut to this many hours
: "${MAX_STEPS:=1200}"
: "${SAVE_STEPS:=200}"
: "${WARMUP:=200}"
: "${FRACTION:=1.0}"
: "${NUM_PROC:=4}"
: "${LLMS:=g31 s30}"                # s105 dropped 2026-10-06 (failed its bigram smoke gate); LLMS="s105" still works
: "${MATCHED_LLM:=g31}"             # whose bigrams the matched-prompt corpus uses
: "${MMS_QLORA:=0}"
: "${NP:=8}"                        # llama-server parallel slots
: "${CTX:=4096}"                    # context per slot
: "${TEXT_TARGET_H:=24}"            # stop generating text at this many projected hours
: "${ROUNDS:=500 1000 2000 4000 8000 15000}"   # cumulative bigram calls per round
: "${TTS_SHARDS:=4}"
: "${TTS_BATCH:=16}"
: "${TTS_MAX_H:=25}"                # do not synthesize more than this (estimated)
: "${MAX_RESUBMITS:=2}"
: "${RESUBMIT_COUNT:=0}"
: "${ABL_VENV:=$HOME/.venvs/csasr-abl}"
: "${TTS_VENV:=$HOME/.venvs/csasr-tts}"
: "${BOOTSTRAP_PY:=/dist_home/common-apps/conda/bin/python3}"
: "${CUDA_ROOT:=/dist_home/common-apps/cuda/12.5}"

if [[ "$SMOKE" == "1" ]]; then
    # Separate directories throughout: a smoke run must never leave a .done
    # marker where the real study would then skip the work.
    EXP=$WORKDIR/abl-smoke
    RESULTS_DIR=$REPO/results/ablation-smoke
    DATA=$REPO/data/ablation-smoke
    MAX_STEPS=20; SAVE_STEPS=10; WARMUP=2; FRACTION=0.02
    ROUNDS="4"
    SELECT_LIMIT=(--limit 200)
    MAX_RESUBMITS=0
else
    EXP=$WORKDIR/abl
    RESULTS_DIR=$REPO/results/ablation
    DATA=$REPO/data/ablation
    SELECT_LIMIT=()
fi
LOGS=$WORKDIR/logs
LLAMA_DIR=$WORKDIR/llama.cpp
LLAMA_SERVER=$LLAMA_DIR/build/bin/llama-server
export HF_HOME STAGES RUNS SMOKE WORKDIR REAL SYNTH HOURS LLMS MATCHED_LLM MMS_QLORA \
       NP CTX TEXT_TARGET_H ROUNDS TTS_SHARDS TTS_BATCH TTS_MAX_H MAX_RESUBMITS \
       ABL_VENV TTS_VENV NUM_PROC
mkdir -p "$HF_HOME" "$EXP" "$RESULTS_DIR" "$DATA" "$LOGS" manifests

JOB=${SLURM_JOB_ID:-local$$}
PY=$ABL_VENV/bin/python3
TTS_PY=$TTS_VENV/bin/python3

: "${HF_ENV_FILE:=$(dirname "$REPO")/.env}"
if [[ -z "${HF_TOKEN:-}" && -f "$HF_ENV_FILE" ]]; then
    HF_TOKEN=$(grep -m1 -E '^[[:space:]]*HF_TOKEN[[:space:]]*=' "$HF_ENV_FILE" \
        | sed -E 's/^[[:space:]]*HF_TOKEN[[:space:]]*=[[:space:]]*//; s/[[:space:]]*$//; s/^"(.*)"$/\1/')
    [[ -n "$HF_TOKEN" ]] && export HF_TOKEN
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is unset and not found in $HF_ENV_FILE (needed for the private" >&2
    echo "MUCS dev/test splits and the gated TTS model). Add:  HF_TOKEN = <token>" >&2
    exit 1
fi

# =============================================================================
# Plumbing: interruptible commands, locks, walltime hand-over
# =============================================================================
SERVER_PID=""
NP_ACTIVE=1
LOCKS=()
FAILED=()

# Run in the background and wait, so the USR1 trap fires DURING a long command
# instead of after it (bash defers traps until a foreground child returns).
# `<&0` matters: without job control bash points a background command's stdin
# at /dev/null, which would silently feed the `python - <<PY` blocks nothing.
run () { "$@" <&0 & wait $!; }

try_lock () {
    local d=$1/.lock owner
    mkdir -p "$1"
    if ! mkdir "$d" 2>/dev/null; then
        owner=$(cat "$d/owner" 2>/dev/null || true)
        [[ -z "$owner" ]] && return 1                     # being created right now
        if [[ "$owner" != "$JOB" ]] && squeue -h -j "$owner" 2>/dev/null | grep -q .; then
            return 1                                      # a live job holds it
        fi
        rm -rf "$d"                                       # its owner is gone
        mkdir "$d" 2>/dev/null || return 1
    fi
    echo "$JOB" > "$d/owner"
    LOCKS+=("$d")
}

unlock () { rm -rf "$1/.lock"; }

release_locks () {
    local d
    for d in "${LOCKS[@]:-}"; do
        [[ -n "$d" && "$(cat "$d/owner" 2>/dev/null || true)" == "$JOB" ]] && rm -rf "$d"
    done
    LOCKS=()
}

stop_server () {
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}

on_walltime () {
    echo ""
    echo "[walltime] USR1 at $(date): the walltime cap is 10 minutes away"
    # Stop the work BEFORE handing over, so the continuation never resumes from
    # a checkpoint directory this job is still writing into.
    pkill -TERM -P $$ 2>/dev/null || true
    sleep 15
    pkill -KILL -P $$ 2>/dev/null || true
    SERVER_PID=""
    release_locks
    if (( RESUBMIT_COUNT < MAX_RESUBMITS )); then
        export RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1))
        echo "[walltime] resubmitting (continuation $RESUBMIT_COUNT of $MAX_RESUBMITS)"
        sbatch --export=ALL "$REPO/scripts/ablate.sh" || echo "!! resubmit failed"
    else
        echo "[walltime] MAX_RESUBMITS=$MAX_RESUBMITS reached -- not resubmitting."
        echo "           Unfinished work continues with: sbatch scripts/ablate.sh"
    fi
    exit 0
}
trap on_walltime USR1
trap 'stop_server; release_locks' EXIT

# unit <dir> <label> <function> [args...]: run once, ever, by one job at a time.
unit () {
    local dir=$1 label=$2; shift 2
    if [[ -f "$dir/.done" ]]; then echo "[skip] $label -- done"; return 0; fi
    if ! try_lock "$dir"; then echo "[skip] $label -- another job is on it"; return 0; fi
    echo ""
    echo "======================== $label ========================  $(date)"
    if "$@"; then
        touch "$dir/.done"
        echo "[done] $label  $(date)"
    else
        echo "!! [FAILED] $label  $(date)"
        FAILED+=("$label")
    fi
    unlock "$dir"
}

stage_selected () { [[ " $STAGES " == *" $1 "* ]]; }

echo "== $(date) on $(hostname -s)  job $JOB  (continuation $RESUBMIT_COUNT) =="
echo "[repo] $REPO   [exp] $EXP   [results] $RESULTS_DIR"
echo "[stages] $STAGES   ${RUNS:+[runs] $RUNS}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv || true

# =============================================================================
# Environments
# =============================================================================
# peft/bitsandbytes go into a venv layered over the shared conda python, not
# into the shared base itself: other users' jobs import from that base.
setup_abl_venv () {
    if [[ ! -x "$PY" ]]; then
        "$BOOTSTRAP_PY" -m venv --system-site-packages "$ABL_VENV" || return 1
    fi
    # torch is pinned to the base's version ON PURPOSE. Left free, the resolver
    # takes the newest bitsandbytes, which drags a newer torch (and a CUDA 13
    # runtime) into the venv, shadowing the base's working torch+cu121.
    local torch_v
    torch_v=$("$BOOTSTRAP_PY" -c "import torch; print(torch.__version__.split('+')[0])") || return 1
    "$PY" -c "import peft, bitsandbytes" 2>/dev/null \
        || "$PY" -m pip install "peft>=0.13,<0.18" "bitsandbytes>=0.43" "torch==$torch_v" \
        || return 1
    "$PY" -c "import torch, sys; sys.exit(torch.__version__.split('+')[0] != '$torch_v')" \
        || { echo "!! $ABL_VENV shadows the base torch $torch_v -- delete it and resubmit" >&2
             return 1; }
    "$PY" -c "import csasr.eval.select, faster_whisper, ctranslate2" 2>/dev/null \
        || "$PY" -m pip install -e ".[train,decode]" || return 1
    "$PY" -c "import csasr, peft, transformers, torch
print('csasr', csasr.__version__, '| peft', peft.__version__,
      '| transformers', transformers.__version__, '| torch', torch.__version__)"
}

# parler-tts hard-pins transformers==4.46.1, which cannot share an environment
# with training (>=4.49). Its own venv, shadowing the base's transformers.
setup_tts_venv () {
    if [[ ! -x "$TTS_PY" ]]; then
        "$BOOTSTRAP_PY" -m venv --system-site-packages "$TTS_VENV" || return 1
    fi
    "$TTS_PY" -c "import parler_tts, transformers; assert transformers.__version__ == '4.46.1'" \
        2>/dev/null && return 0
    local torch_v
    torch_v=$("$BOOTSTRAP_PY" -c "import torch; print(torch.__version__.split('+')[0])") || return 1
    # torch AND torchaudio pinned to the base's torch: parler's audio codec pulls
    # in torchaudio, and an unpinned one arrives built for a newer torch/CUDA
    # and dies at import with "libcudart.so.13: cannot open shared object file".
    "$TTS_PY" -m pip install "transformers==4.46.1" \
        "git+https://github.com/huggingface/parler-tts.git" \
        "torch==$torch_v" "torchaudio==$torch_v" || return 1
    "$TTS_PY" -c "import csasr" 2>/dev/null || "$TTS_PY" -m pip install -e . --no-deps || return 1
    "$TTS_PY" -c "import parler_tts, transformers
assert transformers.__version__ == '4.46.1', transformers.__version__
print('tts venv: transformers', transformers.__version__)"
}

build_llama () {
    [[ -x "$LLAMA_SERVER" ]] && return 0
    [[ -d "$LLAMA_DIR" ]] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR" || return 1
    # The shared toolkit, with its lib64 baked into the rpath, so the binary
    # finds libcublas on whichever node the job lands on.
    #
    # CUDA architectures are listed EXPLICITLY. The default compiles kernels
    # only for the GPU of the build machine, and a binary built on the login
    # node (RTX 6000 Ada, sm89) aborts on an A100 (sm80) with "no kernel image
    # is available for execution on the device". GGML_NATIVE=OFF for the same
    # reason on the CPU side: the nodes do not share a CPU model.
    ( cd "$LLAMA_DIR" && PATH=$CUDA_ROOT/bin:$PATH cmake -B build -DGGML_CUDA=ON \
          -DCMAKE_CUDA_ARCHITECTURES="80;89" -DGGML_NATIVE=OFF \
          -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
          -DCUDAToolkit_ROOT="$CUDA_ROOT" -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
          -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release \
          -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
          -DCMAKE_BUILD_RPATH="$CUDA_ROOT/lib64" \
      && cmake --build build -j "${SLURM_CPUS_PER_TASK:-8}" --target llama-server ) || return 1
    [[ -x "$LLAMA_SERVER" ]]
}

mkdir -p "$WORKDIR/locks"
unit "$WORKDIR/locks/abl-venv-$(hostname -s)" "environment (train/eval venv)" setup_abl_venv
[[ -x "$PY" ]] || { echo "!! no usable python at $PY" >&2; exit 1; }
# The .done marker is per host only so a venv broken by hand gets re-checked on
# the next node; the check itself is cheap.
"$PY" -c "import peft, bitsandbytes, csasr.eval.select" \
    || { echo "!! $ABL_VENV is incomplete -- delete it and resubmit" >&2; exit 1; }

# =============================================================================
# Data helpers
# =============================================================================
# The existing corpus is 23.95 h; every corpus in the study is cut to $HOURS so
# the LLM axis compares text sources, not amounts of audio.
prepare_g26 () {
    mkdir -p "$DATA/g26"
    run "$PY" - "$SYNTH" "$DATA/g26/ids.json" "$HOURS" <<'PY' || return 1
import json, os, sys
from datasets import load_dataset
from csasr.tts.make_subset import select_subset
repo, out, hours = sys.argv[1], sys.argv[2], float(sys.argv[3])
ds = load_dataset(repo, "synth_t2", split="train", token=os.environ["HF_TOKEN"])
rows = ds.select_columns(["utt_id", "dur", "bigram"]).to_list()
ids = select_subset(rows, hours)
by = {r["utt_id"]: r for r in rows}
json.dump(sorted(ids), open(out, "w"))
print(f"g26: {len(ids):,} of {len(rows):,} clips, "
      f"{sum(by[i]['dur'] for i in ids)/3600:.2f} h, "
      f"{len({by[i]['bigram'] for i in ids}):,} bigrams")
PY
}

pull_mucs_text () {
    [[ -s manifests/mucs_train.jsonl ]] && return 0
    run "$PY" - "$REAL" <<'PY' || return 1
import os, sys
from datasets import load_dataset
from csasr.manifest import write_jsonl
ds = load_dataset(sys.argv[1], "train_text", split="train", token=os.environ["HF_TOKEN"])
write_jsonl("manifests/mucs_train.jsonl", [dict(r) for r in ds])
print(f"{len(ds):,} in-domain sentences for few-shot prompting")
PY
}

corpus_ready () {
    if [[ "$1" == g26 ]]; then [[ -f "$DATA/g26/.done" ]]; else [[ -f "$DATA/$1/tts/.done" ]]; fi
}

# =============================================================================
# Training runs
# =============================================================================
parse_name () {   # sets corpus asr ft tag
    local rest
    corpus=${1%%__*}; rest=${1#*__}
    asr=${rest%%__*}; rest=${rest#*__}
    ft=${rest%%__*}; tag=""
    [[ "$rest" == *__* ]] && tag=${rest#*__}
    return 0
}

train_run () {
    local name=$1 corpus asr ft tag
    parse_name "$name"
    local out=$EXP/$name lr seed=1234
    local -a data model

    if [[ "$corpus" == g26 ]]; then
        data=(--train-hf "$SYNTH" --train-config synth_t2 --subset-ids "$DATA/g26/ids.json")
    else
        data=(--train-manifest "$DATA/$corpus/train.jsonl")
    fi
    [[ "$ft" == full ]] && lr=2e-5 || lr=1e-3      # paper recipe / PEFT Whisper recipe
    [[ "$asr" == mms-1b && "$ft" == full ]] && lr=5e-5
    case "$tag" in
        lr*)   lr=${tag#lr} ;;
        seed*) seed=${tag#seed} ;;
        "")    ;;
        *)     echo "unknown tag '$tag' in $name" >&2; return 1 ;;
    esac

    if [[ "$asr" == mms-1b ]]; then
        run "$PY" -m csasr.train.train_ctc "${data[@]}" \
            --model facebook/mms-1b-all --source-lang hin --out "$out" \
            --dev-hf "$REAL" --dev-config dev --dev-max-utts 300 \
            --ft-method "$ft" --lr "$lr" --batch-size 16 --grad-accum 4 \
            --max-steps "$MAX_STEPS" --save-steps "$SAVE_STEPS" --warmup-steps "$WARMUP" \
            --seed "$seed" --dataset-fraction "$FRACTION" || return 1
        run "$PY" -m csasr.eval.select --engine ctc --run-dir "$out" --name "$name" \
            --zero-shot-dir "$EXP/zeroshot__$asr" --results-dir "$RESULTS_DIR" \
            --real-hf "$REAL" "${SELECT_LIMIT[@]}" --prune \
            --meta "corpus=$corpus" "asr=$asr" "ft=$ft" "tag=$tag" || return 1
        return 0
    fi

    # Effective batch 64 everywhere (the paper's). The per-step split is what
    # fits, MEASURED on the 80 GB A100 (smoke job 12508): small at batch 64
    # without checkpointing peaks at 64 GB; medium at batch 32 without it runs
    # out of memory; large-v2 at batch 16 with it peaks at 34 GB (full), 20 GB
    # (LoRA), 11 GB (QLoRA). Checkpointing changes speed and memory, not results.
    case "$asr" in
        # Not part of the study: small enough to exercise the whole chain on a CPU.
        whisper-tiny)     model=(--model openai/whisper-tiny --batch-size 16 --grad-accum 4
                                 --gradient-checkpointing off) ;;
        whisper-small)    model=(--model openai/whisper-small --batch-size 64 --grad-accum 1
                                 --gradient-checkpointing off) ;;
        whisper-medium)   model=(--model openai/whisper-medium --batch-size 32 --grad-accum 2
                                 --gradient-checkpointing on) ;;
        whisper-large-v2) model=(--model openai/whisper-large-v2 --batch-size 16 --grad-accum 4
                                 --gradient-checkpointing on) ;;
        *) echo "unknown asr '$asr' in $name" >&2; return 1 ;;
    esac

    run "$PY" -m csasr.train.train_whisper "${data[@]}" "${model[@]}" --out "$out" \
        --dev-hf "$REAL" --dev-config dev --language hi \
        --ft-method "$ft" --lr "$lr" \
        --max-steps "$MAX_STEPS" --save-steps "$SAVE_STEPS" --eval-steps "$SAVE_STEPS" \
        --warmup-steps "$WARMUP" --generate-eval off --dev-max-utts 300 \
        --eval-batch-size 16 --save-total-limit 0 --seed "$seed" \
        --num-proc "$NUM_PROC" --dataloader-workers 4 --report-to none \
        --dataset-fraction "$FRACTION" || return 1
    run "$PY" -m csasr.eval.select --run-dir "$out" --name "$name" \
        --zero-shot-dir "$EXP/zeroshot__$asr" --results-dir "$RESULTS_DIR" \
        --real-hf "$REAL" "${SELECT_LIMIT[@]}" --prune \
        --meta "corpus=$corpus" "asr=$asr" "ft=$ft" "tag=$tag" || return 1
}

zeroshot_run () {
    local asr=$1 name=zeroshot__$1
    local -a model
    case "$asr" in
        whisper-*) model=(--zero-shot "openai/$asr") ;;
        mms-1b)    model=(--engine ctc --zero-shot facebook/mms-1b-all --target-lang hin) ;;
    esac
    run "$PY" -m csasr.eval.select "${model[@]}" --run-dir "$EXP/$name" --name "$name" \
        --results-dir "$RESULTS_DIR" --real-hf "$REAL" "${SELECT_LIMIT[@]}" \
        --meta corpus=none "asr=$asr" ft=zeroshot tag= || return 1
}

do_runs () {
    local name corpus asr ft tag
    for name in "$@"; do
        parse_name "$name"
        if ! corpus_ready "$corpus"; then
            echo "[skip] $name -- corpus '$corpus' has no audio yet"
            continue
        fi
        unit "$EXP/$name" "run $name" train_run "$name"
    done
}

# =============================================================================
# LLM text generation
# =============================================================================
declare -A GGUF_REPO=(
    [g26]=unsloth/gemma-4-26B-A4B-it-GGUF   [g31]=unsloth/gemma-4-31B-it-GGUF
    [s30]=sarvamai/sarvam-30b-gguf          [s105]=sarvamai/sarvam-105b-gguf
    [tiny]=unsloth/gemma-4-E2B-it-GGUF )    # 3 GB: smoke-tests the text path, not in the study
declare -A GGUF_GLOB=(
    [g26]="gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"   [g31]="gemma-4-31B-it-Q4_K_M.gguf"
    [s30]="sarvam-30b-Q4_K_M.gguf-*"            [s105]="sarvam-105b-Q4_K_M.gguf-*"
    [tiny]="gemma-4-E2B-it-Q4_K_M.gguf" )

start_server () {   # start_server <llm key>
    local llm=$1 gguf log=$LOGS/llama-server-$1-$JOB.log
    stop_server
    gguf=$("$PY" - "${GGUF_REPO[$llm]}" "${GGUF_GLOB[$llm]}" "$WORKDIR/gguf/$llm" <<'PY'
import os, sys, pathlib
from huggingface_hub import snapshot_download
repo, glob, out = sys.argv[1:4]
d = snapshot_download(repo, allow_patterns=[glob], local_dir=out, token=os.environ.get("HF_TOKEN"))
print(sorted(str(p) for p in pathlib.Path(d).glob(glob))[0])   # first shard
PY
    ) || return 1
    # 64 GB of Sarvam-105B leaves ~14 GB of an 80 GB card for the KV cache, and
    # whether 8 slots of context fit there is not something to guess: if the
    # server dies while loading, halve the slots and try again.
    local np i
    for ((np = NP; np >= 1; np /= 2)); do
        PORT=$(( 18000 + RANDOM % 2000 ))
        echo "[llm] serving $gguf on :$PORT with $np slot(s) (log: $log)"
        # -t: without it llama.cpp starts one thread per core of the NODE, not
        # of the allocation.
        "$LLAMA_SERVER" -m "$gguf" -ngl 999 -c $((CTX * np)) -np "$np" \
            -t "${SLURM_CPUS_PER_TASK:-8}" \
            --host 127.0.0.1 --port "$PORT" --alias "$llm" --jinja --reasoning-budget 0 \
            > "$log" 2>&1 &
        SERVER_PID=$!
        for ((i = 0; i < 480; i++)); do      # up to 40 min: 64 GB of GGUF over NFS
            if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
                NP_ACTIVE=$np
                return 0
            fi
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 5
        done
        echo "!! llama-server did not come up with $np slot(s):" >&2; tail -15 "$log" >&2
        stop_server
    done
    return 1
}

llm_cli () {   # llm_cli <llm key> <module> [args...]
    local llm=$1 mod=$2; shift 2
    run "$PY" -m "csasr.llm.$mod" --backend openai --model "$llm-Q4_K_M" \
        --base-url "http://127.0.0.1:$PORT/v1" --concurrency "$NP_ACTIVE" "$@"
}

# Projected audio for a sentences file: ~2.4 words/s, less the 4.6% that
# projection was measured to overshoot by on the existing corpus.
est_hours () {
    "$PY" - "$1" <<'PY'
import sys
from csasr.manifest import read_jsonl
print(round(sum(len(r["text"].split()) for r in read_jsonl(sys.argv[1])) / 2.4 * 0.954 / 3600, 1))
PY
}

gen_text () {
    local llm=$1 d=$DATA/$1 n calls=0 est=0
    mkdir -p "$d/cache"
    start_server "$llm" || return 1

    llm_cli "$llm" smoke --train-manifest manifests/mucs_train.jsonl \
        --n-calls 4 --bigrams-per-call 10 || { stop_server; return 1; }

    # Same prompts in the same order for every LLM (seeded, prefix-stable), in
    # doubling rounds, until the sentences that SURVIVE validation and the
    # speech filter add up to enough audio. The stopping rule is on usable hours
    # and not on bigram count because the two come apart badly across models:
    # Gemma-4-31B gave 11,000 switch pairs from 2,500 calls, while Sarvam-30B
    # gave 7,200 pairs that turned into only ~12 h of clean sentences. How many
    # calls a model needs is itself reported in the text table. Earlier rounds
    # are free on a rerun: every call is cached.
    for n in $ROUNDS; do
        calls=$n
        llm_cli "$llm" gen_bigrams --train-manifest manifests/mucs_train.jsonl \
            --out "$d/bigrams_raw.jsonl" --cache "$d/cache/bigrams.jsonl" \
            --n-calls "$n" --batch-size 64 || { stop_server; return 1; }
        llm_cli "$llm" filter_bigrams --raw "$d/bigrams_raw.jsonl" \
            --out "$d/bigrams_valid.jsonl" --cache "$d/cache/transcheck.jsonl" \
            --items-per-call 20 --n-samples 3 --batch-size 32 || { stop_server; return 1; }
        llm_cli "$llm" gen_sentences --bigrams "$d/bigrams_valid.jsonl" \
            --out "$d/sentences.jsonl" --cache "$d/cache/sentences.jsonl" \
            --batch-size 64 || { stop_server; return 1; }
        est=$(est_hours "$d/sentences.jsonl") || { stop_server; return 1; }
        echo "[text] $llm: $n calls -> $(wc -l < "$d/bigrams_valid.jsonl") valid bigrams," \
             "$(wc -l < "$d/sentences.jsonl") clean sentences, ~$est h (target $TEXT_TARGET_H h)"
        awk "BEGIN { exit !($est >= $TEXT_TARGET_H) }" && break
    done
    stop_server
    echo "$calls" > "$d/n_calls.txt"
    run "$PY" -m csasr.llm.fix_sentences --in "$d/sentences.jsonl" \
        --bigrams "$d/bigrams_valid.jsonl" --out "$d/sentences.jsonl" || return 1
    # The audit exits 1 on distribution "blockers"; here it is information.
    "$PY" scripts/audit_sentences.py --sentences "$d/sentences.jsonl" \
        --mucs manifests/mucs_train.jsonl --sample 8 || true
}

gen_text_matched () {
    local llm=$1 d=$DATA/${1}m
    mkdir -p "$d/cache"
    cp "$DATA/$llm/bigrams_valid.jsonl" "$d/bigrams_valid.jsonl" || return 1
    start_server "$llm" || return 1
    llm_cli "$llm" gen_sentences --bigrams "$d/bigrams_valid.jsonl" --out "$d/sentences.jsonl" \
        --cache "$d/cache/sentences.jsonl" --batch-size 64 \
        --prompt-variant matched --fewshot-manifest manifests/mucs_train.jsonl \
        || { stop_server; return 1; }
    stop_server
    run "$PY" -m csasr.llm.fix_sentences --in "$d/sentences.jsonl" \
        --bigrams "$d/bigrams_valid.jsonl" --out "$d/sentences.jsonl" || return 1
    "$PY" scripts/audit_sentences.py --sentences "$d/sentences.jsonl" \
        --mucs manifests/mucs_train.jsonl --sample 8 || true
}

# =============================================================================
# TTS
# =============================================================================
synth_corpus () {
    local c=$1 d=$DATA/$1 s
    local -a pids=()
    mkdir -p "$d/audio" "$d/tts"

    # Cap what gets synthesized: anything much beyond $HOURS is thrown away by
    # the cut below. Round-robin over bigrams, so the cap keeps coverage.
    run "$PY" - "$d/sentences.jsonl" "$d/tts/sentences_tts.jsonl" "$TTS_MAX_H" <<'PY' || return 1
import sys
from csasr.manifest import read_jsonl, write_jsonl
from csasr.tts.make_subset import select_subset
src, out, max_h = sys.argv[1], sys.argv[2], float(sys.argv[3])
rows = list(read_jsonl(src))
# ~2.4 words/s, less the 4.6% the Gate 1.5 projection was found to overshoot by.
for r in rows:
    r["utt_id"] = r["sent_id"]; r["dur"] = len(r["text"].split()) / 2.4 * 0.954
est = sum(r["dur"] for r in rows) / 3600
keep = set(select_subset(rows, max_h)) if est > max_h else {r["utt_id"] for r in rows}
n = write_jsonl(out, [{k: v for k, v in r.items() if k not in ("utt_id", "dur")}
                      for r in rows if r["utt_id"] in keep])
print(f"{len(rows):,} sentences (~{est:.1f} h estimated) -> synthesizing {n:,}")
PY
    [[ "$SMOKE" == "1" ]] && local -a limit=(--limit 8) || local -a limit=()

    # Parler is small (938M); several shards share the one card comfortably.
    for ((s = 0; s < TTS_SHARDS; s++)); do
        "$TTS_PY" -m csasr.tts.synthesize --sentences "$d/tts/sentences_tts.jsonl" \
            --audio-dir "$d/audio" --out "$d/tts/shard$s.jsonl" \
            --shard "$s" --num-shards "$TTS_SHARDS" --batch-size "$TTS_BATCH" \
            "${limit[@]}" > "$LOGS/tts-$c-shard$s-$JOB.log" 2>&1 &
        pids+=($!)
    done
    for s in "${!pids[@]}"; do
        if ! wait "${pids[$s]}"; then
            echo "!! TTS shard $s failed:" >&2; tail -15 "$LOGS/tts-$c-shard$s-$JOB.log" >&2
            return 1
        fi
        tail -1 "$LOGS/tts-$c-shard$s-$JOB.log"
    done

    run "$PY" - "$d" "$TTS_SHARDS" "$HOURS" <<'PY' || return 1
import sys
from csasr.manifest import read_jsonl, write_jsonl
from csasr.tts.make_subset import select_subset
d, n_shards, hours = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
rows = [r for i in range(n_shards) for r in read_jsonl(f"{d}/tts/shard{i}.jsonl")]
total = sum(r["dur"] for r in rows) / 3600
keep = set(select_subset(rows, hours)) if total > hours else {r["utt_id"] for r in rows}
kept = [r for r in rows if r["utt_id"] in keep]
write_jsonl(f"{d}/train.jsonl", kept)
print(f"synthesized {len(rows):,} clips / {total:.2f} h -> train.jsonl {len(kept):,} clips / "
      f"{sum(r['dur'] for r in kept)/3600:.2f} h, {len({r.get('bigram') for r in kept}):,} bigrams")
if total < hours:
    print(f"NOTE: short of the {hours:g} h target -- report this corpus's hours with its result")
PY
}

# =============================================================================
# Stages
# =============================================================================
unit "$DATA/g26" "data: cut the existing corpus to $HOURS h" prepare_g26

if [[ -n "$RUNS" ]]; then
    for r in $RUNS; do
        case "$r" in
            zeroshot__*) unit "$EXP/$r" "zero-shot ${r#zeroshot__}" zeroshot_run "${r#zeroshot__}" ;;
            *)           do_runs "$r" ;;
        esac
    done
fi

if stage_selected zeroshot; then
    for asr in whisper-large-v2 whisper-small whisper-medium mms-1b; do
        unit "$EXP/zeroshot__$asr" "zero-shot $asr" zeroshot_run "$asr"
    done
fi

if stage_selected grid; then
    # Anchor first, then the cheap rows, so a partial grid is still a story.
    do_runs g26__whisper-large-v2__full g26__whisper-small__full g26__whisper-medium__full \
            g26__whisper-large-v2__lora g26__whisper-large-v2__qlora \
            g26__whisper-small__lora g26__whisper-small__qlora \
            g26__whisper-medium__lora g26__whisper-medium__qlora \
            g26__whisper-small__lora__lr1e-4
fi

if stage_selected mms; then
    do_runs g26__mms-1b__full g26__mms-1b__lora
    [[ "$MMS_QLORA" == "1" ]] && do_runs g26__mms-1b__qlora
fi

if stage_selected text; then
    unit "$WORKDIR/locks/llama-build" "build llama.cpp (CUDA)" build_llama
    unit "$REPO/manifests/.mucs_train" "data: in-domain transcripts for few-shot prompts" pull_mucs_text
    if [[ -x "$LLAMA_SERVER" && -s manifests/mucs_train.jsonl ]]; then
        for llm in $LLMS; do
            unit "$DATA/$llm/text" "text: $llm (paper prompt)" gen_text "$llm"
        done
        if [[ -f "$DATA/$MATCHED_LLM/text/.done" ]]; then
            unit "$DATA/${MATCHED_LLM}m/text" "text: $MATCHED_LLM (matched prompt)" \
                gen_text_matched "$MATCHED_LLM"
        else
            echo "[skip] matched-prompt text -- $MATCHED_LLM bigrams are not finished"
        fi
    else
        echo "!! text stage skipped: llama-server or manifests/mucs_train.jsonl missing"
        FAILED+=("text stage prerequisites")
    fi
fi

NEW_CORPORA="$LLMS ${MATCHED_LLM}m"

if stage_selected tts; then
    unit "$WORKDIR/locks/tts-venv" "environment (TTS venv)" setup_tts_venv
    for c in $NEW_CORPORA; do
        if [[ -f "$DATA/$c/text/.done" ]]; then
            unit "$DATA/$c/tts" "tts: $c" synth_corpus "$c"
        else
            echo "[skip] tts $c -- its text is not finished"
        fi
    done
fi

if stage_selected llm; then
    for c in $NEW_CORPORA; do
        do_runs "${c}__whisper-large-v2__full"
    done
fi

if stage_selected report; then
    echo ""
    echo "======================== report ========================"
    "$PY" scripts/text_stats.py --data "$DATA" --mucs manifests/mucs_train.jsonl \
        --test-refs results/refs_test.jsonl --out "$RESULTS_DIR/text_stats.json" \
        || echo "!! text_stats failed"
    "$PY" scripts/ablation_report.py --results-dir "$RESULTS_DIR" --exp "$EXP" \
        || echo "!! ablation_report failed"
fi

echo ""
echo "== finished $(date) =="
if (( ${#FAILED[@]} )); then
    echo "FAILED units (rerun with: sbatch scripts/ablate.sh -- finished work is skipped):"
    printf '  - %s\n' "${FAILED[@]}"
    exit 1
fi
