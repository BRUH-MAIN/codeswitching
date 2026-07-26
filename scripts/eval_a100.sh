#!/bin/bash
# Decode and score M6 / M7 / M8 on the A100, straight from the local checkpoints
# that train_a100.sh wrote. No notebook and no Hub round-trip needed --
# csasr.eval.decode takes a local directory just as happily as a hub id.
#
#   cd /path/to/codeswitching               # sbatch FROM THE REPO ROOT
#   sbatch scripts/eval_a100.sh
#   sbatch --export=ALL,MODELS="m6 m7" scripts/eval_a100.sh
#
# WORKDIR must match the training run, or the checkpoints will not be found.
#
# The zero-shot large-v2 row is Gate 3, and it doubles as the BASELINE for this
# column: we already measured 51.9 MER / 42.9 CBA-HE against the paper's 52.0.
# On the whisper-small branch a separate small-zero-shot baseline was needed;
# here the paper's own model is the baseline, so that extra decode is gone.
#
#SBATCH --job-name=csasr-eval
#SBATCH --partition=workq
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=asaicomputenode02
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm-eval-%j.out

set -euo pipefail

PY=${PYTHON:-python3}

# sbatch runs a COPY of this script from the spool dir, so BASH_SOURCE does not
# locate the checkout. $SLURM_SUBMIT_DIR does. See train_a100.sh for the detail.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO=$SLURM_SUBMIT_DIR
else
    REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
cd "$REPO"
echo "[repo] $REPO"

# Must match what train_a100.sh used, or the checkpoints will not be found.
: "${WORKDIR:=${SLURM_SUBMIT_DIR:-$PWD}/csasr-work}"
: "${HF_HOME:=$WORKDIR/hf}"
: "${OUT_ROOT:=$WORKDIR/exp}"
: "${RESULTS:=${SLURM_SUBMIT_DIR:-$PWD}/results}"
: "${MODELS:=m6 m7 m8}"
: "${ZERO_SHOT:=1}"
export HF_HOME
mkdir -p "$RESULTS" "$HF_HOME"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is unset (needed to pull the private test split)." >&2
    exit 1
fi

REAL=RohanRamesh/mucs-he-cs
REFS=$RESULTS/refs_test.jsonl
TABLE=$RESULTS/table2_large.jsonl

echo "== $(date) =="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

# --refs-out writes PER-UTTERANCE references even though decoding is at recording
# level. CBA's denominator has to come from the reference utterances, or it is
# inflated by ~23% with bigrams that span an utterance boundary and were never
# spoken adjacently.
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
# zero-shot at 51.9 against the paper's 52.0. Both CBA rules are reported because
# the paper's matching rule is genuinely ambiguous -- every system is scored
# identically either way, so the ordering is unaffected.
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
        echo "!! $ckpt missing -- skipping $m (did train_a100.sh finish it?)"
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
