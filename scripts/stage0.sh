#!/usr/bin/env bash
# Stage 0 (local, CPU): acquire corpora, clear GATE 0, publish to the Hub.
#
# Track 2 never trains on real code-switched audio, so the 89.8h MUCS train split
# is prepared TEXT-ONLY. Only the 4h dev slice and the 5.2h test set get cut.
#
#   bash scripts/stage0.sh                 # everything
#   bash scripts/stage0.sh --skip-push     # stop before touching the Hub
#
# Requires: pip install -e ".[local,dev]"

set -euo pipefail

PY="${PY:-./.venv/Scripts/python.exe}"
export PYTHONIOENCODING=utf-8
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

REAL_REPO="${REAL_REPO:-RohanRamesh/mucs-he-cs}"
SKIP_PUSH=0
[[ "${1:-}" == "--skip-push" ]] && SKIP_PUSH=1

echo "==> 1/6  download MUCS (7.3 GB + 443 MB, resumable)"
"$PY" -m csasr.data.download --splits test train

echo "==> 2/6  prepare test (cut audio) and train (TEXT ONLY) + 4h dev slice"
"$PY" -m csasr.data.prepare_mucs \
    --root data/mucs/test --out manifests/mucs_test.jsonl \
    --cut-audio --audio-dir data/mucs_cut/test

"$PY" -m csasr.data.prepare_mucs \
    --root data/mucs/train --out manifests/mucs_train.jsonl \
    --dev-hours 4 --dev-out manifests/mucs_dev.jsonl

echo "==> 3/6  cut audio for the 4h dev slice ONLY (never the other 86h)"
"$PY" -m csasr.data.prepare_mucs \
    --root data/mucs/train --out manifests/mucs_dev.jsonl \
    --keep-ids manifests/mucs_dev.jsonl \
    --cut-audio --audio-dir data/mucs_cut/dev

echo "==> 4/6  GATE 0 - reproduce Table 1"
"$PY" scripts/verify_table1.py \
    --train manifests/mucs_train.jsonl --test manifests/mucs_test.jsonl

echo "==> 5/6  Common Voice: stream 15h each of hi and en (en/train's 45 GB is never touched)"
"$PY" -m csasr.data.prepare_cv --langs hi en --hours 15 --max-per-speaker 200

if [[ "$SKIP_PUSH" == "1" ]]; then
    echo "==> 6/6  skipped (--skip-push)"
    exit 0
fi

echo "==> 6/6  publish to $REAL_REPO  (needs: huggingface-cli login)"
"$PY" -m csasr.data.push_to_hub --manifest manifests/mucs_train.jsonl \
    --repo "$REAL_REPO" --config train_text --text-only --verify
for cfg in dev test; do
    "$PY" -m csasr.data.push_to_hub --manifest "manifests/mucs_${cfg}.jsonl" \
        --repo "$REAL_REPO" --config "$cfg" --verify
done
for lang in hi en; do
    "$PY" -m csasr.data.push_to_hub --manifest "manifests/cv_${lang}.jsonl" \
        --repo "$REAL_REPO" --config "cv_${lang}" --verify
done

cat <<'EOF'

Stage 0 complete.

You can now delete the real training audio -- Track 2 never trains on it:
    rm -rf data/mucs/train data/raw/Hindi-English_train.tar.gz    # frees ~17.7 GB

Next: run notebooks/03_eval.ipynb GATE 3 (large-v2 zero-shot, expect ~52.0 MER)
BEFORE generating anything.
EOF
