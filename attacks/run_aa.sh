#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"

PROJECT_ROOT="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &&
    pwd
)"

INPUT_DIR="${1:?Usage: $0 INPUT_DIR OUT_DIR CHECKPOINT CONFIG MODEL_TAG}"
OUT_DIR="${2:?Missing OUT_DIR}"
CHECKPOINT="${3:?Missing CHECKPOINT}"
CONFIG="${4:?Missing CONFIG}"
MODEL_TAG="${5:?Missing MODEL_TAG}"

cd "$PROJECT_ROOT"

"$PYTHON" ./attacks/attack_aa_th_aware.py \
    --input_dir "$INPUT_DIR" \
    --checkpoint "$CHECKPOINT" \
    --train_config "$CONFIG" \
    --out_dir "$OUT_DIR" \
    --model_tag "$MODEL_TAG" \
    --device cuda \
    --label 1 \
    --batch_size 32 \
    --num_workers 0 \
    --eps 0.031372549019607843 \
    --norm Linf \
    --version custom \
    --attacks_to_run apgd-ce,fab,square