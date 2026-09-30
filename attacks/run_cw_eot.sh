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

"$PYTHON" ./attacks/attack_cw_eot_th_aware.py \
    --input_dir "$INPUT_DIR" \
    --checkpoint "$CHECKPOINT" \
    --train_config "$CONFIG" \
    --out_dir "$OUT_DIR" \
    --model_tag "$MODEL_TAG" \
    --device cuda \
    --label 1 \
    --batch_size 30 \
    --num_workers 4 \
    --eps 0.031372549019607843 \
    --norm Linf \
    --steps 100 \
    --lr 0.01 \
    --kappa 1.0 \
    --c 1.0 \
    --dist_weight 0.0 \
    --eval_every 10 \
    --eot_samples 10 \
    --eot_degrees 3 \
    --eot_translate 0.03 \
    --eot_scale_min 0.97 \
    --eot_scale_max 1.03 \
    --eot_noise_std 0.01