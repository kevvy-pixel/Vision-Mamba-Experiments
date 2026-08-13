#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/vim_env/bin/python}"
MODEL="${1:?Usage: ./run_train.sh MODEL PRETRAINED [DEVICE] [BATCH_SIZE] [REFERENCE_DIR]}"
PRETRAINED="${2:?Provide the pretrained checkpoint path}"
DEVICE="${3:-cuda:0}"
BATCH_SIZE="${4:-16}"
REFERENCE_DIR="${5:-${SCRIPT_DIR}/model_sources}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/train_source_domain.py" \
  --source-dir "${SCRIPT_DIR}/Source_9gaze_composed" \
  --reference-dir "${REFERENCE_DIR}" \
  --pretrained "${PRETRAINED}" \
  --model "${MODEL}" \
  --output-dir "${SCRIPT_DIR}/outputs/${MODEL}_source_721_seed42" \
  --epochs 100 --seed 42 --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" "${@:6}"
