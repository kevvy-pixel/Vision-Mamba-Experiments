#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/Vim/vim:${PYTHONPATH:-}"

exec "${SCRIPT_DIR}/vim_env/bin/python" "${SCRIPT_DIR}/train_vim_mamba_9gaze.py" \
  --data-dir "${SCRIPT_DIR}/Source_9gaze_composed" \
  --vim-dir "${SCRIPT_DIR}/Vim" \
  --pretrained "${SCRIPT_DIR}/pretrained_weights/vim_b_midclstok_81p9acc.pth" \
  --output-dir "${SCRIPT_DIR}/outputs/vim_base_mamba_source_721_seed42" \
  --seed 42 --batch-size 1 --frame-chunk-size 1 \
  --device cuda:0 --amp "$@"
