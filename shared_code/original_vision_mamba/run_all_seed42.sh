#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHT_DIR="${SCRIPT_DIR}/pretrained_weights"
RESNET50_WEIGHTS="${RESNET50_WEIGHTS:-${WEIGHT_DIR}/ResNet.pkl}"
DENSENET121_WEIGHTS="${DENSENET121_WEIGHTS:-${WEIGHT_DIR}/densenet121-a639ec97.pth}"
VIT_WEIGHTS="${VIT_WEIGHTS:-${WEIGHT_DIR}/vit_base_patch16_224_in21k.pth}"
SWIN_B_WEIGHTS="${SWIN_B_WEIGHTS:-${WEIGHT_DIR}/swin_b-68c6b09e.pth}"
REFERENCE_DIR="${REFERENCE_DIR:-${SCRIPT_DIR}/model_sources}"
mkdir -p "${SCRIPT_DIR}/logs"

nohup "${SCRIPT_DIR}/run_vim_mamba_9gaze.sh" >"${SCRIPT_DIR}/logs/vim_base_source_721_seed42.log" 2>&1 &
nohup "${SCRIPT_DIR}/run_train.sh" resnet50 "${RESNET50_WEIGHTS}" cuda:1 16 "${REFERENCE_DIR}" >"${SCRIPT_DIR}/logs/resnet50_source_721_seed42.log" 2>&1 &
nohup "${SCRIPT_DIR}/run_train.sh" densenet121 "${DENSENET121_WEIGHTS}" cuda:1 16 "${REFERENCE_DIR}" >"${SCRIPT_DIR}/logs/densenet121_source_721_seed42.log" 2>&1 &
nohup "${SCRIPT_DIR}/run_train.sh" vit "${VIT_WEIGHTS}" cuda:2 8 "${REFERENCE_DIR}" >"${SCRIPT_DIR}/logs/vit_source_721_seed42.log" 2>&1 &
nohup "${SCRIPT_DIR}/run_train.sh" swin_b "${SWIN_B_WEIGHTS}" cuda:3 4 "${REFERENCE_DIR}" >"${SCRIPT_DIR}/logs/swin_b_source_721_seed42.log" 2>&1 &
wait
