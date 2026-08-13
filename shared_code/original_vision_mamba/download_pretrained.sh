#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHT_DIR="${SCRIPT_DIR}/pretrained_weights"
mkdir -p "${WEIGHT_DIR}"

curl -L --continue-at - -o "${WEIGHT_DIR}/vim_b_midclstok_81p9acc.pth" \
  https://huggingface.co/hustvl/Vim-base-midclstok/resolve/main/vim_b_midclstok_81p9acc.pth
curl -L --continue-at - -o "${WEIGHT_DIR}/densenet121-a639ec97.pth" \
  https://download.pytorch.org/models/densenet121-a639ec97.pth
curl -L --continue-at - -o "${WEIGHT_DIR}/swin_b-68c6b09e.pth" \
  https://download.pytorch.org/models/swin_b-68c6b09e.pth
curl -L --continue-at - -o "${WEIGHT_DIR}/vit_base_patch16_224_in21k.pth" \
  https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth

printf '%s\n' 'ResNet.pkl is a project-specific dual-head checkpoint and has no public URL.' \
  'Copy it manually to pretrained_weights/ResNet.pkl.'
