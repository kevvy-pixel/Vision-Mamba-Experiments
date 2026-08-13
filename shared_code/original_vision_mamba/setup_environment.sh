#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-vision-mamba-9gaze}"
conda env create -n "${ENV_NAME}" -f environment.yml
conda run -n "${ENV_NAME}" python -c 'import torch, torchvision, mamba_ssm, causal_conv1d; print(torch.__version__, torch.cuda.is_available())'
