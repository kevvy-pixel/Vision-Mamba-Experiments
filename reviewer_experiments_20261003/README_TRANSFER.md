# Vision-Mamba reviewer experiments: portable package

This directory contains the experiment code and launch scripts only. It does not include the dataset, pretrained weights, feature caches, checkpoints, or old results. Copy the directory to the second workstation and provide those paths explicitly.

## Reproduction contract

- Dataset: a six-class, patient-level dataset. One patient contributes one nine-gaze image and belongs to one class only; no patient is shared across train/validation/test.
- Fixed split: `split_seed=42`. The split must be generated from the same dataset root with `code/make_fixed_split.py`; do not use a different split seed when comparing models.
- Original DenseNet end-to-end baseline: `epochs=100`, `batch_size=4`, `num_workers=0`, AMP enabled, activation checkpointing enabled, DenseNet LR `2e-5`, temporal/head LR `2e-4`, AdamW weight decay `0.05`, `d_model=192`, `d_state=16`, depth `2`, dropout `0.1`, device `cuda:0`.
- The earlier batch-size-1 run is invalid for the original-baseline comparison and must not be reported as the corrected result.

## Launching on four RTX 4090 GPUs

The scripts are single-process CUDA programs, not DDP programs. Run independent seeds/models on separate GPUs (for example, four PowerShell windows with `--device cuda:0` through `--device cuda:3`, or with `CUDA_VISIBLE_DEVICES`), and keep one process per GPU unless you deliberately add DDP support. Example:

```powershell
pwsh .\run_phase1.ps1 -Root D:\Vision-Mamba-Reviewer-Experiments -Data D:\Source_9gaze_composed -Python C:\path\to\python.exe -Pretrained D:\weights\densenet121-a639ec97.pth -Cache D:\cache\densenet121_nine_gaze_features_seed42.pt
```

`run_phase2.ps1` additionally needs `-WeightsRoot`, `-Ref` (reference-domain root for static baselines), and `-Miccai` (CI-GNN source root). These are intentionally explicit because the original machine-specific `E:\...` paths are not portable.

Before running, install the environment matching the source repositories and verify the pretrained weight files. Run `python audit_code.py` from this directory.

No LaTeX file is modified by this package.
