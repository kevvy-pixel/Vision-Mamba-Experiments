# Vision Mamba 9-Gaze experiment status

Protocol: source-only class-stratified 70/20/10 split, seed 42, 100 epochs, validation-only checkpoint selection, one held-out test evaluation.

Dataset: 1716 PNG images. The generated split is 1202 train, 343 validation, and 171 test images.

Runnable queue on this host:

- DenseNet-121, batch size 4
- ViT-Base, batch size 2
- Swin-B, batch size 1

Host constraints:

- ResNet-50 has been intentionally abandoned because the project-specific `ResNet.pkl` is absent and the README provides no public URL.
- Vim + temporal Mamba is blocked because the official CUDA extensions (`selective_scan_cuda` and `causal_conv1d`) are unavailable on this Windows host. The repository's pinned environment is Linux-oriented.
- The additional variants proposed by `Inspiration.md` are research directions and are not implemented by the repository.

Pure-PyTorch Temporal Mamba validation has now been implemented separately,
without custom CUDA extensions. In the matched frozen-feature ablation it reached
0.5965 test accuracy versus 0.5439 for mean pooling (+5.26 percentage points).
See `TEMPORAL_MAMBA_RESULT.md` for the complete protocol and limitations.

Live machine-readable progress is recorded in `status.json`; per-model console output is under `logs/`; checkpoints, split manifests, histories, and test metrics are under `runs/`.
