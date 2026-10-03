# Pure PyTorch Temporal Mamba validation

This isolated experiment contains no dependency on `mamba_ssm`, Triton,
`causal-conv1d`, or custom CUDA extensions. It uses the public pretrained
DenseNet-121 only as a frozen per-gaze spatial feature extractor. The direct
comparison is:

1. `mean`: project and average the same nine cached features (no temporal model).
2. `mamba`: process the same features in clinical order 5→2→3→6→9→8→7→4→1
   using a two-layer pure-PyTorch selective state-space scan.

The dataset split, seed, validation checkpoint selection, and held-out test
protocol match the repository README. This isolates the contribution of the
Temporal Mamba while the original DenseNet/ViT/Swin baselines run separately.
