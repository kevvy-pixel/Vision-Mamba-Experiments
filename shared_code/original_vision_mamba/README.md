# Vision Mamba for 9-Gaze Strabismus Classification

Source-domain comparison of Vim-Base + temporal Mamba against ResNet-50, DenseNet-121, ViT-Base, and Swin-B. This project does **not** perform cross-domain testing: training, validation, and testing all come exclusively from `Source_9gaze_composed`.

## Experiment protocol

- Class-stratified 70% train / 20% validation / 10% test split.
- Fixed random seed 42 for splitting, initialization, loaders, and augmentation.
- Current data produce 1202 train, 343 validation, and 171 test samples.
- All models load pretrained weights before fine-tuning for 100 epochs.
- Best checkpoint is selected only on validation performance; test is evaluated once afterward.
- `split_manifest.csv` records exact membership; both implementations produce identical path sets.
- Horizontal flipping is disabled because it changes the clinical left/right gaze relationship.
- All models share photometric augmentation, while preserving pretrained-backbone normalization.
- Vim receives nine gaze crops in clinical order; image baselines receive the full composed chart. This input organization difference should be reported with results.

## Complete installation

Clone with the pinned official Vim submodule:

```bash
git clone --recurse-submodules https://github.com/YifanWang-AI/Vision-Mamba.git
cd Vision-Mamba
conda env create -f environment.yml
conda activate vision-mamba-9gaze
```

Alternatively run `./setup_environment.sh`. The environment is specified for Python 3.10, PyTorch 2.1.1, CUDA 11.8, torchvision 0.16.1, Mamba-SSM 2.2.2, and causal-conv1d 1.4.0. The existing local environment is 7.8 GB and contains platform-specific binaries, so reproducible dependency definitions are committed instead of the non-portable environment directory.

The repository includes the exact custom ResNet and ViT model definitions under `model_sources/`. Official Vim is pinned through `.gitmodules` to commit `dd0358ad1e42701f22afbefa0717cc8825cf9f45`.

## Pretrained weights

Download public checkpoints:

```bash
./download_pretrained.sh
```

This downloads Vim-Base, DenseNet-121, ViT-Base, and Swin-B into `pretrained_weights/`. The project-specific dual-head `ResNet.pkl` has no public URL and must be copied manually to:

```text
pretrained_weights/ResNet.pkl
```

Vim-Base source: [hustvl/Vim-base-midclstok](https://huggingface.co/hustvl/Vim-base-midclstok).

## Dataset

Place the single source dataset at:

```text
Source_9gaze_composed/<class_name>/<image>
```

Expected classes are `dvd_no`, `eso_no`, `eso_V`, `exo_no`, `exo_A`, and `exo_V`. `Target_9gaze_composed` is neither required nor read.

## Training

Run all five models on four GPUs:

```bash
PYTHON_BIN="$(which python)" ./run_all_seed42.sh
```

Run one baseline:

```bash
PYTHON_BIN="$(which python)" ./run_train.sh MODEL PRETRAINED DEVICE BATCH_SIZE
```

Run Vim-Base only:

```bash
PYTHON_BIN="$(which python)" ./run_vim_mamba_9gaze.sh
```

Vim uses the memory-safe defaults batch size 1, frame chunk size 1, and AMP. Outputs include the split manifest, class counts, epoch history, best validation checkpoint, and held-out same-source test metrics. Datasets, checkpoints, environments, outputs, and logs are intentionally excluded from Git.
