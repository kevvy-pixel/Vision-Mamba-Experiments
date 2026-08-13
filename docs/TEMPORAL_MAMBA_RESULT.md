# Pure-PyTorch Temporal Mamba validation

## Outcome

The clinical-order Pure-PyTorch Temporal Mamba produced a positive matched
ablation result on the held-out source test set.

| Model over identical 9×1024 frozen features | Best validation accuracy | Test accuracy | Test weighted F1 |
|---|---:|---:|---:|
| Mean pooling (no temporal model) | 0.5685 | 0.5439 | 0.4555 |
| Clinical-order Temporal Mamba | **0.6210** | **0.5965** | **0.5476** |
| Absolute improvement | **+5.25 pp** | **+5.26 pp** | **+9.21 pp** |

Temporal Mamba's best checkpoint was selected at epoch 21 using only validation
accuracy. Mean pooling's best checkpoint was selected at epoch 28. The test set
was evaluated once after loading each selected checkpoint.

## Controlled protocol

- Dataset: `Source_9gaze_composed`, 1716 images.
- Fixed seed: 42.
- Stratified split: 1202 train / 343 validation / 171 test.
- Spatial representation: public ImageNet-pretrained DenseNet-121, frozen.
- Input per sample: nine 1024-dimensional gaze features.
- Clinical sequence: 5→2→3→6→9→8→7→4→1.
- Temporal model: two residual Pure-PyTorch Mamba blocks, `d_model=192`,
  `d_state=16`, explicit nine-step selective scan.
- Control: project and average the identical nine cached features.
- Optimizer and schedule: AdamW, 100 epochs, cosine schedule.
- No `mamba_ssm`, Triton, `causal-conv1d`, or custom CUDA extension.

## Interpretation and limits

This experiment supports the hypothesis that ordered gaze transitions contain
useful information beyond unordered mean aggregation. It specifically validates
the Temporal Mamba contribution while controlling the spatial features.

It is not yet a full reproduction of Vim-Base + Temporal Mamba: the spatial
encoder here is a frozen DenseNet-121 because the official Vim stack requires
Linux-oriented custom CUDA extensions unavailable on this host. It is also a
single-seed result and should be followed by repeated seeds and order ablations
(random, row-major, reverse, and circular) before making a strong statistical
claim.

## Artifacts

- Implementation/environment: `temporal_pytorch_env/`
- Feature cache: `temporal_pytorch_runs/densenet121_nine_gaze_features_seed42.pt`
- Temporal Mamba result: `temporal_pytorch_runs/clinical_temporal_mamba_seed42/`
- Mean-pooling result: `temporal_pytorch_runs/mean_pooling_control_seed42/`
- Logs: `logs/clinical_temporal_mamba_seed42.log` and
  `logs/mean_pooling_control_seed42.log`

The original DenseNet/ViT/Swin baseline queue remains a separate active process.
ResNet has been intentionally abandoned because its project-specific pretrained
checkpoint is unavailable.

