# Supplementary experiment status and results

## 1. Repeated seeds (completed)

Seeds 42, 43 and 44 use seed-specific class-stratified splits, initialization
and data-loader order. All methods reuse the same frozen visual feature cache;
the cache is re-indexed by image path for each split. Values below are arithmetic
mean ± sample standard deviation across the three seeds.

| Method | Test accuracy (%) | Test weighted F1 (%) |
|---|---:|---:|
| Mean Pooling | 55.36 ± 0.89 | 46.28 ± 1.12 |
| **Clinical Linear Temporal Mamba** | **58.48 ± 2.03** | **53.34 ± 1.93** |
| Random Order Mamba | 56.34 ± 2.05 | 52.14 ± 0.84 |
| Row-major Mamba | 57.89 ± 1.75 | 52.73 ± 2.07 |

Clinical Mamba remains best on both aggregate metrics. Its mean accuracy is
3.12 percentage points above Mean Pooling, 2.14 above Random Order, and 0.59
above Row-major. The small Clinical–Row-major margin should not be described as
statistically significant from three seeds alone.

Exact per-seed results are in `multi_seed/per_seed_results.csv`; aggregate
values are in `multi_seed/mean_std_summary.csv`.

## 2. Static visual baselines (running, preserved)

The original 100-epoch DenseNet-121 → ViT-Base → Swin-B queue remains active.
It operates on the full composed chart and is separate from the nine-crop
Temporal Mamba experiments. Final comparison will be added only after all three
models have completed validation checkpoint selection and held-out testing.

## 3. End-to-end DenseNet + Temporal Mamba (running)

The new end-to-end experiment jointly optimizes all DenseNet-121 parameters and
the Clinical Pure-PyTorch Temporal Mamba/classification heads:

- input: nine clinical-order 224×224 gaze crops;
- DenseNet learning rate: `2e-5`;
- Temporal Mamba/head learning rate: `2e-4`;
- two Mamba blocks, `d_model=192`, `d_state=16`;
- batch size 4, AMP and activation checkpointing;
- seed 42, 100 epochs, validation-only selection, one final test evaluation.

The implementation passed a real forward/backward/optimizer dry-run. Batch 4
used approximately 1.22 GB peak allocated CUDA memory in isolation. Live state
is under `end_to_end/status.json` and epoch output under `end_to_end/logs/`.

## 4. Clinical error analysis (completed for seed 42)

Six-class, type and pattern confusion matrices have been produced for Mean,
Clinical, Random and Row-major. Each method has raw and row-normalized PNGs,
CSV matrices and complete precision/recall/F1 reports under
`clinical_analysis/<method>/`.

Key findings on the 171-image held-out seed-42 test set:

- The test distribution is severely imbalanced: `exo_no` has 82 cases while
  `exo_A` has only 4, `eso_V` 6 and `dvd_no` 10.
- Clinical Mamba improves eso recall from 52.8% (Mean) to 75.0%.
- Clinical Mamba improves V-pattern recall from 6.7% to 24.4%.
- `exo_V` F1 improves from 13.0% to 36.1%.
- All four analyzed methods have 0 recall for the four-case `exo_A` class.
- Clinical Mamba still has 0 recall for `eso_V` and only 10% recall for
  `dvd_no`; these limitations must be disclosed.

The six-class decoder scores only the six clinically present `(type, pattern)`
combinations, so impossible combinations such as `dvd_A` are not included in
the confusion matrix.

