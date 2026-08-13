# Nine-gaze Temporal Mamba ablation summary

## Fixed protocol

- Frozen ImageNet-pretrained DenseNet-121 features, shape `1716×9×1024`.
- Stratified source split: 1202 train / 343 validation / 171 test.
- Seed 42, 100 epochs, validation-only checkpoint selection.
- Pure-PyTorch explicit selective scan; no custom CUDA extension.
- Single-direction ablations fix two residual Mamba blocks, `d_model=192`,
  `d_state=16`, batch size 64 and the same cached features.
- The held-out test set is evaluated once after restoring the best validation
  checkpoint for each method.

## Table 2: Temporal modeling and order ablation

| Method | Best Val Acc (%) | Test Acc (%) | Test weighted F1 (%) |
|---|---:|---:|---:|
| Mean Pooling | 56.85 | 54.39 | 45.55 |
| **Clinical Linear Temporal Mamba** | **62.10** | **59.65** | **54.76** |
| Random Order Mamba | 60.06 | 58.48 | 52.37 |
| Reverse Order Mamba | 62.10 | 53.80 | 48.66 |
| Row-major Mamba | 61.52 | 56.14 | 53.97 |

Clinical order is the strongest order on held-out test accuracy and weighted F1.
Relative to the alternatives, its accuracy margins are +1.17 percentage points
over random, +5.85 over reverse, and +3.51 over row-major. This supports order
specificity, although the random-order margin is small and requires repeated
seeds before a strong statistical claim.

## Table 3: Structural improvement ablation

| Method | Parameters | Best Val Acc (%) | Test Acc (%) | Test weighted F1 (%) |
|---|---:|---:|---:|---:|
| **Linear Clinical Mamba** | 703,110 | 62.10 | **59.65** | **54.76** |
| + Closed-loop token | 703,302 | 60.35 | 57.31 | 52.14 |
| + Circular PE | 701,958 | 62.10 | 53.80 | 50.64 |
| + Bidirectional encoders | 1,402,758 | **62.39** | 54.97 | 48.94 |
| + Center-ring branch | 1,675,404 | 62.10 | 56.73 | 53.05 |

None of the tested structural additions improves held-out performance over the
simple Clinical Linear Mamba. Bidirectional Mamba obtains the highest validation
accuracy but does not transfer that gain to the test set. The larger models reach
near-perfect training performance while validation remains around 58–62%, which
is consistent with overfitting on this dataset.

## Paper-level interpretation

The supported story is intentionally simple:

1. Temporal Mamba improves over unordered mean pooling.
2. The clinically motivated order is better than random, reverse and row-major
   alternatives on the held-out test set.
3. More complex structural priors do not automatically help. A repeated token is
   not sufficient to encode a physical ring, and doubling or branching the model
   increases overfitting under the current sample size.

Accordingly, the current main method should remain **Clinical Linear Temporal
Mamba**, not the most complex center-ring variant. Closed-loop, circular PE,
bidirectional and center-ring belong in the ablation section as negative results.
Before publication, the highest-priority follow-up is repeated seeds for Mean,
Clinical, Random, Reverse and Row-major; Graph-Mamba remains deferred.

## Folder layout

Each framework is self-contained under `experiments/`:

- `00_mean_pooling/`
- `01_linear_temporal_mamba/`
- `02_random_order_mamba/`
- `03_reverse_order_mamba/`
- `04_row_major_mamba/`
- `05_closed_loop_mamba/`
- `06_circular_pe_mamba/`
- `07_bidirectional_mamba/`
- `08_center_ring_mamba/`

Every folder contains the available best checkpoint, epoch history, fixed split
manifest and test results. New structural variants additionally include an exact
`model_config.json` with their sequences, parameter count and fusion definition.

