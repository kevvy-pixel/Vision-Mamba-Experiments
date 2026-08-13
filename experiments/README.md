# 实验目录索引

| 编号 | 目录 | 对应实验 | 代码入口 | 结果内容 |
|---:|---|---|---|---|
| 00 | `00_setup_and_smoke_tests` | 环境、dry-run、1 epoch benchmark | `train_source_domain_used.py` | 划分与运行检查 |
| 01 | `01_static_visual_baselines` | DenseNet-121、ViT-Base、Swin-B | `train_source_domain_used.py` | history、split、test metrics、logs |
| 02 | `02_temporal_effectiveness` | Mean Pooling vs Clinical Mamba | `train_temporal_ablation.py` | 第一轮与正式消融结果 |
| 03 | `03_order_ablation` | Random、Reverse、Row-major | `train_temporal_ablation.py` | 三种顺序结果 |
| 04 | `04_structural_ablation` | Closed-loop、Circular PE、Bidirectional、Center-ring | `train_gaze_structure.py` | 四组结构消融 |
| 05 | `05_multi_seed_validation` | seeds 42/43/44 | `train_temporal_ablation.py` | 分种子结果与 mean±SD |
| 06 | `06_end_to_end_densenet_mamba` | DenseNet-121 + Clinical Mamba 联合微调 | `train_end_to_end_densenet_mamba.py` | 最佳 epoch 50 与测试结果 |
| 07 | `07_lightweight_cnn_mamba` | MobileNet、Mamba、Attention、端到端 | 多个轻量训练脚本 | POC、Attention 与端到端结果 |
| 08 | `08_six_class_probability_decoder` | Type/Pattern 概率映射到合法六类 | `evaluate_six_class_probability_decoder.py` | 概率、预测与验证摘要 |
| 09 | `09_frozen_six_class_head` | 冻结最佳表征后训练 Linear(192,6) | `train_dense_mamba_head6.py` | CE 与 label smoothing 对比 |
| 10 | `10_clinical_analysis` | 混淆矩阵、类别 Recall/F1 | `clinical_analysis.py` | JSON、CSV 与 PNG |

每个编号目录中的 `code/` 是该实验使用的代码快照，`results/` 是可上传 GitHub 的运行结果。模型权重和缓存张量不在 `results/` 中，详见仓库根目录 `artifacts/EXCLUDED_BINARY_ARTIFACTS.csv`。

