# Vision-Mamba-Experiments

本仓库整理了九眼位斜视六分类项目的全部实验代码、可公开的数值结果、训练历史、数据划分清单、日志、混淆矩阵和综合分析报告。每组实验均采用独立目录，目录内严格分为 `code/` 与 `results/`，便于复现、核查和论文写作。

> 医学数据说明：原始数据集 `Source_9gaze_composed` 不包含在仓库中。本仓库默认按私有仓库发布。所有划分清单中的原始文件路径和患者拼音文件名已替换为确定性哈希样本 ID；日志中的原始数据路径也已清理。使用者仍应在公开发布前再次完成伦理审查。

## 1. 研究问题

项目围绕以下问题逐步验证：

1. 九眼位之间是否存在可利用的顺序信息？
2. Clinical 顺序是否优于随机、反向和行优先顺序？
3. Closed-loop、Circular PE、Bidirectional 和 Center-ring 是否能进一步提升性能？
4. DenseNet-121 + Clinical Mamba 联合训练能否超过静态视觉基线？
5. MobileNetV3-Small 能否作为更轻量的视觉编码器？
6. Mamba 后继续增加 Attention 是否有稳定收益？
7. 冻结最佳表征后，直接六分类头能否优于 Type/Pattern 双头概率融合？

## 2. 数据与统一协议

- 数据集：`Source_9gaze_composed`，共 1,716 张九眼位 PNG。
- 划分：类别分层 70/20/10，训练/验证/测试为 1,202/343/171。
- 主随机种子：42；关键顺序消融补充 43、44。
- 六类：`dvd_no`、`eso_no`、`eso_V`、`exo_A`、`exo_no`、`exo_V`。
- Clinical 顺序：`5 → 2 → 3 → 6 → 9 → 8 → 7 → 4 → 1`。
- Pure-PyTorch Mamba：2 个 residual block，`d_model=192`、`d_state=16`，不依赖自定义 CUDA selective scan。
- 最佳 checkpoint 仅依据验证集选择，测试集用于最终 held-out 评估。

## 3. 仓库目录

```text
Vision-Mamba-Experiments/
├─ experiments/                 # 每个实验均包含 code/ 与 results/
│  ├─ 00_setup_and_smoke_tests/
│  ├─ 01_static_visual_baselines/
│  ├─ 02_temporal_effectiveness/
│  ├─ 03_order_ablation/
│  ├─ 04_structural_ablation/
│  ├─ 05_multi_seed_validation/
│  ├─ 06_end_to_end_densenet_mamba/
│  ├─ 07_lightweight_cnn_mamba/
│  ├─ 08_six_class_probability_decoder/
│  ├─ 09_frozen_six_class_head/
│  └─ 10_clinical_analysis/
├─ shared_code/                 # 原始 Vision Mamba、Pure-PyTorch Mamba 与参考源码快照
├─ reports/                     # Word 综合分析报告与报告图表
├─ docs/                        # 全局状态、汇总表与补充实验说明
├─ artifacts/                   # 未上传二进制制品的路径、大小与 SHA-256
└─ tools/                       # 整理和完整性检查脚本
```

详细实验到目录的对应关系见 [experiments/README.md](experiments/README.md)。

## 4. 核心结果

### 4.1 静态视觉基线与最终模型

| 模型 | 输入 | Test Accuracy | Weighted F1 | Macro F1 |
|---|---|---:|---:|---:|
| ViT-Base | 整张九眼位合成图 | 59.65% | 53.34% | 28.78% |
| Swin-B | 整张九眼位合成图 | 63.16% | 52.72% | 31.39% |
| DenseNet-121 | 整张九眼位合成图 | 66.08% | 60.68% | 43.38% |
| MobileNetV3-Small + Clinical Mamba（端到端） | 9 gaze crop | 61.99% | 55.97% | 32.60% |
| DenseNet-121 + Clinical Mamba（端到端双头） | 9 gaze crop | 67.25% | 63.88% | 47.22% |
| 冻结最佳编码器 + 六分类 CE 头 | 9 gaze crop | **68.42%** | **65.34%** | **48.14%** |

最后一项是当前测试指标最高的探索性候选，但不能直接宣称已经替代正式主模型：其验证准确率由 64.43% 降至 63.85%，171 个测试样本中仅 2 个预测由错误变为正确，McNemar exact `p=0.50`。因此证据最完整的论文主模型仍是端到端 DenseNet-121 + Clinical Mamba 双头模型，六分类 CE 头需要补充 head-only seeds 43/44。

### 4.2 Temporal modeling 与顺序消融

| 方法 | Test Accuracy | Weighted F1 |
|---|---:|---:|
| Mean Pooling | 54.39% | 45.55% |
| **Clinical Linear Temporal Mamba** | **59.65%** | **54.76%** |
| Random Order | 58.48% | 52.37% |
| Reverse Order | 53.80% | 48.66% |
| Row-major | 56.14% | 53.97% |

Clinical Mamba 相对 Mean Pooling 提升 5.26 个百分点 Accuracy 和 9.21 个百分点 weighted F1，证明九眼位顺序信息有效。Clinical 顺序在 seed 42 上优于 Random、Reverse 和 Row-major，但三种子均值仅比 Row-major 高 0.59 个百分点，因此应表述为“总体趋势最好”，而不是“统计显著优于”。

### 4.3 结构增强消融

| 方法 | Test Accuracy | Weighted F1 |
|---|---:|---:|
| Clinical Linear Mamba | **59.65%** | **54.76%** |
| + Closed-loop token | 57.31% | 52.14% |
| + Circular PE | 53.80% | 50.64% |
| + Bidirectional | 54.97% | 48.94% |
| + Center-ring | 56.73% | 53.05% |

所有复杂结构均未超过 Linear Clinical Mamba。更大模型在训练集接近饱和，但验证/测试收益没有同步出现，说明当前样本量下主要风险是过拟合，而不是模型容量不足。

### 4.4 轻量 CNN、Mamba 与 Attention

| 方法 | Test Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|
| MobileNet + Mean Pooling | 57.89% | 48.42% | 25.28% |
| **MobileNet + Clinical Mamba + Mean** | **61.40%** | **54.74%** | 29.45% |
| MobileNet + Clinical Mamba + Attention | 60.23% | 54.14% | **31.82%** |

Mamba 与 Attention 在结构上不存在“原生冲突”，但在本数据集上额外 Gated Attention 没有带来稳定总体收益：相对 Mamba + Mean，Accuracy 下降 1.17 个百分点、weighted F1 下降 0.60 个百分点。它只提高了 macro F1，因此不作为默认最终分类头。

## 5. 综合分析报告

完整中文报告位于 `reports/Vision_Mamba_九眼位实验综合分析报告.docx`，主要包括：

- Swin-B 完成状态核查；
- DenseNet、ViT、Swin 静态基线对比；
- Temporal modeling、顺序和结构消融；
- seeds 42/43/44 稳定性分析；
- DenseNet 与 MobileNet 端到端结果；
- 六分类混淆矩阵和临床类别分析；
- 冻结编码器六分类头验证、配对统计与校准结果；
- 论文主结论、局限和下一步实验优先级。

## 6. 复现方式

1. 创建 Python 环境并安装 `shared_code/temporal_mamba/requirements.txt` 中的依赖。
2. 将数据集放在本地合规目录，运行脚本时通过命令行参数指定数据路径；仓库不内置数据。
3. 静态基线使用 `experiments/01_static_visual_baselines/code/train_source_domain_used.py`。
4. Pure-PyTorch Mamba 与消融实验从相应实验目录的 `code/` 启动。
5. 每个实验的参数、输出结构和结果说明见该目录下的 `README.md`。发布版 split manifest 使用 `sample_<SHA256前16位>.png`，同一原始样本跨实验保持相同 ID。

原始脚本保留了本机实验路径作为历史记录；在其他机器运行前，请将数据和输出参数替换为本地路径。

## 7. 二进制制品与 GitHub 限制

本仓库没有提交数据集、虚拟环境、预训练权重、模型 checkpoint 和缓存特征。原因包括医学数据治理、GitHub 单文件 100 MB 限制及仓库体积控制。所有被排除二进制文件均记录在 `artifacts/EXCLUDED_BINARY_ARTIFACTS.csv`，包括原始相对路径、文件大小和 SHA-256；这样可以验证本地制品是否与本次实验一致。

## 8. 当前结论

- 九眼位顺序信息有效，Clinical Temporal Mamba 明显优于 Mean Pooling。
- 简单 Linear Clinical Mamba 比闭环、环形位置编码、双向和 Center-ring 更稳健。
- DenseNet-121 + Clinical Mamba 是当前证据最完整的主模型。
- 六分类 CE 头取得 68.42% 的探索性最高测试准确率，但必须补充重复种子后才能升级为正式最终头。
- MobileNetV3-Small 适合作为约 1.54M 参数的轻量部署分支；Attention 不是当前优先优化方向。
