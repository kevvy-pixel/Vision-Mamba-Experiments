# 轻量 CNN + Clinical Mamba + Attention 概念验证

## 实验协议

- Seed：42
- 数据划分：与原 `01_linear_temporal_mamba` 的 1,716 个样本逐项一致
- 输入：九个 224×224 gaze crop
- 顺序：5→2→3→6→9→8→7→4→1
- 视觉编码器：ImageNet 预训练 MobileNetV3-Small，冻结特征
- 时序模块：2 层 Pure-PyTorch Mamba，`d_model=192`、`d_state=16`
- 训练：只训练分类/时序模块，40 epochs，依据验证集 Accuracy 选择 checkpoint

## 测试结果

| 方法 | Test Accuracy | Weighted F1 | Macro F1 | Type Acc | Pattern Acc | 可训练参数 |
|---|---:|---:|---:|---:|---:|---:|
| MobileNet + Mean Pooling | 57.89% | 48.42% | 25.28% | 83.04% | 72.51% | 114,054 |
| MobileNet + Clinical Mamba + Mean | **61.40%** | **54.74%** | 29.45% | 86.55% | **72.51%** | 617,094 |
| MobileNet + Clinical Mamba + Attention | 60.23% | 54.14% | **31.82%** | **87.13%** | 70.76% | 654,246 |

## 与既有结果比较

| 方法 | Test Accuracy | Weighted F1 | 说明 |
|---|---:|---:|---|
| 冻结 DenseNet + Mean Pooling | 54.39% | 45.55% | 原时序对照 |
| 冻结 DenseNet + Clinical Mamba | 59.65% | **54.76%** | 原 Clinical Mamba |
| 冻结 MobileNet + Clinical Mamba | **61.40%** | 54.74% | 本次轻量 CNN 概念验证 |
| 冻结 MobileNet + Clinical Mamba + Attention | 60.23% | 54.14% | 本次 Attention 验证 |
| 端到端 DenseNet + Clinical Mamba | **67.25%** | **63.88%** | 当前最终性能上限 |

## 结论

1. **轻量 CNN + Clinical Mamba 可行。** 在同一 MobileNet 特征上，Clinical Mamba 相对 Mean Pooling 提升 Accuracy 3.51 个百分点、Weighted F1 6.32 个百分点。
2. **MobileNet 可以作为 DenseNet 的轻量替代候选。** 冻结 MobileNet + Clinical Mamba 比冻结 DenseNet + Clinical Mamba 高 1.75 个百分点 Accuracy，Weighted F1 基本一致（-0.02 个百分点）。
3. **当前 Gated Attention 不应作为默认最终头。** 相对 Mamba + Mean，它的 Accuracy 下降 1.17 个百分点、Weighted F1 下降 0.60 个百分点；虽然 Macro F1 提升 2.37 个百分点，但总体收益不足。
4. MobileNet 特征编码器约 0.93M 参数，加 Mamba-Mean 后部署总参数约 1.54M；冻结 DenseNet 特征编码器加原 Mamba 约 7.66M，轻量方案参数减少约 80%。
5. 当前验证只证明冻结特征层面的可行性。若追求超过 67.25% 的最终性能，下一步应只对 `MobileNet + Clinical Mamba + Mean` 做端到端微调和多种子验证，而不是优先保留 Attention。

## Attention 权重观察

Clinical 顺序 5、2、3、6、9、8、7、4、1 的平均权重分别为：

`11.41%, 26.95%, 10.05%, 9.91%, 8.17%, 9.75%, 6.50%, 6.61%, 10.65%`

2号眼位权重明显最高，但单种子结果不足以解释为稳定的临床关注模式。
