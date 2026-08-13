# DenseNet-121 + Clinical Mamba 六分类单头短验证

## 设置

- 基础模型：端到端 DenseNet-121 + Clinical Mamba（原最佳 epoch 50）
- 冻结：DenseNet、两层 Clinical Mamba 和归一化层全部冻结
- 新分类头：`Linear(192, 6)`，仅 1,158 个可训练参数
- 暖启动：`head6[k] = type_head[t(k)] + pattern_head[p(k)]`
- 标签顺序：`dvd_no / eso_no / eso_V / exo_A / exo_no / exo_V`
- 数据划分：与原模型完全相同，seed 42，训练/验证/测试 = 1,202/343/171
- 短验证：40 epochs；另比较 `label_smoothing=0.1`

## 结果

| 方法 | Val Accuracy | Test Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|---:|
| 原双头 / 六类概率融合（epoch 0暖启动） | **64.43%** | 67.25% | 63.88% | 47.22% |
| 六分类单头 CE（最佳训练 epoch 1） | 63.85% | **68.42%** | **65.34%** | **48.14%** |
| 六分类单头 + label smoothing 0.1（epoch 1） | 64.14% | 66.67% | 63.75% | 46.86% |

六分类CE头相对原模型：

- Test Accuracy：+1.17个百分点（多判对2/171个样本）
- Weighted F1：+1.46个百分点
- Macro F1：+0.92个百分点
- 改善集中在 `exo_V`：Recall由35.90%升至41.03%，F1由43.75%升至48.48%
- `exo_A`仍为0 Recall

## 配对与校准检查

- 两个模型只有2个测试样本预测不同，且都由错误变为正确。
- McNemar exact p = 0.50，不支持统计显著改进。
- Accuracy差值bootstrap 95%区间约为 `[0.00, 2.92]`个百分点。
- 原模型平均置信度93.44%，NLL 2.1163。
- CE单头平均置信度93.47%，NLL 2.1180：没有改善过度自信。
- Label smoothing将平均置信度降至92.97%，NLL改善为2.0164，但Accuracy下降至66.67%。

## 与其他模型比较

| 模型 | Test Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|
| ViT静态基线 | 59.65% | 53.34% | 28.78% |
| Swin-B静态基线 | 63.16% | 52.72% | 31.39% |
| DenseNet-121静态基线 | 66.08% | 60.68% | 43.38% |
| MobileNet + Clinical Mamba（端到端） | 61.99% | 55.97% | 32.60% |
| DenseNet + Clinical Mamba原双头 | 67.25% | 63.88% | 47.22% |
| DenseNet + Clinical Mamba + 六分类CE头 | **68.42%** | **65.34%** | **48.14%** |

## 结论

六分类单头是有价值的候选：它只增加/替换极少参数，并在本次测试集上取得当前最高指标。但是，训练后验证Accuracy低于未训练暖启动状态，且测试提升仅来自2个样本，McNemar检验不显著。因此不能立即把68.42%作为已确证的主模型提升。

严格按验证集Accuracy选模时，应保留epoch 0原双头/概率融合；六分类CE头应标记为探索性结果。下一步只需对head做seeds 43/44或固定latent的重复种子验证，无需重训DenseNet/Mamba。如果多种子仍稳定提升，再将其升级为最终分类头。
