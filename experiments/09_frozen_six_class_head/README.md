# 09 冻结编码器六分类头

结构：冻结 DenseNet-121 + Clinical Mamba + LayerNorm，Mean Pool 得到 192 维患者表征，只训练 `Linear(192,6)`，共 1,158 个参数。

| 方案 | Val Acc | Test Acc | Weighted F1 | Macro F1 |
|---|---:|---:|---:|---:|
| 原双头概率融合 | 64.43% | 67.25% | 63.88% | 47.22% |
| 六分类 CE | 63.85% | **68.42%** | **65.34%** | **48.14%** |
| Label smoothing 0.1 | 64.14% | 66.67% | 63.75% | 46.86% |

CE 头是探索性最高测试结果；McNemar exact `p=0.50`，需补充 head-only seeds 43/44 后再决定是否升级为正式最终头。

