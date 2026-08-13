# 07 轻量 CNN + Mamba + Attention

- 冻结 MobileNet + Mean：57.89% Accuracy。
- 冻结 MobileNet + Clinical Mamba + Mean：61.40%。
- 冻结 MobileNet + Clinical Mamba + Attention：60.23%。
- 端到端 MobileNet + Clinical Mamba：61.99%。

MobileNet 方案约 1.54M 参数，适合部署；额外 Attention 没有提高总体 Accuracy/F1，因此不是默认最终头。

