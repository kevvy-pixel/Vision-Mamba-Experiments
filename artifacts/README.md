# 二进制制品说明

模型 checkpoint、预训练权重、缓存 gaze 特征和其他张量文件没有直接提交到 GitHub。`EXCLUDED_BINARY_ARTIFACTS.csv` 记录其原始相对路径、类型、字节数、MiB 和 SHA-256，可用于本地完整性核验。

排除原因：

1. ViT/Swin 权重和 checkpoint 单文件超过 GitHub 100 MB 限制；
2. 全部二进制制品体积超过常规 Git 仓库适宜范围；
3. 缓存特征可能携带从医学数据派生的信息，不应默认公开；
4. 数值指标、history、配置、日志、split manifest 和图表已经完整归档在各实验 `results/` 中。

