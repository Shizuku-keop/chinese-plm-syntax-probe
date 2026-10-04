# 可行性核查：笔记本 + 中国大陆网络 + 零预算

> 核查日期：2026-09-26。**结论：完全可行。** CPU 笔记本抽取约 5000 句的 BERT 隐状态是 15–60 分钟的量级，不是过夜，更不是一周。

## 数据集（已核实）

| 资源 | 事实 |
|---|---|
| UD Chinese-GSD | [树库页](https://universaldependencies.org/treebanks/zh_gsd/index.html) · [GitHub](https://github.com/UniversalDependencies/UD_Chinese-GSD) · 4,997 句 / 123,289 词元 · CoNLL-U · CC BY-SA 4.0 · **繁体** |
| GSD 划分 | train 3,997 / dev 500 / test 500 |
| **UD Chinese-GSDSimp** | [树库页](https://universaldependencies.org/treebanks/zh_gsdsimp/index.html) · [GitHub](https://github.com/UniversalDependencies/UD_Chinese-GSDSimp) · 同 4,997 句的简体版，OpenCC 转换+人工修正 · CC BY-SA 4.0 · **HFL 简体模型应选它** |
| UD Chinese-PUD | 1,000 句，仅测试集，可作额外留出测试 |
| CLiMP | [GitHub](https://github.com/beileixiang/CLiMP) · 16,000 最简对 · 行为基准，可作讨论章节的对照 |
| CTB | LDC 付费，跳过 |

下载路线：`raw.githubusercontent.com` 在本机超时；**jsDelivr**（`cdn.jsdelivr.net/gh/<org>/<repo>@master/<file>`）是大陆可靠的 raw 文件通道；github.com 主站可达。

## 模型（已核实）

| HF id | 参数量 | ModelScope |
|---|---|---|
| hfl/chinese-bert-wwm-ext | ~102M | `dienstag/chinese-bert-wwm-ext` ✅ |
| hfl/chinese-roberta-wwm-ext | ~102M | `dienstag/chinese-roberta-wwm-ext` ✅ |
| hfl/chinese-macbert-base | ~102M | `dienstag/chinese-macbert-base` ✅ |
| google-bert/bert-base-chinese | ~102M | `AI-ModelScope/bert-base-chinese` ✅ |
| google-bert/bert-base-multilingual-cased | ~178M | `AI-ModelScope/bert-base-multilingual-cased` ✅ |
| Qwen/Qwen2.5-0.5B | ~494M（24 层，hidden 896） | `Qwen/Qwen2.5-0.5B` ✅ |

huggingface.co 在本机不可达（封锁属实）；hf-mirror.com 的页面与 API 可用，但**权重文件会 302 到 `cas-bridge.xethub.hf.co`，本机无法解析——hf-mirror 不能用于下载权重**。已验证的可行路线：**ModelScope 直链下载**（`https://modelscope.cn/models/<id>/resolve/master/<file>`，支持 curl 断点续传），权重落盘到 `models/` 目录后用本地路径加载。

## 算力测算

- BERT-base 抽取 5,000 短句隐状态（13 层全开，`output_hidden_states=True`）：现代 CPU 约 **15–60 分钟**；存储约 1.3 GB（fp16）。
- 探针训练（单个 768×rank 线性层，L1 损失）：CPU 上几分钟。
- Qwen2.5-0.5B：CPU 约 1–3 小时，可做跨架构对比。
- 免费 GPU 备胎（均已核实存在但用不上）：Kaggle 30h/周（大陆直连传闻未一手验证）、百度 AI 星河社区（V100/A100 签到时长）、阿里云 PAI-DSW 试用（A10/V100，250 小时/月×3 个月）、AutoDL（¥1.9/h 4090）。

## 参考代码与工具

- [john-hewitt/structural-probes](https://github.com/john-hewitt/structural-probes) 存在（405 星，最后提交 2024-03），但栈是 2019 年的（`pytorch-pretrained-bert` 已废弃）。**2026 年的正确做法：读它学数学（距离/深度探针、UUAS、Spearman），自己动手写约 150 行的极简探针**——顺便解决老仓库从未处理的"中文字符分词器 ↔ UD 词对齐"问题（首子词池化）。
- PyPI 可达：`conllu` v6.0.0、`udapi` v0.5.2；骨干用 `transformers` + `torch`。

## 推荐技术栈

1. 数据：UD Chinese-**GSDSimp**（jsDelivr 下载），PUD 作额外测试。
2. 模型：`hfl/chinese-macbert-base` + `hfl/chinese-roberta-wwm-ext` + `bert-base-chinese`（ModelScope 直链下载，已验证）；Qwen2.5-0.5B 作跨架构对照。
3. 实现：PyTorch 自写极简结构探针。
4. 算力：本地 CPU 即可。
