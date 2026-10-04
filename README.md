# 中文预训练语言模型的句法几何：结构探针、预训练策略与控制检验

本仓库是一篇研究论文的**完整可复现材料**：论文中英全文与 PDF、全部结果文件、图表、流水线代码与复现命令。

- 论文：[中文](paper/paper_zh.md) · [English](paper/paper_en.md) · PDF：[中文](paper/paper_zh_final.pdf) · [English](paper/paper_en_final.pdf)
- 汇总表（由脚本从结果文件自动生成）：[docs/results-tables.md](docs/results-tables.md)

## 结论一句话

汉语句法树可以被线性变换后的平方距离重建：三个中文单语模型的 dev UUAS（无向依存边准确率）峰值**同在第 8 层**（0.587–0.589），远高于随机树基线 0.096；这个"中层句法几何"在双向控制任务（selectivity）与 MDL 检验下存活，且不随预训练遮蔽策略改变、也不依赖中文特有的字↔词对齐方式。

## 主要结果

下表数字全部由 `src/summarize.py` 从 `results/*.json` 现算（与论文正文一致）。

| 模型 | 参数量 | UUAS 峰值层 | dev UUAS | test UUAS | 层 8 深度 Δ | 层 8 距离 ΔUUAS（首字池化） | 层 8 距离 ΔUUAS（均值池化） | MDL 最低层 / kbit |
|---|---|---|---|---|---|---|---|---|
| `bert-base-chinese` | ~102M | 8 | 0.587 | 0.596 | +0.071 | 0.491 | 0.519 | 8 / 157.1 |
| `hfl/chinese-roberta-wwm-ext` | ~102M | 8 | 0.587 | 0.605 | +0.072 | 0.489 | 0.512 | 8 / 157.9 |
| `hfl/chinese-macbert-base` | ~102M | 8 | 0.589 | 0.597 | +0.113 | 0.492 | 0.519 | 8 / 160.7 |
| `bert-base-multilingual-cased` | ~178M | 7 | 0.604 | 0.619 | +0.095 | 0.492 | （未做） | 7 / 158.3 |

- 随机树控制的 dev UUAS 全层落在 **0.093–0.104**，与随机树基线（0.096）无法区分；随机树在 **train** 上的 Spearman 为正（0.167–0.170）而 dev 上为 0，说明随机结构只是被"背"下来、并未进入表示。
- 深度控制的 selectivity 在层 0–4 为**负**（低至 −0.16），层 5 起转正——低层的高指标主要是词形记忆。
- 均值池化消融（词首子词 → 词内均值）：峰值层不变，dev UUAS 上移约 3 个百分点，深度与距离两类 selectivity 同步上移，MDL 最低码长同时缩短（如 157.1 → 150.6 kbit）。

## 复现

单台笔记本 CPU（无 GPU，未使用 CUDA），全部结果合计约 **30 小时**。逐步命令与每步产物见论文 **§3.5.8 复现步骤与环境**；最短路径：

```bash
python -m venv .venv
./.venv/Scripts/python -m pip install -r requirements.txt
./.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# 1) 树库（UD Chinese-GSDSimp，CC BY-SA 4.0，随 UD 数据发布）
mkdir -p data/ud_gsdsimp
for s in train dev test; do
  curl -L -o "data/ud_gsdsimp/zh_gsdsimp-ud-$s.conllu" \
    "https://cdn.jsdelivr.net/gh/UniversalDependencies/UD_Chinese-GSDSimp@master/zh_gsdsimp-ud-$s.conllu"
done

# 2) 模型（ModelScope 直链 → models/<名称>/）
./.venv/Scripts/python src/download_model.py AI-ModelScope/bert-base-chinese
./.venv/Scripts/python src/download_model.py dienstag/chinese-roberta-wwm-ext
./.venv/Scripts/python src/download_model.py dienstag/chinese-macbert-base
./.venv/Scripts/python src/download_model.py AI-ModelScope/bert-base-multilingual-cased

# 3) 表示抽取（四模型首字池化 + 三单语模型均值池化）
# 4) 过拟合门 + 基线     : run_first_probe.py --overfit64 / baselines.py
# 5) 主线探针            : run_first_probe.py --model <名称>
# 6) 控制实验（深度/距离）: run_control.py --control {depth,distance} --model <名称>
# 7) MDL（含自检）        : mdl.py / run_mdl.py --model <名称>
# 8) 汇总与绘图           : summarize.py / plot_results.py
```

批量驱动脚本（本仓库实际跑出结果用的）：`run_mdl_all.sh`、`run_extras.sh`、`run_extras_resume.sh`、`run_distance_means.sh`、`run_seeds.sh`。

## 目录结构

```
src/                 流水线代码（treebank / extract / probe / control_tasks / mdl / 各 run_* 入口）
run_*.sh             批量驱动脚本（串行跑多模型、断点续跑、自动汇总）
paper/               论文中英 Markdown 与定稿 PDF
docs/                汇总表 results-tables.md、可行性分析、文献版图
figures/             图 1–4（由 src/plot_results.py 生成）
results/*.json       全部结果文件（每层指标、控制实验、MDL；表与图的一切数字都从这里现算）
requirements.txt     环境依赖（版本与论文 §3.5.8 一致）
```

## 数据、模型与许可

- **数据**：UD Chinese-GSDSimp（简体，train 3,997 / dev 500 / test 500 句），许可 **CC BY-SA 4.0**，随 Universal Dependencies 数据发布（框架引用 Nivre et al., LREC 2016）。本仓库**不再分发**树库与模型权重，复现命令直接从原始来源下载。
- **模型**：`bert-base-chinese` 出自 Google Research（Apache 2.0）；`hfl/chinese-roberta-wwm-ext` 与 `hfl/chinese-macbert-base` 出自 HFL（Cui et al. 2021 / Cui et al. 2020）；`bert-base-multilingual-cased` 出自 Devlin et al. 2019。请按各自原始发布的许可使用。
- **本仓库代码**：MIT，见 [LICENSE](LICENSE)。
- **结果文件与图表**：`results/*.json`、`figures/*.png`、`docs/*.md`、`paper/*` 为本项目产出，可随论文引用。

## English

This repository contains the complete reproducible material for a study of whether Chinese pre-trained language models encode dependency syntax in the geometry of their representations, following Hewitt & Manning (2019) structural probes, Hewitt & Liang (2019) selectivity controls, and Voita & Titov (2020) MDL probing. Four models are probed layer by layer on UD Chinese-GSDSimp: three Chinese monolingual models with different pre-training objectives (`bert-base-chinese`, `chinese-roberta-wwm-ext`, `chinese-macbert-base`) and multilingual mBERT. The dev UUAS peak of all three monolingual models falls in **layer 8** (0.587–0.589, against a random-tree baseline of 0.096); the peak survives both control tasks and the MDL test; it is robust to the masking strategy and to the Chinese-specific character↔word pooling. Full paper: [paper/paper_en.md](paper/paper_en.md).
