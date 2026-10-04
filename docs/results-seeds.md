# 多种子鲁棒性（seed 0 / 1 / 2，三个中文单语模型）

所有数字由 `src/summarize_seeds.py` 从 `results/*_seed{1,2}.json` 与 seed 0 的主结果现算。

| 模型 | 种子 | UUAS 峰值层 | dev UUAS | test UUAS | 层 8 深度 Δ | MDL 最低层 | MDL 最低 kbit |
|---|---|---|---|---|---|---|---|
| BERT | 0 | 8 | 0.587 | 0.596 | 0.071 | 8 | 157.1 |
| BERT | 1 | （缺数据） | | | | | |
| BERT | 2 | （缺数据） | | | | | |
| RoBERTa-wwm-ext | 0 | 8 | 0.587 | 0.605 | 0.072 | 8 | 157.9 |
| RoBERTa-wwm-ext | 1 | （缺数据） | | | | | |
| RoBERTa-wwm-ext | 2 | （缺数据） | | | | | |
| MacBERT | 0 | 8 | 0.589 | 0.597 | 0.113 | 8 | 160.7 |
| MacBERT | 1 | （缺数据） | | | | | |
| MacBERT | 2 | （缺数据） | | | | | |

## 种子间的稳定性

| 模型 | 峰值层（各 seed） | dev UUAS 各 seed | dev UUAS 极差（pp） | 层 8 深度 Δ 各 seed | Δ 极差（pp） |
|---|---|---|---|---|---|
| BERT | 8 | 0.587 | 0.0 | 0.071 | — |
| RoBERTa-wwm-ext | 8 | 0.587 | 0.0 | 0.072 | — |
| MacBERT | 8 | 0.589 | 0.0 | 0.113 | — |

## 模型间排序在各 seed 上是否稳定（dev UUAS 峰值层）

- MacBERT − BERT：seed 0: +0.2 pp
- RoBERTa − BERT：seed 0: -0.0 pp
- MacBERT − RoBERTa：seed 0: +0.2 pp

