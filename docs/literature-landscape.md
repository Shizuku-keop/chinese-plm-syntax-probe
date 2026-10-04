# 文献版图核查：中文预训练模型的探针研究

> 核查日期：2026-09-26。所有条目均经 ACL Anthology / arXiv / PMC 原页核实。

## 1. 奠基性探针文献

| 论文 | 出处 | 方法一句话 |
|---|---|---|
| Hewitt & Manning, "A Structural Probe for Finding Syntax in Word Representations" (2019) | NAACL-HLT 2019, pp. 4129–4138 · [N19-1419](https://aclanthology.org/N19-1419/) · [代码](https://github.com/john-hewitt/structural-probes) | 学一个线性变换 B，使变换后向量的平方 L2 距离逼近依存树距离；用最小生成树重建句法树 |
| Tenney, Das & Pavlick, "BERT Rediscovers the Classical NLP Pipeline" (2019) | ACL 2019, pp. 4593–4601 · [P19-1452](https://aclanthology.org/P19-1452/) | 对 BERT 各层做 edge probing，发现各层按经典 NLP 流水线顺序处理任务 |
| Voita & Titov, "Information-Theoretic Probing with Minimum Description Length" (2020) | EMNLP 2020, pp. 183–196 · [2020.emnlp-main.14](https://aclanthology.org/2020.emnlp-main.14/) | 用 MDL（在线编码长度）替代准确率，衡量表示传递某语言学性质的难易 |
| Pimentel et al., "Information-Theoretic Probing for Linguistic Structure" (2020) | ACL 2020, pp. 4609–4622 · [2020.acl-main.420](https://aclanthology.org/2020.acl-main.420/) | 主张以"提取的信息量（比特）"衡量探针质量，权衡探针复杂度与准确率 |
| Pimentel et al., "Pareto Probing" (2020) | EMNLP 2020, pp. 3138–3153 · [arXiv 2010.02180](https://arxiv.org/abs/2010.02180) | 在准确率/复杂度帕累托前沿上比较探针 |
| Clark et al., "What Does BERT Look At?" (2019) | BlackboxNLP @ ACL 2019, pp. 276–286 · [W19-4828](https://aclanthology.org/W19-4828/) | 发现跟踪依存关系的注意力头 |
| Hewitt & Liang, "Designing and Interpreting Probes with Control Tasks" (2019) | EMNLP-IJCNLP 2019, pp. 2733–2743 · [D19-1275](https://aclanthology.org/D19-1275/) | 提出控制任务与 selectivity = 语言任务准确率 − 控制任务准确率，剔除靠记忆词形的探针 |

## 2. 中文相关工作

- **CLiMP**（Xiang et al., EACL 2021, pp. 2784–2790, [2021.eacl-main.242](https://aclanthology.org/2021.eacl-main.242/)，[数据](https://github.com/beileixiang/CLiMP)）：16 组句法对比覆盖 9 大现象（量词-名词搭配、把字句、被字句、约束、填空-缺位、了/着/过体态等），每组 1000 对。**行为基准**（句子概率），不探内部表示；只测过 bert-base-chinese，未测 RoBERTa-wwm/MacBERT。
- **SLING**（Song et al., EMNLP 2022, pp. 4606–4634, [2022.emnlp-main.305](https://aclanthology.org/2022.emnlp-main.305/)）：38K 最简对，批评 CLiMP 部分现象的模板设计。
- **ZhoBLiMP**（Liu et al., TACL Vol. 14, 2026, pp. 755–771, [2026.tacl-1.34](https://aclanthology.org/2026.tacl-1.34/)；预印本 [arXiv 2411.06096](https://arxiv.org/abs/2411.06096)）：正式论文标题为 "A Systematic Assessment of Language Models with Linguistic Minimal Pairs in Chinese"，"ZhoBLiMP" 是数据集名。最大中文最简对集（35K 对、15 现象），并训练了 Zh-Pythia——22 个不同规模/训练量的中文模型，是现成的规模阶梯。
- **Zheng & Liu 2023**，"What does Chinese BERT learn about syntactic knowledge?"，PeerJ CS 9:e1478（[PMC10403162](https://pmc.ncbi.nlm.nih.gov/articles/PMC10403162/)）：**与本项目最接近的工作**。在 bert-base-chinese 上做注意力头探针（UUAS）+ Conneau 式表层任务（TreeDepth/BShift/DepRel），覆盖把/被/了/着/过。**但无 Hewitt–Manning 结构探针、无 selectivity、无 MDL、只研究单一模型。**
- 其他：Wang et al. COLING 2020（中文 BERT 是否编码词结构）；Koto et al. NAACL 2021（篇章探针，含中文 BERT）；Findings of ACL 2022 中文 GEC 探针（仅下游任务对比 wwm 模型）。

## 3. 跨语言探针

- **Chi, Hewitt & Manning 2020**（ACL 2020, pp. 5564–5577, [2020.acl-main.493](https://aclanthology.org/2020.acl-main.493/)）：对 mBERT 在 11 个 UD 树库（**含 Chinese-GSD**）上做结构探针，发现跨语言共享句法子空间；**中文是编码较差的语言之一**；无 selectivity/MDL 控制。标题为 "Finding Universal Grammatical Relations in Multilingual BERT"。
- Kulmizev et al. 2020（ACL 2020, pp. 4077–4091, [2020.acl-main.375](https://aclanthology.org/2020.acl-main.375/)）：13 语言上比较 UD 与 SUD 标注体系。标题为 "Do Neural Language Models Show Preferences for Syntactic Formalisms?"。
- Xu et al. 2022（EMNLP 2022, [代码](https://github.com/ningyuxu/cl-syntactic-difference-mbert)）：度量 mBERT 中语言间句法距离，含 Chinese-GSDSimp。

## 3.1 补充核查（2026-10-02，一手页面）

- 页码补齐：Chi et al. 2020 → 5564–5577；Kulmizev et al. 2020 → 4077–4091；Xiang et al. 2021 → 2784–2790；Song et al. 2022 → 4606–4634；Cui et al. MacBERT → 657–668；Rogers et al. → TACL 8:842–866。
- **标题勘误**：ZhoBLiMP 的正式论文标题是 "A Systematic Assessment of Language Models with Linguistic Minimal Pairs in Chinese"（TACL Vol. 14, 2026, pp. 755–771），不是 "ZhoBLiMP: a Benchmark for Chinese Linguistic Minimal Pairs"。
- 模型出处：`bert-base-chinese` → Devlin et al., NAACL-HLT 2019, pp. 4171–4186（Google Research，Apache 2.0）；`hfl/chinese-roberta-wwm-ext` → Cui et al., TASLP 29:3504–3514, 2021；`hfl/chinese-macbert-base` → Cui et al., Findings of EMNLP 2020, pp. 657–668。
- UD Chinese-GSDSimp **无独立论文**（不得编造）：随 UD 数据发布引用，框架引 Nivre et al., LREC 2016, pp. 1659–1666；贡献者 Peng Qi、Koichi Yasuoka；许可 CC BY-SA 4.0。

## 4. 新颖性结论

**"Hewitt–Manning 结构探针 + 中文单语模型家族（bert-base-chinese / RoBERTa-wwm-ext / MacBERT）+ UD 汉语树库 + selectivity 控制 + MDL 控制"这一组合没有已发表的等价工作。** 附近只有：mBERT 多语顺带带过中文（无控制）；单一模型的注意力/表层任务探针（无结构探针、无控制）；行为最简对基准（不探表示）。

可行扩展（按性价比排序）：
1. **模型家族对比**（最推荐）：wwm 是否改变句法编码的*几何*而不只是下游分数？
2. **补上缺失的控制**：对中文施加 selectivity + MDL 标准检验本身就是可发表的复现发现。
3. 规模曲线：Zh-Pythia 阶梯上探句法涌现。
4. 现象聚焦：把/被、了/着/过的表示级探针，对接 CLiMP/SLING 中模型行为失败的类目。
