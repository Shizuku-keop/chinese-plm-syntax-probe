"""生成论文图 1–4（中文标签输出 figures/*.png；`--lang en` 输出英文标签 figures/en/*.png）。

图内不再画总标题（标题由论文正文的图题承担），只保留分图 (a)/(b)/(c) 标题。
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

MODELS = [
    ("bert-base-chinese", "BERT", "#1f77b4", "-"),
    ("chinese-roberta-wwm-ext", "RoBERTa-wwm-ext", "#d62728", "--"),
    ("chinese-macbert-base", "MacBERT", "#2ca02c", "-."),
    ("bert-base-multilingual-cased", "mBERT", "#9467bd", ":"),
]
ZH = MODELS[:3]
R = "results"

L = {
    "zh": {
        "mbert": "mBERT（多语）",
        "layer": "层",
        "uuas": "dev UUAS",
        "dspr": "dev 距离 Spearman",
        "hspr_delta": "Δ深度 Spearman（gold − 控制）",
        "uuas_delta": "ΔUUAS（gold − 随机树控制）",
        "compression": "压缩率（均匀编码 / 实际比特）",
        "test_acc": "test 准确率",
        "mdl_compression": "MDL 压缩率",
        "baseline": "随机树基线",
        "first": "（首字）",
        "mean": "（均值）",
        "f1a": "(a) 依存距离可解码性（UUAS）",
        "f1b": "(b) 距离矩阵相关性（Spearman）",
        "f2a": "(a) 深度探针 selectivity",
        "f2b": "(b) 距离探针 selectivity",
        "f3a": "(a) 依存关系标签的 MDL 压缩率",
        "f3b": "(b) 依存关系标签分类准确率",
        "f4a": "(a) 词表示池化的影响：UUAS",
        "f4b": "(b) 词表示池化的影响：MDL",
        "f4c": "(c) 距离探针 selectivity",
        "skip": "图 4(c) 跳过 {m} 均值池化距离曲线（控制数据不完整）",
    },
    "en": {
        "mbert": "mBERT (multilingual)",
        "layer": "Layer",
        "uuas": "dev UUAS",
        "dspr": "dev distance Spearman",
        "hspr_delta": "Δ depth Spearman (gold − control)",
        "uuas_delta": "ΔUUAS (gold − random-tree control)",
        "compression": "compression ratio (uniform / actual bits)",
        "test_acc": "test accuracy",
        "mdl_compression": "MDL compression ratio",
        "baseline": "random-tree baseline",
        "first": " (first-subtoken)",
        "mean": " (mean)",
        "f1a": "(a) Dependency-distance decodability (UUAS)",
        "f1b": "(b) Distance-matrix correlation (Spearman)",
        "f2a": "(a) Depth-probe selectivity",
        "f2b": "(b) Distance-probe selectivity",
        "f3a": "(a) MDL compression ratio for dependency labels",
        "f3b": "(b) Dependency-label classification accuracy",
        "f4a": "(a) Effect of word-representation pooling: UUAS",
        "f4b": "(b) Effect of word-representation pooling: MDL",
        "f4c": "(c) Distance-probe selectivity",
        "skip": "Figure 4(c) skipping {m} mean-pooling distance curves (control data incomplete)",
    },
}


def by_layer(name):
    with open(os.path.join(R, name), encoding="utf-8") as f:
        return {rec["layer"]: rec for rec in json.load(f)["layers"]}


def bundle(model):
    cx = f"control_distance_{model}.json"
    return (by_layer(f"first_probe_{model}.json"),
            by_layer(f"control_depth_{model}.json"),
            by_layer(cx) if os.path.exists(os.path.join(R, cx)) else None,
            by_layer(f"mdl_{model}.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["zh", "en"], default="zh",
                    help="图内标签语言（zh 输出 figures/，en 输出 figures/en/）")
    args = ap.parse_args()
    t = L[args.lang]
    out = "figures" if args.lang == "zh" else os.path.join("figures", "en")
    models = [(m, t["mbert"] if m.endswith("multilingual-cased") else s, c, ls)
              for m, s, c, ls in MODELS]

    os.makedirs(out, exist_ok=True)
    B = {m: bundle(m) for m, _, _, _ in models}
    with open(os.path.join(R, "baselines.json"), encoding="utf-8") as f:
        base = json.load(f)["splits"]["dev"]
    layers = sorted(B[models[0][0]][0])

    # ---- 图 1 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    ax = axes[0]
    for m, short, color, ls in models:
        ax.plot(layers, [B[m][0][l]["distance"]["dev"]["uuas"] for l in layers],
                color=color, ls=ls, marker="o", ms=3, label=short)
    ax.axhline(base["random_tree"]["uuas"], color="gray", lw=1, ls=":", label=t["baseline"])
    ax.set_xlabel(t["layer"]); ax.set_ylabel(t["uuas"]); ax.set_title(t["f1a"])
    ax.axvline(8, color="black", lw=0.6, alpha=0.3)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1]
    for m, short, color, ls in models:
        ax.plot(layers, [B[m][0][l]["distance"]["dev"]["spearman"] for l in layers],
                color=color, ls=ls, marker="o", ms=3, label=short)
    ax.axhline(base["random_tree"]["spearman"], color="gray", lw=1, ls=":", label=t["baseline"])
    ax.set_xlabel(t["layer"]); ax.set_ylabel(t["dspr"]); ax.set_title(t["f1b"])
    ax.axvline(8, color="black", lw=0.6, alpha=0.3)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_syntax_geometry.png"), dpi=200)

    # ---- 图 2 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    ax = axes[0]
    for m, short, color, ls in models:
        ax.plot(layers, [B[m][0][l]["depth"]["dev"]["spearman"] - B[m][1][l]["dev"]["spearman"] for l in layers],
                color=color, ls=ls, marker="o", ms=3, label=short)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel(t["layer"]); ax.set_ylabel(t["hspr_delta"])
    ax.set_title(t["f2a"]); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax = axes[1]
    for m, short, color, ls in models:
        ax.plot(layers, [B[m][0][l]["distance"]["dev"]["uuas"] - B[m][2][l]["dev"]["uuas"] for l in layers],
                color=color, ls=ls, marker="o", ms=3, label=short)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel(t["layer"]); ax.set_ylabel(t["uuas_delta"])
    ax.set_title(t["f2b"]); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_selectivity.png"), dpi=200)

    # ---- 图 3 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    ax = axes[0]
    for m, short, color, ls in models:
        ax.plot(layers, [B[m][3][l]["compression"] for l in layers],
                color=color, ls=ls, marker="o", ms=3, label=short)
    ax.set_xlabel(t["layer"]); ax.set_ylabel(t["compression"])
    ax.set_title(t["f3a"]); ax.axvline(8, color="black", lw=0.6, alpha=0.3)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax = axes[1]
    for m, short, color, ls in models:
        ax.plot(layers, [B[m][3][l]["test_acc"] for l in layers],
                color=color, ls=ls, marker="o", ms=3, label=short)
    ax.set_xlabel(t["layer"]); ax.set_ylabel(t["test_acc"])
    ax.set_title(t["f3b"]); ax.axvline(8, color="black", lw=0.6, alpha=0.3)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_mdl.png"), dpi=200)

    # ---- 图 4：均值池化消融 ----
    mean = {}
    for m, _, _, _ in ZH:
        p, cd, cx, md = bundle(m + "_mean")
        if cx is None or len(cx) != len(p):
            print(t["skip"].format(m=m))
            cx = None
        mean[m] = (p, cd, cx, md)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
    ax = axes[0]
    for m, short, color, _ in ZH:
        ax.plot(layers, [B[m][0][l]["distance"]["dev"]["uuas"] for l in layers],
                color=color, ls="-", marker="o", ms=3, label=short + t["first"])
        ax.plot(layers, [mean[m][0][l]["distance"]["dev"]["uuas"] for l in layers],
                color=color, ls="--", marker="s", ms=3, label=short + t["mean"])
    ax.axvline(8, color="black", lw=0.6, alpha=0.3)
    ax.set_xlabel(t["layer"]); ax.set_ylabel(t["uuas"]); ax.set_title(t["f4a"])
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax = axes[1]
    for m, short, color, _ in ZH:
        ax.plot(layers, [B[m][3][l]["compression"] for l in layers],
                color=color, ls="-", marker="o", ms=3, label=short + t["first"])
        ax.plot(layers, [mean[m][3][l]["compression"] for l in layers],
                color=color, ls="--", marker="s", ms=3, label=short + t["mean"])
    ax.axvline(8, color="black", lw=0.6, alpha=0.3)
    ax.set_xlabel(t["layer"]); ax.set_ylabel(t["mdl_compression"]); ax.set_title(t["f4b"])
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax = axes[2]
    for m, short, color, _ in ZH:
        ax.plot(layers, [B[m][0][l]["distance"]["dev"]["uuas"] - B[m][2][l]["dev"]["uuas"] for l in layers],
                color=color, ls="-", marker="o", ms=3, label=short + t["first"])
        if mean[m][2] is not None:
            ax.plot(layers, [mean[m][0][l]["distance"]["dev"]["uuas"] - mean[m][2][l]["dev"]["uuas"] for l in layers],
                    color=color, ls="--", marker="s", ms=3, label=short + t["mean"])
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel(t["layer"]); ax.set_ylabel(t["uuas_delta"])
    ax.set_title(t["f4c"]); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_pooling_ablation.png"), dpi=200)

    print("wrote:", sorted(os.listdir(out)))


if __name__ == "__main__":
    main()
