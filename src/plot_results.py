"""生成论文图 1–4（输出 figures/*.png）。"""
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
    ("bert-base-multilingual-cased", "mBERT（多语）", "#9467bd", ":"),
]
ZH = MODELS[:3]
R = "results"
OUT = "figures"


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
    os.makedirs(OUT, exist_ok=True)
    B = {m: bundle(m) for m, _, _, _ in MODELS}
    with open(os.path.join(R, "baselines.json"), encoding="utf-8") as f:
        base = json.load(f)["splits"]["dev"]
    layers = sorted(B[MODELS[0][0]][0])

    # ---- 图 1 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for m, short, color, ls in MODELS:
        ax.plot(layers, [B[m][0][l]["distance"]["dev"]["uuas"] for l in layers],
                color=color, ls=ls, marker="o", ms=3, label=short)
    ax.axhline(base["random_tree"]["uuas"], color="gray", lw=1, ls=":", label="随机树基线")
    ax.set_xlabel("层"); ax.set_ylabel("dev UUAS"); ax.set_title("(a) 依存距离可解码性（UUAS）")
    ax.axvline(8, color="black", lw=0.6, alpha=0.3)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1]
    for m, short, color, ls in MODELS:
        ax.plot(layers, [B[m][0][l]["distance"]["dev"]["spearman"] for l in layers],
                color=color, ls=ls, marker="o", ms=3, label=short)
    ax.axhline(base["random_tree"]["spearman"], color="gray", lw=1, ls=":", label="随机树基线")
    ax.set_xlabel("层"); ax.set_ylabel("dev 距离 Spearman"); ax.set_title("(b) 距离矩阵相关性（Spearman）")
    ax.axvline(8, color="black", lw=0.6, alpha=0.3)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("图 1  四个中文相关 PLM 的句法几何随层变化（dev 500 句，rank 128）", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_syntax_geometry.png"), dpi=200)

    # ---- 图 2 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for m, short, color, ls in MODELS:
        ax.plot(layers, [B[m][0][l]["depth"]["dev"]["spearman"] - B[m][1][l]["dev"]["spearman"] for l in layers],
                color=color, ls=ls, marker="o", ms=3, label=short)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("层"); ax.set_ylabel("Δ深度 Spearman（gold − 控制）")
    ax.set_title("(a) 深度探针 selectivity"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax = axes[1]
    for m, short, color, ls in MODELS:
        ax.plot(layers, [B[m][0][l]["distance"]["dev"]["uuas"] - B[m][2][l]["dev"]["uuas"] for l in layers],
                color=color, ls=ls, marker="o", ms=3, label=short)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("层"); ax.set_ylabel("ΔUUAS（gold − 随机树控制）")
    ax.set_title("(b) 距离探针 selectivity"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("图 2  控制任务下的 selectivity：负值 = 探针在拟合控制任务而非句法", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_selectivity.png"), dpi=200)

    # ---- 图 3 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for m, short, color, ls in MODELS:
        ax.plot(layers, [B[m][3][l]["compression"] for l in layers],
                color=color, ls=ls, marker="o", ms=3, label=short)
    ax.set_xlabel("层"); ax.set_ylabel("压缩率（均匀编码 / 实际比特）")
    ax.set_title("(a) 依存关系标签的 MDL 压缩率"); ax.axvline(8, color="black", lw=0.6, alpha=0.3)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax = axes[1]
    for m, short, color, ls in MODELS:
        ax.plot(layers, [B[m][3][l]["test_acc"] for l in layers],
                color=color, ls=ls, marker="o", ms=3, label=short)
    ax.set_xlabel("层"); ax.set_ylabel("test 准确率")
    ax.set_title("(b) 依存关系标签分类准确率"); ax.axvline(8, color="black", lw=0.6, alpha=0.3)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("图 3  MDL 检验：45 类依存关系标签的在线编码长度", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_mdl.png"), dpi=200)

    # ---- 图 4：均值池化消融 ----
    mean = {}
    for m, _, _, _ in ZH:
        p, cd, cx, md = bundle(m + "_mean")
        if cx is None or len(cx) != len(p):
            print(f"图 4(c) 跳过 {m} 均值池化距离曲线（控制数据不完整）")
            cx = None
        mean[m] = (p, cd, cx, md)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    for m, short, color, _ in ZH:
        ax.plot(layers, [B[m][0][l]["distance"]["dev"]["uuas"] for l in layers],
                color=color, ls="-", marker="o", ms=3, label=f"{short}（首字）")
        ax.plot(layers, [mean[m][0][l]["distance"]["dev"]["uuas"] for l in layers],
                color=color, ls="--", marker="s", ms=3, label=f"{short}（均值）")
    ax.axvline(8, color="black", lw=0.6, alpha=0.3)
    ax.set_xlabel("层"); ax.set_ylabel("dev UUAS"); ax.set_title("(a) 词表示池化的影响：UUAS")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax = axes[1]
    for m, short, color, _ in ZH:
        ax.plot(layers, [B[m][3][l]["compression"] for l in layers],
                color=color, ls="-", marker="o", ms=3, label=f"{short}（首字）")
        ax.plot(layers, [mean[m][3][l]["compression"] for l in layers],
                color=color, ls="--", marker="s", ms=3, label=f"{short}（均值）")
    ax.axvline(8, color="black", lw=0.6, alpha=0.3)
    ax.set_xlabel("层"); ax.set_ylabel("MDL 压缩率"); ax.set_title("(b) 词表示池化的影响：MDL")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax = axes[2]
    for m, short, color, _ in ZH:
        ax.plot(layers, [B[m][0][l]["distance"]["dev"]["uuas"] - B[m][2][l]["dev"]["uuas"] for l in layers],
                color=color, ls="-", marker="o", ms=3, label=f"{short}（首字）")
        if mean[m][2] is not None:
            ax.plot(layers, [mean[m][0][l]["distance"]["dev"]["uuas"] - mean[m][2][l]["dev"]["uuas"] for l in layers],
                    color=color, ls="--", marker="s", ms=3, label=f"{short}（均值）")
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("层"); ax.set_ylabel("ΔUUAS（gold − 随机树控制）")
    ax.set_title("(c) 距离探针 selectivity"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle("图 4  均值池化消融：峰值层不变，绝对值整体上移", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_pooling_ablation.png"), dpi=200)

    print("wrote:", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
