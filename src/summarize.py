"""把 results/*.json 汇总成 docs/results-tables.md（论文用表 1–8）。"""
import json
import os

ZH = [
    ("bert-base-chinese", "BERT"),
    ("chinese-roberta-wwm-ext", "RoBERTa-wwm-ext"),
    ("chinese-macbert-base", "MacBERT"),
]
MBERT = ("bert-base-multilingual-cased", "mBERT")
R = "results"


def load(name):
    with open(os.path.join(R, name), encoding="utf-8") as f:
        return json.load(f)


def by_layer(obj):
    return {r["layer"]: r for r in obj["layers"]}


def maybe(name):
    path = os.path.join(R, name)
    return by_layer(load(name)) if os.path.exists(path) else None


def probe_bundle(model):
    return (
        by_layer(load(f"first_probe_{model}.json")),
        by_layer(load(f"control_depth_{model}.json")),
        maybe(f"control_distance_{model}.json"),
        by_layer(load(f"mdl_{model}.json")),
    )


def f(x, n=3):
    return f"{x:.{n}f}"


def peak_uuas(p):
    return max(p.values(), key=lambda r: r["distance"]["dev"]["uuas"])


def peak_hspr(p):
    return max(p.values(), key=lambda r: r["depth"]["dev"]["spearman"])


def mdl_min(m):
    return min(m.values(), key=lambda r: r["bits"])


def main():
    P = {m: probe_bundle(m) for m, _ in ZH}
    M = probe_bundle(MBERT[0])
    n_layers = len(P[ZH[0][0]][0])
    base = load("baselines.json")
    dev, test = base["splits"]["dev"], base["splits"]["test"]

    out = ["# 结果汇总表（自动生成，来源 results/*.json）\n"]
    out.append(f"- 层数：{n_layers}（0 = 输入嵌入输出，12 = 最后一层）")
    out.append(f"- 基线 dev：线性链 UUAS {f(dev['linear_chain']['uuas'])} / Spr {f(dev['linear_chain']['spearman'])}；"
               f"随机树 UUAS {f(dev['random_tree']['uuas'])} / Spr {f(dev['random_tree']['spearman'])}")
    out.append(f"- 基线 test：线性链 {f(test['linear_chain']['uuas'])} / {f(test['linear_chain']['spearman'])}；"
               f"随机树 {f(test['random_tree']['uuas'])} / {f(test['random_tree']['spearman'])}\n")

    out.append("## 表 1 dev 结构探针逐层指标（首字池化）\n")
    out.append("| 层 | " + " | ".join(
        f"{s} UUAS | {s} dSpr | {s} hSpr | {s} 根" for _, s in ZH) + " |")
    out.append("|" + "---|" * (1 + 4 * len(ZH)))
    for l in range(n_layers):
        row = [str(l)]
        for m, _ in ZH:
            p = P[m][0][l]
            row += [f(p["distance"]["dev"]["uuas"]), f(p["distance"]["dev"]["spearman"]),
                    f(p["depth"]["dev"]["spearman"]), f(p["depth"]["dev"]["root_acc"])]
        out.append("| " + " | ".join(row) + " |")
    out.append("")

    out.append("## 表 2 test UUAS / 距离 Spearman\n")
    out.append("| 层 | " + " | ".join(f"{s} UUAS | {s} dSpr" for _, s in ZH) + " |")
    out.append("|" + "---|" * (1 + 2 * len(ZH)))
    for l in range(n_layers):
        row = [str(l)]
        for m, _ in ZH:
            row += [f(P[m][0][l]["distance"]["test"]["uuas"]), f(P[m][0][l]["distance"]["test"]["spearman"])]
        out.append("| " + " | ".join(row) + " |")
    out.append("")

    out.append("## 表 3 深度 selectivity（gold 深度 Spearman − 控制深度 Spearman，dev）\n")
    out.append("| 层 | " + " | ".join(f"{s} Δ | {s} 控制" for _, s in ZH) + " |")
    out.append("|" + "---|" * (1 + 2 * len(ZH)))
    for l in range(n_layers):
        row = [str(l)]
        for m, _ in ZH:
            g, c = P[m][0][l]["depth"]["dev"]["spearman"], P[m][1][l]["dev"]["spearman"]
            row += [f(g - c), f(c)]
        out.append("| " + " | ".join(row) + " |")
    out.append("")

    out.append("## 表 4 距离 selectivity（gold UUAS − 随机树控制 UUAS，dev）\n")
    out.append("| 层 | " + " | ".join(f"{s} Δ | {s} 控制" for _, s in ZH) + " |")
    out.append("|" + "---|" * (1 + 2 * len(ZH)))
    for l in range(n_layers):
        row = [str(l)]
        for m, _ in ZH:
            g, c = P[m][0][l]["distance"]["dev"]["uuas"], P[m][2][l]["dev"]["uuas"]
            row += [f(g - c), f(c)]
        out.append("| " + " | ".join(row) + " |")
    out.append("")

    m0 = load("mdl_bert-base-chinese.json")
    out.append("## 表 5 MDL（deprel 分类在线编码长度）\n")
    out.append(f"- 类别数 {m0['n_classes']}，均匀编码 {m0['uniform_bits']:,.0f} bit（压缩率 = 均匀 / 实际）\n")
    out.append("| 层 | " + " | ".join(f"{s} kbit | {s} 压缩率 | {s} acc" for _, s in ZH) + " |")
    out.append("|" + "---|" * (1 + 3 * len(ZH)))
    for l in range(n_layers):
        row = [str(l)]
        for m, _ in ZH:
            r = P[m][3][l]
            row += [f"{r['bits']/1000:,.1f}", f(r["compression"], 2), f(r["test_acc"])]
        out.append("| " + " | ".join(row) + " |")
    out.append("")

    out.append("## 表 6 mBERT（bert-base-multilingual-cased）逐层指标\n")
    out.append("| 层 | dev UUAS | dev dSpr | dev hSpr | test UUAS | test dSpr | 深度 Δ（dev） | 距离 Δ（dev） | 距离控制 UUAS |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for l in range(n_layers):
        p, cd, cx, _ = M
        g_d, c_d = p[l]["depth"]["dev"]["spearman"], cd[l]["dev"]["spearman"]
        g_x = p[l]["distance"]["dev"]["uuas"]
        out.append("| " + " | ".join([str(l), f(g_x), f(p[l]["distance"]["dev"]["spearman"]),
                                      f(g_d), f(p[l]["distance"]["test"]["uuas"]), f(p[l]["distance"]["test"]["spearman"]),
                                      f(g_d - c_d), f(g_x - cx[l]["dev"]["uuas"]), f(cx[l]["dev"]["uuas"])]) + " |")
    out.append("")
    mb = mdl_min(M[3])
    out.append(f"- mBERT MDL 最低：层 {mb['layer']}，{mb['bits']/1000:,.1f} kbit（压缩 {mb['compression']:.2f}×，acc {mb['test_acc']:.3f}）\n")

    out.append("## 表 7 四模型峰值对比\n")
    out.append("| 模型 | UUAS 峰值层 | dev UUAS | dev dSpr | test UUAS | test dSpr | 深度峰值层 | dev hSpr | 层8 深度 Δ | MDL 最低层 | MDL 最低 kbit |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    rows = [(m, s, *P[m]) for m, s in ZH] + [(MBERT[0], MBERT[1], *M)]
    for m, s, p, cd, cx, md in rows:
        pk, dk, mk = peak_uuas(p), peak_hspr(p), mdl_min(md)
        sel = p[8]["depth"]["dev"]["spearman"] - cd[8]["dev"]["spearman"]
        out.append("| " + " | ".join([s, str(pk["layer"]), f(pk["distance"]["dev"]["uuas"]), f(pk["distance"]["dev"]["spearman"]),
                                      f(pk["distance"]["test"]["uuas"]), f(pk["distance"]["test"]["spearman"]),
                                      str(dk["layer"]), f(dk["depth"]["dev"]["spearman"]), f(sel, 3),
                                      str(mk["layer"]), f"{mk['bits']/1000:,.1f}"]) + " |")
    out.append("")

    out.append("## 表 8 均值池化消融（与首字池化对比）\n")
    out.append("| 模型 | 池化 | UUAS 峰值层 | dev UUAS | dev dSpr | test UUAS | 层8 深度 Δ | MDL 最低层 | MDL 最低 kbit |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for m, s in ZH:
        for tag, name in [("首字", m), ("均值", m + "_mean")]:
            p, cd, cx, md = probe_bundle(name)
            pk, mk = peak_uuas(p), mdl_min(md)
            sel = p[8]["depth"]["dev"]["spearman"] - cd[8]["dev"]["spearman"]
            out.append("| " + " | ".join([s, tag, str(pk["layer"]), f(pk["distance"]["dev"]["uuas"]),
                                          f(pk["distance"]["dev"]["spearman"]), f(pk["distance"]["test"]["uuas"]),
                                          f(sel), str(mk["layer"]), f"{mk['bits']/1000:,.1f}"]) + " |")
    out.append("")
    for m, s in ZH:
        cx = by_layer(load(f"control_distance_{m}_mean.json")) if os.path.exists(
            os.path.join(R, f"control_distance_{m}_mean.json")) else None
        if cx and len(cx) == n_layers:
            p = by_layer(load(f"first_probe_{m}_mean.json"))
            out.append(f"- {s} 均值池化距离控制：dev UUAS 层 0–12 范围 "
                       f"{min(r['dev']['uuas'] for r in cx.values()):.3f}–{max(r['dev']['uuas'] for r in cx.values()):.3f}；"
                       f"层 8 距离 ΔUUAS {p[8]['distance']['dev']['uuas'] - cx[8]['dev']['uuas']:.3f}")
    out.append("")

    out.append("### 主线探针耗时（CPU，13 层合计）\n")
    for m, s in ZH + [MBERT]:
        p = by_layer(load(f"first_probe_{m}.json"))
        out.append(f"- {s}：{sum(r.get('seconds', 0) for r in p.values())/60:.1f} 分钟")
    out.append("")

    text = "\n".join(out) + "\n"
    os.makedirs("docs", exist_ok=True)
    with open("docs/results-tables.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print("已写入 docs/results-tables.md")


if __name__ == "__main__":
    main()
