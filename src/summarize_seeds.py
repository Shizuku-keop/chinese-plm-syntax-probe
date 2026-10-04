"""多种子鲁棒性汇总：读 results/*_seed<N>.json（连同 seed 0 的主结果），输出 docs/results-seeds.md。

用法：./.venv/Scripts/python src/summarize_seeds.py
"""
import json
from pathlib import Path

R = Path("results")
SEEDS = [0, 1, 2]
ZH = [
    ("bert-base-chinese", "BERT"),
    ("chinese-roberta-wwm-ext", "RoBERTa-wwm-ext"),
    ("chinese-macbert-base", "MacBERT"),
]


def load(name: str):
    p = R / name
    if not p.exists():
        return None
    return {r["layer"]: r for r in json.loads(p.read_text(encoding="utf-8"))["layers"]}


def f(x, nd=3):
    return f"{x:.{nd}f}"


def cell(probe, ctrl, mdl):
    """返回 (峰值层, dev UUAS, test UUAS, 层8深度Δ, MDL最低层, MDL最低 kbit)，缺数据的项为 None。"""
    if probe is None:
        return None
    pk = max(probe.values(), key=lambda r: r["distance"]["dev"]["uuas"])
    sel8 = (probe[8]["depth"]["dev"]["spearman"] - ctrl[8]["dev"]["spearman"]
            if ctrl and 8 in ctrl else None)
    mk = min(mdl.values(), key=lambda r: r["bits"]) if mdl else None
    return (pk["layer"], pk["distance"]["dev"]["uuas"], pk["distance"]["test"]["uuas"],
            sel8, mk["layer"] if mk else None, mk["bits"] / 1000 if mk else None)


def main() -> None:
    out = ["# 多种子鲁棒性（seed 0 / 1 / 2，三个中文单语模型）\n",
           "所有数字由 `src/summarize_seeds.py` 从 `results/*_seed{1,2}.json` 与 seed 0 的主结果现算。\n",
           "| 模型 | 种子 | UUAS 峰值层 | dev UUAS | test UUAS | 层 8 深度 Δ | MDL 最低层 | MDL 最低 kbit |",
           "|---|---|---|---|---|---|---|---|"]
    data = {}
    for m, s in ZH:
        for sd in SEEDS:
            tag = "" if sd == 0 else f"_seed{sd}"
            rows = cell(load(f"first_probe_{m}{tag}.json"),
                        load(f"control_depth_{m}{tag}.json"),
                        load(f"mdl_{m}{tag}.json"))
            data[(m, sd)] = rows
            if rows is None:
                out.append(f"| {s} | {sd} | （缺数据） | | | | | |")
                continue
            pk, du, tu, sel, ml, mk = rows
            out.append("| " + " | ".join([
                s, str(sd), str(pk), f(du), f(tu),
                f(sel) if sel is not None else "—",
                str(ml) if ml is not None else "—",
                f(mk, 1) if mk is not None else "—",
            ]) + " |")
    out.append("")

    out.append("## 种子间的稳定性\n")
    out.append("| 模型 | 峰值层（各 seed） | dev UUAS 各 seed | dev UUAS 极差（pp） | 层 8 深度 Δ 各 seed | Δ 极差（pp） |")
    out.append("|---|---|---|---|---|---|")
    for m, s in ZH:
        rows = [data[(m, sd)] for sd in SEEDS]
        have = [r for r in rows if r]
        if not have:
            out.append(f"| {s} | （缺数据） | | | | |")
            continue
        pks = " / ".join(str(r[0]) for r in rows if r)
        dus = " / ".join(f(r[1]) for r in rows if r)
        rng = (max(r[1] for r in have) - min(r[1] for r in have)) * 100
        sels = [r[3] for r in have if r[3] is not None]
        seltxt = " / ".join(f(x) for x in sels) if sels else "—"
        selrng = (max(sels) - min(sels)) * 100 if len(sels) > 1 else None
        out.append("| " + " | ".join([
            s, pks, dus, f"{rng:.1f}",
            seltxt, f"{selrng:.1f}" if selrng is not None else "—",
        ]) + " |")
    out.append("")

    out.append("## 模型间排序在各 seed 上是否稳定（dev UUAS 峰值层）\n")
    pairs = [("chinese-macbert-base", "bert-base-chinese", "MacBERT − BERT"),
             ("chinese-roberta-wwm-ext", "bert-base-chinese", "RoBERTa − BERT"),
             ("chinese-macbert-base", "chinese-roberta-wwm-ext", "MacBERT − RoBERTa")]
    for a, b, label in pairs:
        diffs = []
        for sd in SEEDS:
            ra, rb = data.get((a, sd)), data.get((b, sd))
            if ra and rb:
                diffs.append(f"seed {sd}: {(ra[1] - rb[1]) * 100:+.1f} pp")
        out.append(f"- {label}：" + ("；".join(diffs) if diffs else "（缺数据）"))
    out.append("")

    text = "\n".join(out) + "\n"
    Path("docs").mkdir(exist_ok=True)
    Path("docs/results-seeds.md").write_text(text, encoding="utf-8")
    print("已写入 docs/results-seeds.md")


if __name__ == "__main__":
    main()
