"""首个探针实验：bert-base-chinese 13 层 × 距离/深度结构探针。

对每层隐状态在 train 上训练距离探针与深度探针，在 dev/test 上报告
UUAS、距离 Spearman、深度 Spearman（与根准确率）。每层结束后重写
JSON（临时文件原子替换），进程中断时已完成的层不丢失。

用法:
  python src/run_first_probe.py                  # 全部 13 层
  python src/run_first_probe.py --overfit64      # 过拟合自检：同 64 句训练+评估
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from probe import (
    DepthProbe,
    DistanceProbe,
    eval_depth_probe,
    eval_distance_probe,
    split_sentences,
    train_probe,
)
from treebank import parse_treebank

DATA_DIR = Path("data/ud_gsdsimp")
REPR_DIR = Path("results/representations")


def load_split(model_name: str, split: str):
    sents = parse_treebank(str(DATA_DIR / f"zh_gsdsimp-ud-{split}.conllu"))
    npz = np.load(REPR_DIR / model_name / f"{split}.npz")
    lengths = npz["lengths"]
    assert int(lengths.sum()) == sum(len(s["words"]) for s in sents), "npz 词数与树库不一致"
    return sents, npz, lengths


def run_overfit64(args) -> None:
    """距离探针在同 64 句上训练并评估，Spearman 应 > 0.9（证明能拟合信号）。"""
    sents, npz, lengths = load_split(args.model, "train")
    sents = sents[:64]
    reprs = split_sentences(npz[f"layer_{args.layer}"].astype(np.float32), lengths)[:64]
    dists = [s["distance"] for s in sents]
    heads = [s["heads"] for s in sents]
    torch.manual_seed(0)
    probe = DistanceProbe(reprs[0].shape[-1], args.rank)
    train_probe(probe, reprs, dists, epochs=args.epochs, lr=args.lr, batch_size=8, seed=0)
    res = eval_distance_probe(probe, reprs, heads, dists)
    print(f"overfit64 layer={args.layer}: UUAS={res['uuas']:.4f} Spearman={res['spearman']:.4f}")
    ok = res["spearman"] > 0.9
    print("OVERFIT GATE:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-chinese")
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="输出文件名后缀（多种子鲁棒性实验用，如 _seed1）")
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--overfit64", action="store_true")
    args = ap.parse_args()
    if args.overfit64:
        run_overfit64(args)

    splits = {sp: load_split(args.model, sp) for sp in ["train", "dev", "test"]}
    n_layers = len([k for k in splits["train"][1].files if k.startswith("layer_")])
    print(f"模型={args.model} rank={args.rank} epochs={args.epochs} lr={args.lr} 层数={n_layers}")

    dest = Path(f"results/first_probe_{args.model}{args.tag}.json")
    dest.parent.mkdir(exist_ok=True)
    if dest.exists():
        rows = json.loads(dest.read_text(encoding="utf-8"))["layers"]
        done = {r["layer"] for r in rows}
        if done:
            print(f"断点续跑：跳过已完成层 {sorted(done)}")
    else:
        rows, done = [], set()

    def checkpoint() -> None:
        out = {
            "model": args.model,
            "rank": args.rank,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "layers": rows,
        }
        tmp = dest.with_name(dest.name + ".tmp")
        tmp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(dest)

    for layer in range(n_layers):
        if layer in done:
            continue
        t0 = time.time()
        per = {}
        for sp, (sents, npz, lengths) in splits.items():
            per[sp] = {
                "repr": split_sentences(npz[f"layer_{layer}"].astype(np.float32), lengths),
                "heads": [s["heads"] for s in sents],
                "dist": [s["distance"] for s in sents],
                "depth": [s["depth"] for s in sents],
            }
        dim = per["train"]["repr"][0].shape[-1]

        torch.manual_seed(args.seed)
        dprobe = DistanceProbe(dim, args.rank)
        train_probe(dprobe, per["train"]["repr"], per["train"]["dist"],
                    args.epochs, args.lr, args.batch_size, seed=args.seed)
        torch.manual_seed(args.seed)
        hprobe = DepthProbe(dim, args.rank)
        train_probe(hprobe, per["train"]["repr"], per["train"]["depth"],
                    args.epochs, args.lr, args.batch_size, seed=args.seed)

        row = {"layer": layer, "distance": {}, "depth": {}}
        for sp in ["dev", "test"]:
            row["distance"][sp] = eval_distance_probe(
                dprobe, per[sp]["repr"], per[sp]["heads"], per[sp]["dist"])
            row["depth"][sp] = eval_depth_probe(hprobe, per[sp]["repr"], per[sp]["depth"])
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        checkpoint()
        d, p = row["distance"], row["depth"]
        print(
            f"layer {layer:2d} | dev UUAS {d['dev']['uuas']:.3f} dSpr {d['dev']['spearman']:.3f} "
            f"| test UUAS {d['test']['uuas']:.3f} dSpr {d['test']['spearman']:.3f} "
            f"| dev hSpr {p['dev']['spearman']:.3f} root {p['dev']['root_acc']:.3f} "
            f"| test hSpr {p['test']['spearman']:.3f} root {p['test']['root_acc']:.3f} "
            f"| {row['seconds']}s",
            flush=True,
        )

    print(f"已写入 {dest}")


if __name__ == "__main__":
    main()
