"""控制任务探针（Hewitt & Liang 2019）：选择性 = 金标准指标 − 控制指标。

--control depth：在词类型→固定随机深度的控制标签上训练并评估深度探针；
--control distance：在每句一棵固定随机树的距离矩阵上训练并评估距离探针。
超参与主实验一致。每层结束重写 JSON（临时文件原子替换），进程中断时已完成的
层不丢失。若 results/first_probe_<model>.json 存在，每层附一行选择性检查。

用法:
  python src/run_control.py --control depth --model bert-base-chinese
  python src/run_control.py --control distance --layers 0 1 2   # 快速自检
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from control_tasks import depth_control_labels, distance_control_trees
from probe import (
    DepthProbe,
    DistanceProbe,
    eval_depth_probe,
    eval_distance_probe,
    split_sentences,
    train_probe,
)
from run_first_probe import load_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", choices=["depth", "distance"], required=True)
    ap.add_argument("--model", default="bert-base-chinese")
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="输出文件名后缀（多种子鲁棒性实验用，如 _seed1）")
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="只跑这些层（默认全部 13 层）")
    args = ap.parse_args()

    splits = {sp: load_split(args.model, sp) for sp in ["train", "dev", "test"]}
    sents = {sp: v[0] for sp, v in splits.items()}
    if args.control == "depth":
        ctrl = depth_control_labels(sents, seed=args.seed)
    else:
        ctrl = distance_control_trees(sents, seed=args.seed)
    n_layers = len([k for k in splits["train"][1].files if k.startswith("layer_")])
    layers = args.layers if args.layers is not None else list(range(n_layers))
    print(f"模型={args.model} 控制={args.control} rank={args.rank} epochs={args.epochs} "
          f"lr={args.lr} 层={layers}", flush=True)

    dest = Path(f"results/control_{args.control}_{args.model}{args.tag}.json")
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
            "control": args.control,
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

    gold_file = Path(f"results/first_probe_{args.model}.json")
    gold = {}
    if gold_file.exists():
        gold = {r["layer"]: r for r in json.loads(gold_file.read_text(encoding="utf-8"))["layers"]}

    for layer in layers:
        if layer in done:
            continue
        t0 = time.time()
        reprs = {sp: split_sentences(npz[f"layer_{layer}"].astype(np.float32), lengths)
                 for sp, (_, npz, lengths) in splits.items()}
        dim = reprs["train"][0].shape[-1]

        torch.manual_seed(args.seed)
        row = {"layer": layer}
        if args.control == "depth":
            probe = DepthProbe(dim, args.rank)
            train_probe(probe, reprs["train"], ctrl["train"],
                        args.epochs, args.lr, args.batch_size, seed=args.seed)
            for sp in ["train", "dev", "test"]:
                row[sp] = eval_depth_probe(probe, reprs[sp], ctrl[sp])
        else:
            probe = DistanceProbe(dim, args.rank)
            train_probe(probe, reprs["train"], [t["distance"] for t in ctrl["train"]],
                        args.epochs, args.lr, args.batch_size, seed=args.seed)
            for sp in ["train", "dev", "test"]:
                row[sp] = eval_distance_probe(
                    probe, reprs[sp],
                    [t["heads"] for t in ctrl[sp]], [t["distance"] for t in ctrl[sp]])
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        checkpoint()

        g = gold.get(layer)
        if args.control == "depth":
            sel = g["depth"]["dev"]["spearman"] - row["dev"]["spearman"] if g else float("nan")
            print(
                f"layer {layer:2d} | 控制深度 train hSpr {row['train']['spearman']:.3f} "
                f"dev {row['dev']['spearman']:.3f} test {row['test']['spearman']:.3f} "
                f"| {row['seconds']}s",
                flush=True,
            )
            print(
                f"[检查] layer {layer:2d} 控制train记忆应高: hSpr={row['train']['spearman']:.3f} "
                f"| 选择性(dev hSpr)=金标准−控制={sel:+.3f} 应为正",
                flush=True,
            )
        else:
            sel_u = g["distance"]["dev"]["uuas"] - row["dev"]["uuas"] if g else float("nan")
            sel_s = g["distance"]["dev"]["spearman"] - row["dev"]["spearman"] if g else float("nan")
            print(
                f"layer {layer:2d} | 控制距离 train UUAS {row['train']['uuas']:.3f} "
                f"dSpr {row['train']['spearman']:.3f} | dev UUAS {row['dev']['uuas']:.3f} "
                f"dSpr {row['dev']['spearman']:.3f} | test UUAS {row['test']['uuas']:.3f} "
                f"dSpr {row['test']['spearman']:.3f} | {row['seconds']}s",
                flush=True,
            )
            print(
                f"[检查] layer {layer:2d} 选择性(dev UUAS)={sel_u:+.3f} "
                f"选择性(dev dSpr)={sel_s:+.3f} 应为正",
                flush=True,
            )

    print(f"已写入 {dest}")


if __name__ == "__main__":
    main()
