"""MDL 探针实验：<model> 13 层 × deprel 分类在线码长（Voita & Titov 2020）。

标签集取自 train split（含语言特定子类，如 case:loc）；test 中未见标签映射为
-1（最终探针必错，仅影响参考准确率）。码长只在 train 上累计。每层结束后重写
JSON（临时文件原子替换），进程中断时已完成的层不丢失。

用法:
  python src/run_mdl.py --model bert-base-chinese
  python src/run_mdl.py --layers 0 8   # 快速自检
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from mdl import PORTIONS, run_layer
from run_first_probe import load_split


def deprel_labels(sents: list[dict], vocab: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """逐词 deprel 索引（按句序展平）。vocab 为空时取自本 split 的排序标签集。"""
    if vocab is None:
        vocab = sorted({d for s in sents for d in s["deprel"]})
    idx = {d: i for i, d in enumerate(vocab)}
    y = np.array([idx.get(d, -1) for s in sents for d in s["deprel"]], dtype=np.int64)
    return y, vocab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-chinese")
    ap.add_argument("--epochs", type=int, default=5, help="每个前缀块训练的 epoch 数（固定，无早停）")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="输出文件名后缀（多种子鲁棒性实验用，如 _seed1）")
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="只跑这些层（默认全部 13 层）")
    args = ap.parse_args()

    splits = {sp: load_split(args.model, sp) for sp in ["train", "dev", "test"]}
    y_train, vocab = deprel_labels(splits["train"][0])
    y_test, _ = deprel_labels(splits["test"][0], vocab)
    n_classes = len(vocab)
    n_unseen = int((y_test < 0).sum())
    n_layers = len([k for k in splits["train"][1].files if k.startswith("layer_")])
    layers = args.layers if args.layers is not None else list(range(n_layers))
    uniform_bits = len(y_train) * float(np.log2(n_classes))
    print(f"模型={args.model} 类别数={n_classes} train词={len(y_train)} test未见标签={n_unseen} "
          f"均匀码长={uniform_bits:.0f}bit epochs/前缀={args.epochs} 层={layers}", flush=True)

    dest = Path(f"results/mdl_{args.model}{args.tag}.json")
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
            "task": "deprel",
            "n_classes": n_classes,
            "labels": vocab,
            "portions": PORTIONS,
            "epochs_per_prefix": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "uniform_bits": uniform_bits,
            "layers": rows,
        }
        tmp = dest.with_name(dest.name + ".tmp")
        tmp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(dest)

    train_npz, test_npz = splits["train"][1], splits["test"][1]
    for layer in layers:
        if layer in done:
            continue
        t0 = time.time()
        X_train = train_npz[f"layer_{layer}"].astype(np.float32)
        X_test = test_npz[f"layer_{layer}"].astype(np.float32)
        row = {"layer": layer}
        row.update(run_layer(X_train, y_train, X_test, y_test, n_classes,
                             epochs=args.epochs, lr=args.lr,
                             batch_size=args.batch_size, seed=args.seed))
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        checkpoint()
        print(
            f"layer {layer:2d} | MDL {row['bits']:.0f} bit | 压缩率 {row['compression']:.3f} "
            f"| test acc {row['test_acc']:.3f} | {row['seconds']}s",
            flush=True,
        )

    print(f"已写入 {dest}")


if __name__ == "__main__":
    main()
