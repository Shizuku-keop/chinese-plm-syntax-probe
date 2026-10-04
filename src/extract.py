"""隐状态抽取：字符→词对齐 + 子词池化，各层表示存为 npz。

对齐：顺序在 '# text =' 中匹配 UD 词形；词间允许空白（SpaceAfter 缺失时，
如外语词 "The Bellagio Group"）。失配时从上次位置起再查找并计数；词形被
normalizer 剥离而无子词覆盖时（如组合音符在 lowercase 规范化下被删），
退取词后首个非特殊子词并计数。
池化：默认取词内首个子词（first），可选 mean；[CLS]/[SEP] 不参与。

用法: python src/extract.py <model_dir> [--pooling first|mean] [--splits train dev test]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from treebank import parse_treebank

DATA_DIR = Path("data/ud_gsdsimp")
OUT_DIR = Path("results/representations")


def align_words(text: str, words: list[str]) -> tuple[list[tuple[int, int]], int]:
    """顺序匹配词形，返回每个词的字符区间 [start, end) 与失配再找回次数。"""
    spans = []
    cursor = 0
    n_fallback = 0
    for w in words:
        if text.startswith(w, cursor):
            start = cursor
        else:
            c = cursor
            while c < len(text) and text[c].isspace():
                c += 1
            if text.startswith(w, c):
                start = c
            else:
                idx = text.find(w, cursor)
                if idx < 0:
                    raise ValueError(f"词形无法在原文中定位: {w!r} @ {text!r}")
                start = idx
                n_fallback += 1
        spans.append((start, start + len(w)))
        cursor = start + len(w)
    return spans, n_fallback


def word_token_indices(
    spans: list[tuple[int, int]], offsets: list[tuple[int, int]]
) -> tuple[list[list[int]], int]:
    """每个词取 offset 与之重叠的非特殊子词下标（特殊符号 offset 为 (0,0)）。

    词被 normalizer 剥离而无重叠子词时，退取词后首个非特殊子词，返回回退次数。
    """
    out = []
    n_fallback = 0
    non_special = [t for t, (s, e) in enumerate(offsets) if (s, e) != (0, 0)]
    for ws, we in spans:
        idx = [t for t in non_special if offsets[t][0] < we and offsets[t][1] > ws]
        if not idx:
            after = [t for t in non_special if offsets[t][0] >= we]
            idx = [after[0] if after else non_special[-1]]
            n_fallback += 1
        out.append(idx)
    return out, n_fallback


@torch.no_grad()
def extract_split(model, tok, sents: list[dict], pooling: str, batch_size: int = 16) -> dict:
    """返回 {'layer_i': (total_words, 768) float16, 'lengths': (n_sents,)}。"""
    n_layers = model.config.num_hidden_layers + 1
    dim = model.config.hidden_size
    lengths = np.array([len(s["words"]) for s in sents], dtype=np.int32)
    total_words = int(lengths.sum())
    layers = [np.zeros((total_words, dim), dtype=np.float16) for _ in range(n_layers)]
    n_fallback = 0
    n_zero_cov = 0
    row = 0
    for b0 in range(0, len(sents), batch_size):
        batch = sents[b0 : b0 + batch_size]
        texts = [s["text"] for s in batch]
        enc = tok(texts, padding=True, return_tensors="pt", return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        hs = model(**enc, output_hidden_states=True).hidden_states
        for b, s in enumerate(batch):
            spans, fb = align_words(s["text"], s["words"])
            n_fallback += fb
            tok_idx, zc = word_token_indices(spans, offsets[b].tolist())
            n_zero_cov += zc
            n = len(s["words"])
            for li, h in enumerate(hs):
                if pooling == "first":
                    word_h = h[b, torch.tensor([ix[0] for ix in tok_idx])]
                else:
                    word_h = torch.stack([h[b, ix].mean(0) for ix in tok_idx])
                layers[li][row : row + n] = word_h.half().numpy()
            row += n
        if (b0 // batch_size) % 50 == 0:
            print(f"  {b0}/{len(sents)} 句...", flush=True)
    assert row == total_words
    out = {f"layer_{i}": layers[i] for i in range(n_layers)}
    out["lengths"] = lengths
    print(f"  失配再找回: {n_fallback} 词, 无子词覆盖回退: {n_zero_cov} 词", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--pooling", choices=["first", "mean"], default="first")
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    model_name = Path(args.model_dir).name + ("" if args.pooling == "first" else f"_{args.pooling}")
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModel.from_pretrained(args.model_dir)
    model.eval()

    for split in args.splits:
        sents = parse_treebank(str(DATA_DIR / f"zh_gsdsimp-ud-{split}.conllu"))
        n_words = sum(len(s["words"]) for s in sents)
        print(f"[{split}] {len(sents)} 句, {n_words} 词", flush=True)
        out = extract_split(model, tok, sents, args.pooling, args.batch_size)
        assert out["lengths"].sum() == n_words, "词数与树库不一致"
        dest = OUT_DIR / model_name
        dest.mkdir(parents=True, exist_ok=True)
        np.savez(dest / f"{split}.npz", **out)
        print(f"[{split}] 已保存 -> {dest / (split + '.npz')}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
