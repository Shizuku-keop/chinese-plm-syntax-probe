"""基线：线性链与均匀随机树的 UUAS / 距离 Spearman（dev/test）。

线性链：词 i 接词 i+1（末词视为根），预测距离为 |i-j|。
随机树：每句 10 棵 Prüfer 序列均匀随机标号树（每 split 各自用种子 0 初始化），
句内对 10 棵取平均后再按句平均。评估口径与 probe.py 一致：
n<2 的句跳过，UUAS 比对无向边，Spearman 取上三角词对并剔除 NaN。
结果打印并写入 results/baselines.json。

用法: python src/baselines.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr

from treebank import parse_treebank

DATA_DIR = Path("data/ud_gsdsimp")
N_RANDOM = 10
SEED = 0


def _random_tree_edges(n: int, rng: np.random.Generator) -> set[frozenset]:
    """Prüfer 序列解码的均匀随机标号树，返回无向边集合。"""
    if n == 2:
        return {frozenset((0, 1))}
    seq = rng.integers(0, n, size=n - 2)
    deg = np.ones(n, dtype=np.int64)
    for x in seq:
        deg[x] += 1
    edges = []
    for x in seq:
        leaf = int(np.flatnonzero(deg == 1)[0])
        edges.append((leaf, int(x)))
        deg[leaf] -= 1
        deg[x] -= 1
    u, v = (int(i) for i in np.flatnonzero(deg == 1))
    edges.append((u, v))
    return {frozenset(e) for e in edges}


def _tree_distances(n: int, edges: set[frozenset]) -> np.ndarray:
    adj = np.zeros((n, n), dtype=np.float64)
    for e in edges:
        i, j = tuple(e)
        adj[i, j] = adj[j, i] = 1.0
    return shortest_path(adj, directed=False)


def _score(pred_edges, pred_dist, gold_edges, gold_triu, iu) -> tuple[float, float | None]:
    uuas = len(pred_edges & gold_edges) / len(gold_edges)
    spr = spearmanr(pred_dist[iu], gold_triu).statistic
    return uuas, (None if np.isnan(spr) else float(spr))


def eval_split(sents: list[dict]) -> dict:
    rng = np.random.default_rng(SEED)
    chain_uuas, chain_spr, rand_uuas, rand_spr = [], [], [], []
    for s in sents:
        heads = s["heads"]
        n = len(heads)
        if n < 2:
            continue
        gold_edges = {frozenset((i, int(h))) for i, h in enumerate(heads) if h >= 0}
        iu = np.triu_indices(n, k=1)
        gold_triu = np.asarray(s["distance"], dtype=np.float64)[iu]

        chain_edges = {frozenset((i, i + 1)) for i in range(n - 1)}
        chain_dist = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
        u, sp = _score(chain_edges, chain_dist, gold_edges, gold_triu, iu)
        chain_uuas.append(u)
        if sp is not None:
            chain_spr.append(sp)

        ru, rs = [], []
        for _ in range(N_RANDOM):
            edges = _random_tree_edges(n, rng)
            u, sp = _score(edges, _tree_distances(n, edges), gold_edges, gold_triu, iu)
            ru.append(u)
            if sp is not None:
                rs.append(sp)
        rand_uuas.append(float(np.mean(ru)))
        if rs:
            rand_spr.append(float(np.mean(rs)))

    return {
        "n_sentences": len(chain_uuas),
        "linear_chain": {"uuas": float(np.mean(chain_uuas)), "spearman": float(np.mean(chain_spr))},
        "random_tree": {"uuas": float(np.mean(rand_uuas)), "spearman": float(np.mean(rand_spr))},
    }


def main() -> None:
    out = {"seed": SEED, "n_random_trees": N_RANDOM, "splits": {}}
    for sp in ["dev", "test"]:
        sents = parse_treebank(str(DATA_DIR / f"zh_gsdsimp-ud-{sp}.conllu"))
        res = eval_split(sents)
        out["splits"][sp] = res
        c, r = res["linear_chain"], res["random_tree"]
        print(
            f"[{sp}] {res['n_sentences']} 句 | "
            f"线性链 UUAS {c['uuas']:.4f} dSpr {c['spearman']:.4f} | "
            f"随机树×{N_RANDOM} UUAS {r['uuas']:.4f} dSpr {r['spearman']:.4f}",
            flush=True,
        )
    dest = Path("results/baselines.json")
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已写入 {dest}")


if __name__ == "__main__":
    main()
