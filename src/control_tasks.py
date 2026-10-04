"""控制任务（Hewitt & Liang 2019）：为选择性实验生成控制标签与控制树。

深度控制：每个词类型（UD 词形）固定一个随机深度，均匀取自首个 split（train）
观测到的深度范围 [0, max_depth]；类型表按 splits 顺序逐句逐词首次遇到时从同一
种子流抽取，故训练集未见的类型也有固定标签。探针可靠记忆词类型得分，控制任务
高分不反映结构；选择性 = 金标准任务指标 − 控制任务指标。

距离控制（设计选择）：不按词类型条件化——整棵树并非词级标签，逐类型条件化无
标准做法。每句只按句长生成一棵固定的均匀随机标号树（Prüfer 序列），根固定为
词 0；splits 顺序逐句从同一种子流抽取，保证确定性。
"""
from collections import deque

import numpy as np
from scipy.sparse.csgraph import shortest_path


def depth_control_labels(splits: dict[str, list[dict]], seed: int = 0) -> dict[str, list[np.ndarray]]:
    """逐句控制深度标签。splits 的首个键须为 train（深度范围取自它）。"""
    first = next(iter(splits))
    max_depth = int(max(int(s["depth"].max()) for s in splits[first]))
    rng = np.random.default_rng(seed)
    table: dict[str, int] = {}
    out = {}
    for name, sents in splits.items():
        labels = []
        for s in sents:
            lab = np.empty(len(s["words"]), dtype=np.int64)
            for i, w in enumerate(s["words"]):
                if w not in table:
                    table[w] = int(rng.integers(0, max_depth + 1))
                lab[i] = table[w]
            labels.append(lab)
        out[name] = labels
    return out


def _random_heads(n: int, rng: np.random.Generator) -> np.ndarray:
    """Prüfer 序列均匀随机标号树，以词 0 为根，返回 0-based 父节点（根 = -1）。"""
    if n == 1:
        return np.array([-1], dtype=np.int32)
    if n == 2:
        return np.array([-1, 0], dtype=np.int32)
    seq = rng.integers(0, n, size=n - 2)
    deg = np.ones(n, dtype=np.int64)
    for x in seq:
        deg[x] += 1
    adj: list[list[int]] = [[] for _ in range(n)]
    for x in seq:
        leaf = int(np.flatnonzero(deg == 1)[0])
        adj[leaf].append(int(x))
        adj[int(x)].append(leaf)
        deg[leaf] -= 1
        deg[x] -= 1
    u, v = (int(i) for i in np.flatnonzero(deg == 1))
    adj[u].append(v)
    adj[v].append(u)
    heads = np.full(n, -2, dtype=np.int32)
    heads[0] = -1
    q = deque([0])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if heads[v] == -2:
                heads[v] = u
                q.append(v)
    assert (heads > -2).all(), "随机树应连通"
    return heads


def _tree_distances(heads: np.ndarray) -> np.ndarray:
    n = len(heads)
    adj = np.zeros((n, n), dtype=np.float64)
    for i, h in enumerate(heads):
        if h >= 0:
            adj[i, h] = adj[h, i] = 1.0
    return shortest_path(adj, directed=False)


def distance_control_trees(splits: dict[str, list[dict]], seed: int = 0) -> dict[str, list[dict]]:
    """逐句一棵固定随机树，返回 {split: [{"heads", "distance"}]}，顺序与输入一致。"""
    rng = np.random.default_rng(seed)
    out = {}
    for name, sents in splits.items():
        trees = []
        for s in sents:
            heads = _random_heads(len(s["words"]), rng)
            trees.append({"heads": heads, "distance": _tree_distances(heads)})
        out[name] = trees
    return out
