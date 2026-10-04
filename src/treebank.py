"""解析 UD CoNLL-U 树库：词序列、0-based 父节点、树距离矩阵、节点深度与依存关系标签。

距离矩阵与深度均保留标点词（UD 标点计入索引），故命名为 *_incl_punct。
"""
from collections import deque

import numpy as np
from conllu import parse_incr


def _bfs(adj: list[list[int]], start: int) -> np.ndarray:
    n = len(adj)
    dist = np.full(n, -1, dtype=np.int32)
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def _distances_and_depths(heads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """由 0-based 父节点（根为 -1）求无向最短路距离矩阵与各词到根的深度。"""
    n = len(heads)
    adj: list[list[int]] = [[] for _ in range(n)]
    for i, h in enumerate(heads):
        if h >= 0:
            adj[i].append(h)
            adj[h].append(i)
    dist_incl_punct = np.stack([_bfs(adj, s) for s in range(n)])
    root = int(np.flatnonzero(heads < 0)[0])
    depth_incl_punct = _bfs(adj, root)
    assert (dist_incl_punct >= 0).all(), "存在不连通的词（树应为连通图）"
    return dist_incl_punct, depth_incl_punct


def parse_treebank(path: str) -> list[dict]:
    """返回句子列表。每句：
    words  词形列表；heads  0-based 父节点（根 = -1）；deprel  依存关系标签列表；
    distance  (n, n) 无向树距离（含标点）；depth  (n,) 到根距离（含标点，根 = 0）；
    text   '# text =' 原文。
    """
    sents = []
    with open(path, encoding="utf-8") as f:
        for sent in parse_incr(f):
            words = [t["form"] for t in sent]
            heads = np.array([t["head"] - 1 for t in sent], dtype=np.int32)
            dist, depth = _distances_and_depths(heads)
            sents.append(
                {
                    "words": words,
                    "heads": heads,
                    "deprel": [t["deprel"] for t in sent],
                    "distance": dist,
                    "depth": depth,
                    "text": sent.metadata["text"],
                }
            )
    return sents


if __name__ == "__main__":
    import sys

    for path in sys.argv[1:]:
        sents = parse_treebank(path)
        n_words = sum(len(s["words"]) for s in sents)
        sym = all(np.array_equal(s["distance"], s["distance"].T) for s in sents[:200])
        diag = all(not s["distance"].diagonal().any() for s in sents[:200])
        print(
            f"{path}: {len(sents)} 句, {n_words} 词, "
            f"对称(前200句)={sym}, 零对角(前200句)={diag}"
        )
