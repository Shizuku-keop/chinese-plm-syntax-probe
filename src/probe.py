"""结构探针（Hewitt & Manning 2019）：距离探针与深度探针。

距离探针 d_pred(h_i, h_j) = ||B(h_i - h_j)||^2 对齐金标准树距离（L1）。
由线性性 ||B(h_i - h_j)||^2 = ||Bh_i - Bh_j||^2，先把词投影到 rank 维再取
两两距离，避免在 768 维上构造 n^2 个差向量。
深度探针 ||B h_i||^2 对齐金标准深度（L1）。Adam 按句组批训练。
评估：UUAS（预测距离矩阵跑最小生成树比对无向边）、Spearman（逐句后平均）。
"""
import numpy as np
import torch
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import spearmanr


class DistanceProbe(torch.nn.Module):
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.B = torch.nn.Parameter(torch.randn(rank, dim) * 0.01)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # (b, n, d) -> (b, n, n)
        u = torch.einsum("rd,bnd->bnr", self.B, h)
        diffs = u.unsqueeze(2) - u.unsqueeze(1)  # (b, n, n, rank)
        return (diffs**2).sum(-1)


class DepthProbe(torch.nn.Module):
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.B = torch.nn.Parameter(torch.randn(rank, dim) * 0.01)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # (b, n, d) -> (b, n)
        proj = torch.einsum("rd,bnd->bnr", self.B, h)
        return (proj**2).sum(-1)


def split_sentences(repr: np.ndarray, lengths: np.ndarray) -> list[np.ndarray]:
    """把 (total_words, dim) 按 lengths 切成逐句列表。"""
    cuts = np.concatenate([[0], np.cumsum(lengths)])
    return [repr[a:b] for a, b in zip(cuts[:-1], cuts[1:])]


def make_batches(lengths: np.ndarray, batch_size: int, pair_budget: int, rng) -> list[list[int]]:
    """按句组批：句数上限 batch_size，且批内 n_max^2 * 句数 不超 pair_budget。"""
    order = rng.permutation(len(lengths))
    batches, cur, n_max = [], [], 0
    for i in order:
        new_max = max(n_max, int(lengths[i]))
        if cur and (len(cur) >= batch_size or new_max * new_max * (len(cur) + 1) > pair_budget):
            batches.append(cur)
            cur, n_max = [], 0
            new_max = int(lengths[i])
        cur.append(int(i))
        n_max = new_max
    if cur:
        batches.append(cur)
    return batches


def train_probe(
    probe: torch.nn.Module,
    reprs: list[np.ndarray],
    golds: list[np.ndarray],
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int = 0,
    pair_budget: int = 120_000,
) -> torch.nn.Module:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    lengths = np.array([len(g) for g in golds])
    is_distance = isinstance(probe, DistanceProbe)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for epoch in range(epochs):
        tot_loss, n_batch = 0.0, 0
        for batch in make_batches(lengths, batch_size, pair_budget, rng):
            n_max = max(len(reprs[i]) for i in batch)
            h = torch.zeros(len(batch), n_max, reprs[0].shape[-1])
            gold = torch.zeros(len(batch), n_max, n_max) if is_distance else torch.zeros(len(batch), n_max)
            mask = torch.zeros(len(batch), n_max, dtype=torch.bool)
            for b, i in enumerate(batch):
                n = len(reprs[i])
                h[b, :n] = torch.from_numpy(reprs[i])
                mask[b, :n] = True
                g = torch.from_numpy(np.asarray(golds[i], dtype=np.float32))
                if is_distance:
                    gold[b, :n, :n] = g
                else:
                    gold[b, :n] = g
            pred = probe(h)
            if is_distance:
                pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
                loss = (pred - gold).abs()[pair_mask].mean()
            else:
                loss = (pred - gold).abs()[mask].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item()
            n_batch += 1
        sched.step()
        if epoch % 5 == 4 or epoch == 0:
            print(f"    epoch {epoch + 1}/{epochs}  L1={tot_loss / n_batch:.4f}", flush=True)
    return probe


def _undirected_edges(heads: np.ndarray) -> set[frozenset]:
    return {frozenset((i, int(h))) for i, h in enumerate(heads) if h >= 0}


def eval_distance_probe(
    probe: torch.nn.Module, reprs: list[np.ndarray], heads_list: list[np.ndarray], dists: list[np.ndarray]
) -> dict:
    """逐句 UUAS（最小生成树比对金标准无向边）与距离 Spearman（上三角词对）。"""
    uuas_list, spr_list = [], []
    with torch.no_grad():
        for h_np, heads, gold in zip(reprs, heads_list, dists):
            n = len(heads)
            if n < 2:
                continue
            pred = probe(torch.from_numpy(h_np).unsqueeze(0).float())[0].numpy()
            mst = minimum_spanning_tree(pred).tocoo()
            pred_edges = {frozenset((int(i), int(j))) for i, j in zip(mst.row, mst.col)}
            gold_edges = _undirected_edges(heads)
            uuas_list.append(len(pred_edges & gold_edges) / len(gold_edges))
            iu = np.triu_indices(n, k=1)
            spr = spearmanr(pred[iu], np.asarray(gold, dtype=np.float64)[iu]).statistic
            if not np.isnan(spr):
                spr_list.append(spr)
    return {"uuas": float(np.mean(uuas_list)), "spearman": float(np.mean(spr_list))}


def eval_depth_probe(
    probe: torch.nn.Module, reprs: list[np.ndarray], depths: list[np.ndarray]
) -> dict:
    """逐句深度 Spearman 与根准确率（预测深度最小者应为根）。"""
    spr_list, root_ok = [], []
    with torch.no_grad():
        for h_np, gold in zip(reprs, depths):
            n = len(gold)
            if n < 2:
                continue
            pred = probe(torch.from_numpy(h_np).unsqueeze(0).float())[0].numpy()
            spr = spearmanr(pred, np.asarray(gold, dtype=np.float64)).statistic
            if not np.isnan(spr):
                spr_list.append(spr)
            root_ok.append(int(pred.argmin()) == int(np.argmin(gold)))
    return {"spearman": float(np.mean(spr_list)), "root_acc": float(np.mean(root_ok))}
