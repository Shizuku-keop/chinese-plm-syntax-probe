"""MDL 探针（Voita & Titov 2020）：以在线码长衡量表征传达 deprel 标签的难易。

把 train 全部词按固定随机序切成 K 个前缀块（portions = 0.1, 0.2, ..., 1.0）：
首块标签用均匀码（log2(n_classes) bit/词）；其后每块先用之前全部数据训练线性
探针（dim → n_classes，交叉熵 + Adam），再以探针预测概率编码本块金标签，
代价 −log2 p(gold)。各块累加即 MDL（bit），越小表征越好；压缩率 = 均匀码长 / MDL。
探针能力极弱，故每前缀固定 epochs（默认 5），与论文一致不做早停；码长只在
train 上计算，test 仅报告最终探针准确率作参考。
"""
import numpy as np
import torch

PORTIONS = [t / 10 for t in range(1, 11)]


def train_classifier(
    X: np.ndarray, y: np.ndarray, n_classes: int, epochs: int, lr: float, batch_size: int, seed: int = 0
) -> torch.nn.Linear:
    torch.manual_seed(seed)
    clf = torch.nn.Linear(X.shape[1], n_classes)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    for epoch in range(epochs):
        order = rng.permutation(len(X))
        tot, nb = 0.0, 0
        for a in range(0, len(X), batch_size):
            idx = torch.from_numpy(order[a : a + batch_size])
            loss = torch.nn.functional.cross_entropy(clf(Xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        if epoch == 0 or epoch % 5 == 4:
            print(f"    epoch {epoch + 1}/{epochs}  CE={tot / nb:.4f}", flush=True)
    return clf


def _uniform_classifier(dim: int, n_classes: int) -> torch.nn.Linear:
    """零权重线性层：对任意输入输出均匀分布，用于单元自检。"""
    clf = torch.nn.Linear(dim, n_classes)
    with torch.no_grad():
        clf.weight.zero_()
        clf.bias.zero_()
    return clf


def mdl_codelength(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    portions: list[float],
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int = 0,
    train_fn=train_classifier,
) -> tuple[float, torch.nn.Linear]:
    """在线码长。返回 (总 bit, 最后一个前缀训出的探针)。train_fn 可替换以自检。"""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    Xp, yp = X[order], y[order]
    bounds = [0] + [int(round(t * len(X))) for t in portions]
    bits = (bounds[1] - bounds[0]) * float(np.log2(n_classes))
    clf = None
    for k in range(1, len(bounds) - 1):
        a, b = bounds[k], bounds[k + 1]
        print(f"  前缀 {portions[k - 1]:.1f}→{portions[k]:.1f}：训练 {a} 词，编码 {b - a} 词", flush=True)
        clf = train_fn(Xp[:a], yp[:a], n_classes, epochs, lr, batch_size, seed)
        with torch.no_grad():
            logp = torch.log_softmax(clf(torch.from_numpy(Xp[a:b])), dim=-1)
            nll = -logp[torch.arange(b - a), torch.from_numpy(yp[a:b])].sum()
            bits += float(nll) / float(np.log(2))
    return bits, clf


def accuracy(clf: torch.nn.Linear, X: np.ndarray, y: np.ndarray, batch_size: int = 8192) -> float:
    correct = 0
    with torch.no_grad():
        for a in range(0, len(X), batch_size):
            pred = clf(torch.from_numpy(X[a : a + batch_size])).argmax(-1).numpy()
            correct += int((pred == y[a : a + batch_size]).sum())
    return correct / len(X)


def run_layer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    portions: list[float] = PORTIONS,
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 1024,
    seed: int = 0,
) -> dict:
    """单层的 MDL、压缩率与最终探针 test 准确率。"""
    bits, clf = mdl_codelength(X_train, y_train, n_classes, portions, epochs, lr, batch_size, seed)
    uniform = len(X_train) * float(np.log2(n_classes))
    return {
        "bits": bits,
        "compression": uniform / bits,
        "test_acc": accuracy(clf, X_test, y_test),
    }


def run(
    layer_reprs: dict[int, np.ndarray],
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    **kw,
) -> list[dict]:
    """逐层跑 run_layer，返回 [{"layer", "bits", "compression", "test_acc"}]（不做断点）。"""
    rows = []
    for layer in sorted(layer_reprs):
        row = {"layer": layer}
        row.update(run_layer(layer_reprs[layer], y_train, X_test, y_test, n_classes, **kw))
        rows.append(row)
    return rows


def uniform_selftest() -> None:
    """单元自检：探针恒预测均匀分布时，MDL 应精确等于均匀码长。"""
    rng = np.random.default_rng(0)
    n, dim, k = 2000, 16, 7
    X = rng.normal(size=(n, dim)).astype(np.float32)
    y = rng.integers(0, k, size=n)
    bits, _ = mdl_codelength(
        X, y, k, PORTIONS, epochs=1, lr=1e-3, batch_size=256,
        train_fn=lambda X_, y_, k_, *a, **kw: _uniform_classifier(dim, k_),
    )
    uniform = n * float(np.log2(k))
    ok = abs(bits - uniform) < 1e-3 * uniform
    print(f"自检: MDL={bits:.1f} bit  均匀码长={uniform:.1f} bit  相对误差={abs(bits - uniform) / uniform:.2e}")
    print("UNIFORM GATE:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    uniform_selftest()
