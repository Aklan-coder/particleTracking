"""
train_model.py — Phase 2, step 2: train one simple, explainable classifier
per resolution.

Model: regularized logistic regression on hand-inspectable features —
deliberately NOT a neural network. If a linear model on stated features
beats geometry past the cliff, the signal is unambiguous and we can say
exactly WHAT carried it. Implemented in plain numpy (gradient descent);
no ML libraries required.

Features per example (all from the cell patch, unless --with-size):
  occupied-cell count within the patch     (footprint proxy)
  patch height statistics: mean/max of mean_h
  variance-channel statistics: mean/max of var_h    <- dome-vs-plateau
  radial height profile: mean height in 4 concentric rings around the
  patch center (a dome falls off with radius; a plateau does not)
  aspect ratio of the occupied cells (box footprints are elongated)

Two variants trained per h:
  pattern-only      (size features removed) -> is it SHAPE that carries it?
  pattern+size      (all features)          -> everything a radar would see

Output: results/model_h{h}_{variant}.npz (weights + feature names + scaler)
Usage:  python3 train_model.py
"""
import os

import numpy as np

import config

H_TRAIN = [10, 20, 30, 40]
RINGS = 4


def features(X, n_cells, with_size):
    """(N,3,P,P) patches -> (N,F) feature matrix + names."""
    N, _, P, _ = X.shape
    occ, mh, vh = X[:, 0], X[:, 1], X[:, 2]
    eps = 1e-9
    nocc = occ.sum(axis=(1, 2)) + eps

    # radial rings around patch center
    yy, xx = np.mgrid[0:P, 0:P]
    r = np.hypot(yy - P // 2, xx - P // 2)
    edges = np.linspace(0, r.max() + eps, RINGS + 1)
    ring_means = []
    for k in range(RINGS):
        m = (r >= edges[k]) & (r < edges[k + 1])
        w = occ * m                      # only occupied cells count
        ring_means.append((mh * w).sum(axis=(1, 2))
                          / (w.sum(axis=(1, 2)) + eps))

    # aspect ratio of occupied cells (2nd moments)
    ys = (occ * yy).sum(axis=(1, 2)) / nocc
    xs = (occ * xx).sum(axis=(1, 2)) / nocc
    vy = (occ * (yy - ys[:, None, None]) ** 2).sum(axis=(1, 2)) / nocc
    vx = (occ * (xx - xs[:, None, None]) ** 2).sum(axis=(1, 2)) / nocc
    aspect = np.sqrt((np.maximum(vy, vx) + eps)
                     / (np.minimum(vy, vx) + eps))

    cols = [
        ("mean_height", (mh * occ).sum(axis=(1, 2)) / nocc),
        ("max_height", mh.max(axis=(1, 2))),
        ("mean_variance", (vh * occ).sum(axis=(1, 2)) / nocc),
        ("max_variance", vh.max(axis=(1, 2))),
        ("aspect_ratio", aspect),
    ]
    cols += [(f"ring{k}_height", ring_means[k]) for k in range(RINGS)]
    if with_size:
        cols.append(("occupied_cells", n_cells.astype(float)))
        cols.append(("patch_occupancy", nocc))
    names = [c[0] for c in cols]
    F = np.column_stack([c[1] for c in cols])
    return F, names


def train_logreg(F, y, l2=1e-2, iters=3000, lr=0.1):
    """Plain-numpy logistic regression with standardization."""
    mu, sd = F.mean(axis=0), F.std(axis=0) + 1e-9
    Z = (F - mu) / sd
    N, D = Z.shape
    w = np.zeros(D)
    b = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Z @ w + b)))
        g = Z.T @ (p - y) / N + l2 * w
        gb = float((p - y).mean())
        w -= lr * g
        b -= lr * gb
    return w, b, mu, sd


def main():
    for h in H_TRAIN:
        d = np.load(os.path.join(config.RESULTS_DIR,
                                 f"dataset_h{h}.npz"), allow_pickle=True)
        for variant, with_size in [("pattern", False), ("full", True)]:
            F, names = features(d["X_train"], d["n_train"], with_size)
            w, b, mu, sd = train_logreg(F, d["y_train"].astype(float))
            # training accuracy (sanity only; real score is eval on test)
            p = 1 / (1 + np.exp(-(((F - mu) / sd) @ w + b)))
            acc = float(((p > 0.5) == d["y_train"]).mean())
            out = os.path.join(config.RESULTS_DIR,
                               f"model_h{h}_{variant}.npz")
            np.savez(out, w=w, b=b, mu=mu, sd=sd,
                     names=np.array(names), with_size=with_size)
            print(f"h={h:2d} {variant:8s}: trained on "
                  f"{len(F)} examples | train acc {acc:.3f} | saved {out}")


if __name__ == "__main__":
    main()