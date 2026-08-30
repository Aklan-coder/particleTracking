"""
train_model2.py — trains the multi-frame trajectory model on the windows
built by make_dataset2.py.

Companion to train_model.py, same design commitment: plain-numpy, no ML
library, so what the model learned is readable from the weights.

train_model.py:  logistic regression -> classify ball vs box (one frame)
train_model2.py: linear regression   -> predict (du, dv) displacement
                                         from a k-frame window

Model: multivariate linear regression (2 outputs: delta_u, delta_v),
       plain-numpy gradient descent, same style as train_logreg() in
       train_model.py.

Baseline for comparison (already in your pipeline, not re-implemented
here): constant velocity, i.e. displacement = du * horizon_time. This
script reports the model's error alongside that baseline so you can see
directly whether the k-frame window adds anything over "just use the
current Kalman velocity" — the same comparison predict.py already makes
between constant-velocity and shape-conditioned prediction.

Output: results/traj_model_k{k}_h{horizon}.npz (w, b, mu, sd, feature
        names) + printed/saved train vs test error report (mm).
Usage:  python3 train_model2.py            (matches make_dataset2.py's
                                             default k=5, horizon=10)
        python3 train_model2.py --k 5 --h 10
"""
import os
import sys

import numpy as np

import config


def train_linreg(X, Y, l2=1e-2, iters=3000, lr=0.1):
    """Plain-numpy linear regression, 2 outputs at once (du, dv),
    standardized inputs — same recipe as train_model.py's train_logreg,
    just no sigmoid (this is regression, not classification)."""
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Z = (X - mu) / sd
    N, D = Z.shape
    O = Y.shape[1]
    W = np.zeros((D, O))
    b = np.zeros(O)
    for _ in range(iters):
        pred = Z @ W + b
        err = pred - Y
        gW = Z.T @ err / N + l2 * W
        gb = err.mean(axis=0)
        W -= lr * gW
        b -= lr * gb
    return W, b, mu, sd


def predict_disp(X, W, b, mu, sd):
    Z = (X - mu) / sd
    return Z @ W + b


def baseline_cv(U, X, feature_names, horizon):
    """Constant-velocity baseline: displacement = current velocity *
    horizon time. Uses the SAME current-frame velocity already in the
    window features (du, dv columns), so it's a fair, cheap comparison —
    exactly what predict.py already treats as the shape-blind null
    model."""
    du_i = feature_names.index("du")
    dv_i = feature_names.index("dv")
    t = horizon * config.DT
    return np.column_stack([X[:, du_i] * t, X[:, dv_i] * t])


def report(name, Y_true, Y_pred, U):
    """Median position error in mm — same metric predict.py uses, so
    this is directly comparable to results/prediction_summary.txt."""
    err = np.hypot(Y_pred[:, 0] - Y_true[:, 0], Y_pred[:, 1] - Y_true[:, 1])
    print(f"  {name:22s}: median err {np.median(err):6.2f} mm  "
         f"| mean err {err.mean():6.2f} mm  | n={len(err)}")
    return err


def main(k=5, horizon=10):
    path = os.path.join(config.RESULTS_DIR,
                        f"traj_dataset_k{k}_h{horizon}.npz")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found — run make_dataset2.py "
                         f"--k {k} --h {horizon} first")
    d = np.load(path, allow_pickle=True)
    feature_names = [str(n) for n in d["feature_names"]]

    X_train, Y_train, U_train = d["X_train"], d["Y_train"], d["U_train"]
    X_test, Y_test, U_test = d["X_test"], d["Y_test"], d["U_test"]
    print(f"window k={k}  horizon=+{horizon} frames")
    print(f"train windows: {len(Y_train)} | test windows: {len(Y_test)}\n")

    W, b, mu, sd = train_linreg(X_train, Y_train)

    print("=== TRAIN set (data the model learned from) ===")
    pred_train = predict_disp(X_train, W, b, mu, sd)
    report("multi-frame model", Y_train, pred_train, U_train)
    cv_train = baseline_cv(U_train, X_train, feature_names, horizon)
    report("constant-velocity baseline", Y_train, cv_train, U_train)

    print("\n=== TEST set (data the model NEVER saw) ===")
    pred_test = predict_disp(X_test, W, b, mu, sd)
    err_model = report("multi-frame model", Y_test, pred_test, U_test)
    cv_test = baseline_cv(U_test, X_test, feature_names, horizon)
    err_cv = report("constant-velocity baseline", Y_test, cv_test, U_test)

    improve = 100 * (np.median(err_cv) - np.median(err_model)) / np.median(err_cv)
    print(f"\nmulti-frame model vs constant-velocity on TEST: "
         f"{improve:+.1f}% median error change "
         f"(positive = model is better)")

    # what did it learn? (readable weights, same spirit as train_model.py)
    print("\ntop feature weights (|w| for delta_u, delta_v):")
    mags = np.abs(W).sum(axis=1)
    order = np.argsort(-mags)
    for idx in order[:5]:
        print(f"  {feature_names[idx]:8s}: "
             f"w_du={W[idx,0]:+.3f}  w_dv={W[idx,1]:+.3f}")

    out = os.path.join(config.RESULTS_DIR,
                       f"traj_model_k{k}_h{horizon}.npz")
    np.savez(out, W=W, b=b, mu=mu, sd=sd,
             feature_names=np.array(feature_names), k=k, horizon=horizon)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    k = (int(sys.argv[sys.argv.index("--k") + 1])
         if "--k" in sys.argv else 5)
    horizon = (int(sys.argv[sys.argv.index("--h") + 1])
              if "--h" in sys.argv else 10)
    main(k, horizon)