"""
plot.py — ONE script, run once, makes ALL the result pictures:

  1. results/class_accuracy_compare2.png
       ball vs box identification accuracy (learned model vs geometry),
       PLUS confidence error (MSE / MAE) on the same picture.
  2. results/traj_compare.png
       trajectory model vs constant-velocity baseline, across window
       sizes k=1,3,5,10.
  3. results/traj_predictions_path.png
       one real track: predicted position vs true position, drawn on
       the actual path.
  4. results/traj_error_hist.png
       trajectory error distribution (model vs baseline), all test
       examples.

Requires (already produced by earlier steps — no new pipeline runs
needed): results/dataset_h*.npz, results/model_h*_full.npz,
results/sweep_results.csv, results/*_tracks.csv

Usage: python3 plot.py            (does everything, default k=5, h=10
                                    for the trajectory pictures)
       python3 plot.py --k 3 --h 10
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from train_model import H_TRAIN, features
import make_dataset2 as md2
import train_model2 as tm2

K_VALUES = [1, 3, 5, 10]


# ===========================================================================
# PART 1 — classification: accuracy + MSE/MAE (ball vs box, learned vs geometry)
# ===========================================================================
def _load_geometry():
    path = os.path.join(config.RESULTS_DIR, "sweep_results.csv")
    rows = list(csv.DictReader(open(path)))
    ball, box = {}, {}
    for r in rows:
        h = float(r["h"])
        if r["recording"] == "ball_static":
            ball[h] = float(r["id_rate"])
        elif r["recording"] == "box_static":
            box[h] = float(r["id_rate"])
    hs = sorted(set(ball) & set(box))
    return (np.array(hs), np.array([ball[h] for h in hs]),
           np.array([box[h] for h in hs]))


def _learned_predictions(h):
    d = np.load(os.path.join(config.RESULTS_DIR, f"dataset_h{h}.npz"),
               allow_pickle=True)
    model = np.load(os.path.join(config.RESULTS_DIR,
                                 f"model_h{h}_full.npz"), allow_pickle=True)
    F, _ = features(d["X_test"], d["n_test"], with_size=True)
    Z = (F - model["mu"]) / model["sd"]
    p = 1 / (1 + np.exp(-(Z @ model["w"] + float(model["b"]))))
    y = d["y_test"].astype(float)
    return p, y


def plot_classification():
    hs_l = H_TRAIN
    ball_acc, box_acc, mse_list, mae_list = [], [], [], []
    for h in hs_l:
        p, y = _learned_predictions(h)
        pred = p > 0.5
        ball_acc.append(float((pred[y == 0] == 0).mean()))
        box_acc.append(float((pred[y == 1] == 1).mean()))
        mse_list.append(float(np.mean((p - y) ** 2)))
        mae_list.append(float(np.mean(np.abs(p - y))))

    hs_g, ball_g, box_g = _load_geometry()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(hs_g, ball_g, "o--", color="tab:blue", alpha=0.4, label="geometry — ball")
    ax1.plot(hs_g, box_g, "s--", color="tab:red", alpha=0.4, label="geometry — box")
    ax1.plot(hs_l, ball_acc, "o-", color="tab:blue", lw=2, label="learned model — ball accuracy")
    ax1.plot(hs_l, box_acc, "s-", color="tab:red", lw=2, label="learned model — box accuracy")
    ax1.axhline(0.5, color="grey", lw=0.8, ls=":", label="coin flip (0.5)")
    ax1.set_xlabel("cell size h (mm)")
    ax1.set_ylabel("identification accuracy")
    ax1.set_ylim(-0.02, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(hs_l, mse_list, "D-", color="gold", lw=2.5, markersize=8,
             markeredgecolor="black", label="MSE (Brier score)")
    ax2.plot(hs_l, mae_list, "^-", color="orange", lw=2, markersize=7,
             markeredgecolor="black", label="MAE")
    ax2.set_ylabel("prediction error (MSE / MAE, lower = better)")
    ax2.set_ylim(-0.02, 0.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center left")
    ax1.set_title("Ball vs box: accuracy AND prediction confidence error\n"
                 "(MSE/MAE near 0 = confident + correct)")
    ax1.grid(alpha=0.3)
    out = os.path.join(config.RESULTS_DIR, "class_accuracy_compare2.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ===========================================================================
# PART 2 — trajectory: model vs baseline, across window sizes k
# ===========================================================================
def _run_traj(k, horizon):
    md2.main(k, horizon)
    path = os.path.join(config.RESULTS_DIR, f"traj_dataset_k{k}_h{horizon}.npz")
    d = np.load(path, allow_pickle=True)
    feature_names = [str(n) for n in d["feature_names"]]
    X_train, Y_train = d["X_train"], d["Y_train"]
    X_test, Y_test, U_test = d["X_test"], d["Y_test"], d["U_test"]

    W, b, mu, sd = tm2.train_linreg(X_train, Y_train)
    pred_test = tm2.predict_disp(X_test, W, b, mu, sd)
    err_model = np.hypot(pred_test[:, 0] - Y_test[:, 0], pred_test[:, 1] - Y_test[:, 1])
    cv_test = tm2.baseline_cv(U_test, X_test, feature_names, horizon)
    err_cv = np.hypot(cv_test[:, 0] - Y_test[:, 0], cv_test[:, 1] - Y_test[:, 1])
    return (float(np.median(err_model)), float(np.median(err_cv)),
           W, b, mu, sd, d, feature_names, pred_test, cv_test, err_model, err_cv)


def plot_traj_compare(horizon):
    model_errs, cv_errs = [], []
    for k in K_VALUES:
        me, ce, *_ = _run_traj(k, horizon)
        model_errs.append(me)
        cv_errs.append(ce)
        print(f"  k={k:2d}: multi-frame model {me:.2f} mm | baseline {ce:.2f} mm")

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(K_VALUES))
    width = 0.35
    ax.bar(x - width/2, model_errs, width, label="multi-frame model", color="tab:blue")
    ax.bar(x + width/2, cv_errs, width, label="constant-velocity baseline", color="grey")
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in K_VALUES])
    ax.set_ylabel("median position error on TEST set (mm)")
    ax.set_xlabel("window size (frames)")
    ax.set_title(f"Multi-frame trajectory model vs baseline (+{horizon} frame prediction)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    out = os.path.join(config.RESULTS_DIR, "traj_compare.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ===========================================================================
# PART 3 — trajectory: predicted vs true path, + error histogram (one k, h)
# ===========================================================================
def plot_traj_predictions(k, horizon):
    _, _, W, b, mu, sd, d, feature_names, pred_test, cv_test, err_model, err_cv = \
        _run_traj(k, horizon)
    U_test, Y_test = d["U_test"], d["Y_test"]
    meta_test = [str(m) for m in d["meta_test"]]

    tracks = {}
    for i, m in enumerate(meta_test):
        rec, tid, frame = m.split(":")
        tracks.setdefault((rec, tid), []).append(i)
    best_key = max(tracks, key=lambda kk: len(tracks[kk]))
    idxs = sorted(tracks[best_key], key=lambda i: int(meta_test[i].split(":")[2]))

    now_pos = U_test[idxs]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(now_pos[:, 0], now_pos[:, 1], "-", color="lightgrey", lw=1,
           label="actual path (current positions)", zorder=1)
    sample = idxs[::max(1, len(idxs)//15)]
    for i in sample:
        u0, v0 = U_test[i]
        ut, vt = U_test[i] + Y_test[i]
        up, vp = U_test[i] + pred_test[i]
        ax.plot([u0, ut], [v0, vt], "g-", lw=1, alpha=0.6)
        ax.plot([u0, up], [v0, vp], "b--", lw=1, alpha=0.6)
    ax.scatter(*zip(*[(U_test[i][0]+Y_test[i][0], U_test[i][1]+Y_test[i][1]) for i in sample]),
              c="green", marker="o", s=40, label="true future position", zorder=3)
    ax.scatter(*zip(*[(U_test[i][0]+pred_test[i][0], U_test[i][1]+pred_test[i][1]) for i in sample]),
              c="blue", marker="x", s=50, label="model's predicted position", zorder=3)
    ax.set_xlabel("u (mm)")
    ax.set_ylabel("v (mm)")
    ax.set_title(f"{best_key[0]} track {best_key[1]}: predicted vs true position "
                f"(+{horizon} frames, k={k})")
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    out1 = os.path.join(config.RESULTS_DIR, "traj_predictions_path.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out1}")

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, max(err_model.max(), err_cv.max()), 40)
    ax.hist(err_cv, bins=bins, alpha=0.5, color="grey", label="constant-velocity baseline")
    ax.hist(err_model, bins=bins, alpha=0.6, color="tab:blue", label="multi-frame model")
    ax.axvline(np.median(err_cv), color="grey", ls="--", lw=1.5)
    ax.axvline(np.median(err_model), color="tab:blue", ls="--", lw=1.5)
    ax.set_xlabel("position error (mm)")
    ax.set_ylabel("number of test examples")
    ax.set_title(f"Error distribution on TEST set (k={k}, +{horizon} frames)")
    ax.legend()
    out2 = os.path.join(config.RESULTS_DIR, "traj_error_hist.png")
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out2}")


# ===========================================================================
def main(k=5, horizon=10):
    print("=== 1/3: classification accuracy + MSE/MAE ===")
    plot_classification()
    print("\n=== 2/3: trajectory model vs baseline, across window sizes ===")
    plot_traj_compare(horizon)
    print(f"\n=== 3/3: trajectory predicted-vs-true path + error histogram (k={k}) ===")
    plot_traj_predictions(k, horizon)
    print("\nDONE. All PNGs saved in results/:")
    print("  class_accuracy_compare2.png")
    print("  traj_compare.png")
    print("  traj_predictions_path.png")
    print("  traj_error_hist.png")


if __name__ == "__main__":
    k = (int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else 5)
    horizon = (int(sys.argv[sys.argv.index("--h") + 1]) if "--h" in sys.argv else 10)
    main(k, horizon)