"""
eval_model.py — Phase 2, step 3: score the trained models on the held-out
TEST frames (never seen in training; time-split) and produce the answer
figure: learned accuracy vs h, overlaid on the geometric identification
cliff.

Also reports: accuracy split by static vs moving recordings, per-class
accuracy (ball vs box), and each model's most influential features (the
'what did it learn' answer - available because the model is linear).

Output: results/learning_vs_geometry.png, results/phase2_results.csv
Usage:  python3 eval_model.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from train_model import H_TRAIN, features

# geometric identification at the same h (from the measured sweep;
# ball/box averaged over static+moving) - the baseline to beat
GEOMETRY_ID = {10: 0.965, 20: 0.9575, 30: 0.20, 40: 0.02}


def predict(model, F):
    Z = (F - model["mu"]) / model["sd"]
    return 1 / (1 + np.exp(-(Z @ model["w"] + float(model["b"])))) > 0.5


def main():
    rows = []
    curves = {"pattern": [], "full": []}
    for h in H_TRAIN:
        d = np.load(os.path.join(config.RESULTS_DIR,
                                 f"dataset_h{h}.npz"), allow_pickle=True)
        y = d["y_test"]
        meta = [str(m) for m in d["meta_test"]]
        moving = np.array(["moving" in m for m in meta])
        for variant, with_size in [("pattern", False), ("full", True)]:
            model = np.load(os.path.join(
                config.RESULTS_DIR, f"model_h{h}_{variant}.npz"),
                allow_pickle=True)
            F, names = features(d["X_test"], d["n_test"], with_size)
            pred = predict(model, F)
            acc = float((pred == y).mean())
            acc_static = float((pred[~moving] == y[~moving]).mean())
            acc_moving = float((pred[moving] == y[moving]).mean())
            acc_ball = float((pred[y == 0] == 0).mean())
            acc_box = float((pred[y == 1] == 1).mean())
            # top features by |weight| (the 'what did it learn' answer)
            w = model["w"]
            top = sorted(zip([str(n) for n in model["names"]],
                             np.abs(w), w),
                         key=lambda t: -t[1])[:3]
            top_s = ", ".join(f"{n} ({v:+.2f})" for n, _, v in top)
            rows.append([h, variant, f"{acc:.3f}", f"{acc_static:.3f}",
                         f"{acc_moving:.3f}", f"{acc_ball:.3f}",
                         f"{acc_box:.3f}", top_s])
            curves[variant].append(acc)
            print(f"h={h:2d} {variant:8s}: TEST acc {acc:.3f} "
                  f"(static {acc_static:.3f} / moving {acc_moving:.3f} | "
                  f"ball {acc_ball:.3f} / box {acc_box:.3f})")
            print(f"          top features: {top_s}")

    with open(os.path.join(config.RESULTS_DIR,
                           "phase2_results.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["h_mm", "variant", "test_acc", "acc_static",
                    "acc_moving", "acc_ball", "acc_box", "top_features"])
        w.writerows(rows)

    # ---- the answer figure ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    hs = H_TRAIN
    ax.plot(hs, [GEOMETRY_ID[h] for h in hs], "k--", marker="d",
            label="geometry (measured sweep)", lw=2)
    ax.plot(hs, curves["pattern"], color="tab:purple", marker="o",
            label="learned - pattern only", lw=2)
    ax.plot(hs, curves["full"], color="tab:green", marker="s",
            label="learned - pattern + size", lw=2)
    ax.axvspan(25, 40, color="red", alpha=0.08,
               label="geometric cliff zone")
    ax.axhline(0.5, color="grey", lw=0.8, ls=":",
               label="coin flip (0.5)")
    ax.set_xlabel("cell size h (mm)")
    ax.set_ylabel("ball-vs-box accuracy on UNSEEN test frames")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Phase 2: does learning survive past the geometric "
                 "cliff?")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    out = os.path.join(config.RESULTS_DIR, "learning_vs_geometry.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out} and results/phase2_results.csv")


if __name__ == "__main__":
    main()