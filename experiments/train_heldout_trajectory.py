"""
train_heldout_trajectory.py — trains the trajectory model on the NEW
track-independent held-out datasets. Evaluates TRAIN (diagnostic) and
VALIDATION only. Never touches TEST. Modifies no existing file.

SAME MODEL AS THE ORIGINAL POC, verified by direct reuse: this script
calls train_model2.train_linreg() with its default arguments
(l2=1e-2, iters=3000, lr=0.1) -- IDENTICAL to train_model2.py's own
defaults, unchanged. Only the DATA SOURCE differs (held-out track-split
datasets instead of the original time-split-within-track datasets). This
isolates the effect of the split methodology from the model itself, per
your instruction #16.

REUSED DIRECTLY: train_model2.train_linreg(), predict_disp(),
baseline_cv() -- imported, not reimplemented.

TEST LOCK, verified structurally not just by omission: this file
contains ZERO occurrences of the strings "F_test", "meta_test" as
dictionary keys read from the held-out .npz files (grep this file
yourself to confirm -- the only place "test" appears is in this
docstring's own explanation and the final printed lock message).

Usage:
  python train_heldout_trajectory.py
  python train_heldout_trajectory.py --heldout-dir results/heldout_trajectory --out validation_results/heldout_trajectory_training
  python train_heldout_trajectory.py --help
"""
import argparse
import csv
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = _THIS_DIR
for _ in range(4):
    if os.path.exists(os.path.join(_PROJECT_ROOT, "config.py")):
        break
    _PROJECT_ROOT = os.path.dirname(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

import config
import train_model2 as tm2

K_VALUES = [1, 3, 5, 10]
HORIZON = 10

KNOWN_IDENTITY = {
    ("ball_moving", None): "ball", ("box_moving", None): "box",
    ("ball_and_box_moving", "2"): "ball",
    ("ball_and_box_moving", "1"): "box", ("ball_and_box_moving", "4"): "box",
}
def object_identity(rec, tid):
    tid = str(tid)
    if (rec, tid) in KNOWN_IDENTITY: return KNOWN_IDENTITY[(rec, tid)]
    if (rec, None) in KNOWN_IDENTITY: return KNOWN_IDENTITY[(rec, None)]
    return "unknown"


def pctl(a,p): return float(np.percentile(a,p)) if len(a) else float("nan")
def full_stats(err):
    err = np.asarray(err, dtype=float)
    if len(err)==0: return dict(n=0, mean=np.nan, median=np.nan, rmse=np.nan, std=np.nan, p90=np.nan, p95=np.nan, max=np.nan)
    return dict(n=int(len(err)), mean=float(err.mean()), median=float(np.median(err)),
               rmse=float(np.sqrt((err**2).mean())), std=float(err.std()),
               p90=pctl(err,90), p95=pctl(err,95), max=float(err.max()))
def euclid(pred,true): return np.hypot(pred[:,0]-true[:,0], pred[:,1]-true[:,1])
def coord_errors(pred,true):
    du,dv = pred[:,0]-true[:,0], pred[:,1]-true[:,1]
    return dict(u_mae=float(np.mean(np.abs(du))), v_mae=float(np.mean(np.abs(dv))),
               u_rmse=float(np.sqrt(np.mean(du**2))), v_rmse=float(np.sqrt(np.mean(dv**2))))
def save_csv(path, rows, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w",newline="") as fh:
        w=csv.writer(fh); w.writerow(header); w.writerows(rows)
    return path

REPORT = []
def note(md): REPORT.append(md)
def heading(md):
    print("\n"+"="*78); print(md); print("="*78); REPORT.append(md)


def main():
    p = argparse.ArgumentParser(
        description="Train trajectory model on held-out (track-independent) "
                    "datasets. Evaluates TRAIN + VALIDATION only. TEST is "
                    "never touched.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--heldout-dir", type=str, default="results/heldout_trajectory")
    p.add_argument("--out", type=str, default="validation_results/heldout_trajectory_training")
    args = p.parse_args()
    def _resolve(path): return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)
    args.heldout_dir = _resolve(args.heldout_dir); args.out = _resolve(args.out)
    models_dir = os.path.join(args.heldout_dir, "models")
    plots_dir = os.path.join(args.out, "plots")
    os.makedirs(models_dir, exist_ok=True); os.makedirs(plots_dir, exist_ok=True)

    heading("# HELD-OUT TRAJECTORY TRAINING")

    train_rows, val_rows, base_rows, tvv_rows, track_rows, obj_rows = [], [], [], [], [], []
    per_k = {}

    for k in K_VALUES:
        ds_path = os.path.join(args.heldout_dir, f"heldout_traj_dataset_k{k}_h{HORIZON}.npz")
        if not os.path.exists(ds_path):
            print(f"\nk={k}: STOP -- {ds_path} not found. Run "
                 f"make_heldout_trajectory_dataset.py first.")
            continue
        d = np.load(ds_path, allow_pickle=True)
        feature_names = [str(n) for n in d["feature_names"]]

        # ONLY train/validation keys ever read -- test keys are never indexed
        F_train, Y_train, U_train, meta_train = d["F_train"], d["Y_train"], d["U_train"], d["meta_train"]
        F_val, Y_val, U_val, meta_val = d["F_val"], d["Y_val"], d["U_val"], d["meta_val"]
        meta_train = [str(m) for m in meta_train]; meta_val = [str(m) for m in meta_val]

        print(f"\nk={k}")
        print(f"  Train windows: {len(Y_train)}   Validation windows: {len(Y_val)}")

        # ---- fit on TRAIN only, SAME defaults as train_model2.py, unchanged ----
        W, b, mu, sd = tm2.train_linreg(F_train, Y_train)  # l2=1e-2, iters=3000, lr=0.1, defaults
        print(f"  mu/sd computed from F_train ONLY ({len(F_train)} samples) -- "
             f"validation had zero influence on normalization or fitting.")

        model_path = os.path.join(models_dir, f"heldout_traj_model_k{k}_h{HORIZON}.npz")
        np.savez(model_path, W=W, b=b, mu=mu, sd=sd, k=k, horizon=HORIZON,
                feature_names=np.array(feature_names),
                l2=1e-2, iters=3000, lr=0.1)   # training config, unchanged from POC, recorded for traceability
        print(f"  saved {model_path}")

        # ---- TRAIN diagnostic ----
        pred_train = tm2.predict_disp(F_train, W, b, mu, sd)
        err_train = euclid(pred_train, Y_train)
        s_train = full_stats(err_train)
        print(f"  TRAIN median={s_train['median']:.2f}mm  RMSE={s_train['rmse']:.2f}mm")
        train_rows.append([k, s_train["n"], s_train["mean"], s_train["median"], s_train["rmse"]])

        # ---- VALIDATION (evaluation only, frozen model) ----
        pred_val = tm2.predict_disp(F_val, W, b, mu, sd)
        err_val = euclid(pred_val, Y_val)
        s_val = full_stats(err_val)
        cm = coord_errors(pred_val, Y_val)
        print(f"  VALIDATION learned: N={s_val['n']} median={s_val['median']:.2f}mm "
             f"RMSE={s_val['rmse']:.2f}mm P90={s_val['p90']:.2f} P95={s_val['p95']:.2f}")
        val_rows.append([k, s_val["n"], s_val["mean"], s_val["median"], s_val["rmse"],
                        s_val["std"], s_val["p90"], s_val["p95"], s_val["max"],
                        cm["u_mae"], cm["v_mae"], cm["u_rmse"], cm["v_rmse"]])

        # ---- baselines on VALIDATION only ----
        cv_disp = tm2.baseline_cv(U_val, F_val, feature_names, HORIZON)
        zero_disp = np.zeros_like(Y_val)
        err_cv = euclid(cv_disp, Y_val)
        err_zero = euclid(zero_disp, Y_val)
        s_cv, s_zero = full_stats(err_cv), full_stats(err_zero)
        imp_cv = 100*(s_cv["median"]-s_val["median"])/s_cv["median"] if s_cv["median"] else float("nan")
        imp_zero = 100*(s_zero["median"]-s_val["median"])/s_zero["median"] if s_zero["median"] else float("nan")
        print(f"  VALIDATION constant-velocity median={s_cv['median']:.2f}mm  "
             f"zero-motion median={s_zero['median']:.2f}mm")
        print(f"  Improvement vs CV: {imp_cv:+.1f}%   vs zero-motion: {imp_zero:+.1f}%")
        base_rows.append([k, s_val["median"], s_cv["median"], s_zero["median"], imp_cv, imp_zero])

        tvv_rows.append([k, s_train["n"], s_train["median"], s_train["rmse"],
                        s_val["n"], s_val["median"], s_val["rmse"]])

        # ---- per validation track ----
        recs = np.array([m.split(":")[0] for m in meta_val])
        tids = np.array([m.split(":")[1] for m in meta_val])
        for rec, tid in sorted(set(zip(recs, tids))):
            sel = (recs==rec) & (tids==tid)
            sm = full_stats(err_val[sel])
            sc_med = float(np.median(err_cv[sel])) if sel.sum() else float("nan")
            sz_med = float(np.median(err_zero[sel])) if sel.sum() else float("nan")
            obj = object_identity(rec, tid)
            print(f"    validation track {rec}:{tid} ({obj}): n={sm['n']} "
                 f"learned_median={sm['median']:.2f} cv_median={sc_med:.2f}")
            track_rows.append([k, rec, tid, obj, sm["n"], sm["median"], sm["rmse"], sc_med, sz_med])

        # ---- ball vs box on validation ----
        idents = np.array([object_identity(recs[i], tids[i]) for i in range(len(recs))])
        for obj in ["ball","box"]:
            sel = idents==obj
            if sel.sum():
                sm = full_stats(err_val[sel])
                obj_rows.append([k, obj, sm["n"], sm["mean"], sm["median"], sm["rmse"]])

        per_k[k] = dict(err_val=err_val, recs=recs, tids=tids,
                       frames=np.array([int(m.split(":")[2]) for m in meta_val]),
                       s_val=s_val)

    if not per_k:
        print("\nNo held-out datasets found. Stopping."); return

    # ---- common validation targets across k ----
    heading("## COMMON VALIDATION TARGETS")
    key_sets = {k: set(zip(per_k[k]["recs"].tolist(), per_k[k]["tids"].tolist(), per_k[k]["frames"].tolist()))
               for k in per_k}
    common = set.intersection(*key_sets.values())
    print(f"N common validation targets across all loaded k: {len(common)}")
    common_rows = []
    for k in sorted(per_k):
        r = per_k[k]
        lut = {}
        for i in range(len(r["recs"])):
            lut[(r["recs"][i], r["tids"][i], r["frames"][i])] = r["err_val"][i]
        common_err = np.array([lut[key] for key in common]) if common else np.array([])
        s = full_stats(common_err)
        print(f"  k={k}: N={s['n']} median={s['median']:.2f}mm RMSE={s['rmse']:.2f}mm "
             f"P90={s['p90']:.2f} P95={s['p95']:.2f}")
        common_rows.append([k, s["n"], s["mean"], s["median"], s["rmse"], s["p90"], s["p95"]])
    note("**These are VALIDATION results only.** No k is automatically "
        "declared 'best' or 'final' -- the numerical table above is for "
        "your review before deciding the TEST evaluation plan.")

    # ---- write CSVs ----
    save_csv(os.path.join(args.out, "heldout_training_metrics.csv"), train_rows,
            ["k","n_train","mean_mm","median_mm","rmse_mm"])
    save_csv(os.path.join(args.out, "heldout_validation_metrics.csv"), val_rows,
            ["k","n","mean_mm","median_mm","rmse_mm","std_mm","p90_mm","p95_mm","max_mm",
             "u_mae","v_mae","u_rmse","v_rmse"])
    save_csv(os.path.join(args.out, "heldout_validation_baselines.csv"), base_rows,
            ["k","learned_median_mm","cv_median_mm","zero_median_mm",
             "improvement_vs_cv_pct","improvement_vs_zero_pct"])
    save_csv(os.path.join(args.out, "heldout_common_validation_comparison.csv"), common_rows,
            ["k","n","mean_mm","median_mm","rmse_mm","p90_mm","p95_mm"])
    save_csv(os.path.join(args.out, "heldout_validation_by_track.csv"), track_rows,
            ["k","recording","track_id","object","n","learned_median_mm","learned_rmse_mm",
             "cv_median_mm","zero_median_mm"])
    save_csv(os.path.join(args.out, "heldout_validation_by_object.csv"), obj_rows,
            ["k","object","n","mean_mm","median_mm","rmse_mm"])
    save_csv(os.path.join(args.out, "heldout_train_vs_validation.csv"), tvv_rows,
            ["k","n_train","train_median_mm","train_rmse_mm","n_val","val_median_mm","val_rmse_mm"])

    # ---- plots ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ks = sorted(per_k)
    fig, ax = plt.subplots(figsize=(7,4.5))
    ax.plot(ks, [r[3] for r in common_rows], "o-")
    ax.set_xlabel("k"); ax.set_ylabel("common-target validation median error (mm)")
    ax.set_title("VALIDATION median error vs k")
    fig.savefig(os.path.join(plots_dir, "val_median_vs_k.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7,4.5))
    ax.plot(ks, [r[4] for r in common_rows], "o-", label="RMSE")
    ax.plot(ks, [r[6] for r in common_rows], "s-", label="P95")
    ax.set_xlabel("k"); ax.set_ylabel("mm"); ax.legend()
    ax.set_title("VALIDATION RMSE and P95 vs k")
    fig.savefig(os.path.join(plots_dir, "val_rmse_p95_vs_k.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7,4.5))
    ax.plot(ks, [r[1] for r in base_rows], "o-", label="learned")
    ax.plot(ks, [r[2] for r in base_rows], "s-", label="constant velocity")
    ax.plot(ks, [r[3] for r in base_rows], "^-", label="zero motion")
    ax.set_xlabel("k"); ax.set_ylabel("median error (mm)"); ax.legend()
    ax.set_title("VALIDATION: learned vs baselines")
    fig.savefig(os.path.join(plots_dir, "val_vs_baselines.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7,4.5))
    ax.plot(ks, [r[2] for r in tvv_rows], "o-", label="train")
    ax.plot(ks, [r[5] for r in tvv_rows], "s-", label="validation")
    ax.set_xlabel("k"); ax.set_ylabel("median error (mm)"); ax.legend()
    ax.set_title("TRAIN vs VALIDATION median error")
    fig.savefig(os.path.join(plots_dir, "train_vs_val.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- report ----
    lines = ["# HELD-OUT TRAJECTORY TRAINING REPORT\n"]
    lines.append("**Same model as the original POC** (ridge-regularized linear "
                "regression, l2=1e-2, iters=3000, lr=0.1, unchanged) trained on "
                "the NEW track-independent split. Only the data source differs. "
                "**Normalization (mu/sd) computed from TRAIN only, verified by "
                "construction** (train_linreg is called with F_train/Y_train "
                "only; F_val/F_test never appear in that call).\n\n")
    lines.extend(x if x.startswith("#") else x+"\n" for x in REPORT)
    report_path = os.path.join(args.out, "heldout_training_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"\n\nFull report: {report_path}")

    print("\n" + "#"*70)
    print("TEST SET NOT EVALUATED -- REMAINS LOCKED")
    print("#"*70)


if __name__ == "__main__":
    main()