"""
validate_trajectory_model.py — evaluates the EXISTING, FROZEN trajectory
regression model(s) against baselines. Trains nothing. Modifies no
existing file.

INSPECTION FINDINGS:
  Model type: multivariate LINEAR regression, 2 outputs (delta_u, delta_v),
  fit by full-batch GRADIENT DESCENT with L2 (ridge) regularization
  (train_model2.train_linreg: l2=1e-2, iters=3000, lr=0.1) -- this is
  ridge-regularized linear regression, NOT plain closed-form OLS. Verified
  by reading the actual function signature/body, not assumed.

  Prediction target: DISPLACEMENT (delta_u, delta_v) = future position
  minus current position -- NOT absolute future position directly.
  Absolute future position is reconstructed as U_test + predicted
  displacement (same reconstruction the project's own plot.py already
  uses). Verified from make_dataset2.py's Y.append([u_future-u_now,
  v_future-v_now]) and train_model2.predict_disp()'s naming/behavior.

  Saved model format: results/traj_model_k{k}_h{horizon}.npz containing
  W, b, mu, sd (TRAINING-derived normalization, already saved -- reused
  directly, never recomputed here), feature_names, k, horizon (horizon
  IS encoded in the saved file, verified by loading and checking keys).
  One separate frozen model file exists PER (k, horizon) combination.

REUSED DIRECTLY (not reimplemented): train_model2.predict_disp(),
train_model2.baseline_cv() (the EXISTING constant-velocity baseline --
confirmed present in the codebase, reused rather than reinvented, per
instruction #7). Zero-motion baseline (Section 6) has no prior
implementation in this repo -- it is a one-line, undebatable definition
(predicted displacement = (0,0)), implemented directly below, not
"reinventing a model."

NO RETRAINING GUARANTEE: this script contains zero calls to
train_linreg() or any .fit()-like function. For every k, it FIRST checks
whether results/traj_model_k{k}_h10.npz already exists as a saved file.
If it does not, the script STOPS for that k with an explicit message
telling you to run the existing, unmodified train_model2.py yourself --
it never trains anything to fill the gap.

Usage:
  python validate_trajectory_model.py
  python validate_trajectory_model.py --results-dir results --out validation_results/trajectory_model
  python validate_trajectory_model.py --help
"""
import argparse
import csv
import glob
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
import make_dataset2 as md2

K_VALUES = [1, 3, 5, 10]
HORIZON = 10

# Known object identity, manually established (NOT from classify_cluster()),
# per your own prior confirmation:
#   ball_moving recording -> ball
#   box_moving recording -> box
#   ball_and_box_moving: track 2 -> ball, tracks 1 and 4 -> box
KNOWN_IDENTITY = {
    ("ball_moving", None): "ball",
    ("box_moving", None): "box",
    ("ball_and_box_moving", "2"): "ball",
    ("ball_and_box_moving", "1"): "box",
    ("ball_and_box_moving", "4"): "box",
}


def object_identity(rec, tid):
    if (rec, tid) in KNOWN_IDENTITY:
        return KNOWN_IDENTITY[(rec, tid)]
    if (rec, None) in KNOWN_IDENTITY:
        return KNOWN_IDENTITY[(rec, None)]
    return "unknown"


def pctl(arr, p):
    return float(np.percentile(arr, p)) if len(arr) else float("nan")


def full_stats(err):
    err = np.asarray(err, dtype=float)
    if len(err) == 0:
        return dict(n=0)
    return dict(n=int(len(err)), mean=float(err.mean()), median=float(np.median(err)),
               rmse=float(np.sqrt((err**2).mean())), std=float(err.std()),
               min=float(err.min()), p25=pctl(err,25), p50=pctl(err,50),
               p75=pctl(err,75), p90=pctl(err,90), p95=pctl(err,95),
               p99=(pctl(err,99) if len(err)>=100 else float("nan")), max=float(err.max()))


def save_csv(path, rows, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


REPORT = []
def note(md): REPORT.append(md)
def heading(md):
    print("\n" + "="*78); print(md); print("="*78)
    REPORT.append(md)


# ===========================================================================
def load_frozen(k, horizon, results_dir):
    model_path = os.path.join(results_dir, f"traj_model_k{k}_h{horizon}.npz")
    dataset_path = os.path.join(results_dir, f"traj_dataset_k{k}_h{horizon}.npz")
    if not os.path.exists(model_path):
        print(f"\nk={k}: STOP -- {model_path} does not exist. This script "
             f"does NOT train models. Run the existing, unmodified command "
             f"yourself first:\n    python train_model2.py --k {k} --h {horizon}\n"
             f"Then re-run this validator.")
        return None, None
    if not os.path.exists(dataset_path):
        print(f"\nk={k}: STOP -- {dataset_path} does not exist. Run "
             f"make_dataset2.py --k {k} --h {horizon} first (existing, "
             f"unmodified script).")
        return None, None
    model = np.load(model_path, allow_pickle=True)
    dataset = np.load(dataset_path, allow_pickle=True)
    required = ["W", "b", "mu", "sd", "feature_names", "k", "horizon"]
    missing = [r for r in required if r not in model]
    if missing:
        print(f"\nk={k}: STOP -- model file missing required keys {missing}. "
             f"Cannot guarantee frozen evaluation.")
        return None, None
    if int(model["k"]) != k or int(model["horizon"]) != horizon:
        print(f"\nk={k}: STOP -- model file's OWN saved metadata says "
             f"k={int(model['k'])}, horizon={int(model['horizon'])}, which "
             f"does not match the requested k={k}, horizon={horizon}. "
             f"Filename/content mismatch -- not proceeding.")
        return None, None
    return model, dataset


def verify_compatibility(k, model, dataset):
    feature_names = [str(n) for n in dataset["feature_names"]]
    X_test = dataset["X_test"]
    ok = True
    if X_test.shape[1] != 9:
        print(f"  COMPATIBILITY FAIL: expected feature dim 9, got {X_test.shape[1]}"); ok = False
    if dataset["Y_test"].shape[1] != 2:
        print(f"  COMPATIBILITY FAIL: expected target dim 2, got {dataset['Y_test'].shape[1]}"); ok = False
    if list(feature_names) != list(model["feature_names"]):
        print(f"  COMPATIBILITY FAIL: feature order mismatch: dataset={feature_names} "
             f"model={list(model['feature_names'])}"); ok = False
    if model["mu"].shape[0] != X_test.shape[1]:
        print(f"  COMPATIBILITY FAIL: normalization dim mismatch"); ok = False
    nan_ct = int(np.isnan(X_test).sum() + np.isnan(dataset["Y_test"]).sum())
    inf_ct = int(np.isinf(X_test).sum() + np.isinf(dataset["Y_test"]).sum())
    if nan_ct or inf_ct:
        print(f"  COMPATIBILITY FAIL: NaN={nan_ct} Inf={inf_ct} in test data"); ok = False
    if ok:
        print(f"  compatibility OK: feature_dim=9, target_dim=2, horizon={int(model['horizon'])}, "
             f"feature order matches, normalization dims match, no NaN/Inf.")
    return ok


def zero_motion_baseline(Y_test):
    """predicted displacement = (0,0) -- one-line, undebatable definition."""
    return np.zeros_like(Y_test)


def euclid(pred_disp, true_disp):
    return np.hypot(pred_disp[:,0]-true_disp[:,0], pred_disp[:,1]-true_disp[:,1])


def coord_errors(pred_disp, true_disp):
    du_err = pred_disp[:,0]-true_disp[:,0]
    dv_err = pred_disp[:,1]-true_disp[:,1]
    return dict(u_mae=float(np.mean(np.abs(du_err))), v_mae=float(np.mean(np.abs(dv_err))),
               u_rmse=float(np.sqrt(np.mean(du_err**2))), v_rmse=float(np.sqrt(np.mean(dv_err**2))),
               u_signed_mean=float(np.mean(du_err)), v_signed_mean=float(np.mean(dv_err)))


def main():
    p = argparse.ArgumentParser(
        description="Evaluate the EXISTING FROZEN trajectory model(s) against "
                    "zero-motion and constant-velocity baselines. Trains nothing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--out", type=str, default="validation_results/trajectory_model")
    p.add_argument("--horizon", type=int, default=HORIZON)
    args = p.parse_args()

    def _resolve(path):
        return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)
    args.results_dir = _resolve(args.results_dir)
    args.out = _resolve(args.out)
    plots_dir = os.path.join(args.out, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    heading("## MODEL/DATA COMPATIBILITY + LOADING (frozen models only)")
    loaded = {}
    for k in K_VALUES:
        print(f"\nk={k}:")
        model, dataset = load_frozen(k, args.horizon, args.results_dir)
        if model is None:
            continue
        if not verify_compatibility(k, model, dataset):
            print(f"  k={k}: SKIPPING due to compatibility failure above.")
            continue
        loaded[k] = (model, dataset)

    if not loaded:
        print("\nNo frozen models available to evaluate. Stopping -- nothing trained.")
        return

    all_metrics_rows, baseline_rows, by_rec_rows, by_track_rows, by_obj_rows = [], [], [], [], []
    pred_rows, large_error_rows, jump_rows, motion_rows, tvt_rows, resid_rows = [], [], [], [], [], []
    per_k_results = {}

    # load suspicious jumps from the PRIOR trajectory-data validator's output, if present
    jump_path = os.path.join(_PROJECT_ROOT, "validation_results", "trajectory_data", "suspicious_jumps.csv")
    suspicious_jumps = []
    if os.path.exists(jump_path):
        suspicious_jumps = list(csv.DictReader(open(jump_path)))
        print(f"\nLoaded {len(suspicious_jumps)} suspicious jump candidates from "
             f"{jump_path} (prior validator's output, reused not recomputed).")
    else:
        print(f"\nNOTE: {jump_path} not found -- jump-proximity analysis (Section "
             f"14) will be skipped. Run validate_trajectory_data.py first for that "
             f"cross-reference.")

    heading("## PER-k EVALUATION: learned model vs baselines, same test samples")
    for k, (model, dataset) in loaded.items():
        feature_names = [str(n) for n in dataset["feature_names"]]
        X_train, Y_train, U_train = dataset["X_train"], dataset["Y_train"], dataset["U_train"]
        X_test, Y_test, U_test = dataset["X_test"], dataset["Y_test"], dataset["U_test"]
        meta_test = [str(m) for m in dataset["meta_test"]]

        pred_disp = tm2.predict_disp(X_test, model["W"], model["b"], model["mu"], model["sd"])
        cv_disp = tm2.baseline_cv(U_test, X_test, feature_names, args.horizon)
        zero_disp = zero_motion_baseline(Y_test)

        err_model = euclid(pred_disp, Y_test)
        err_cv = euclid(cv_disp, Y_test)
        err_zero = euclid(zero_disp, Y_test)

        s_model, s_cv, s_zero = full_stats(err_model), full_stats(err_cv), full_stats(err_zero)
        cm = coord_errors(pred_disp, Y_test)

        print(f"\nk={k}, horizon={args.horizon}")
        print(f"Learned:  N={s_model['n']} mean={s_model['mean']:.2f} median={s_model['median']:.2f} "
             f"RMSE={s_model['rmse']:.2f} P90={s_model['p90']:.2f} P95={s_model['p95']:.2f}")
        print(f"Constant velocity: mean={s_cv['mean']:.2f} median={s_cv['median']:.2f} RMSE={s_cv['rmse']:.2f}")
        print(f"Zero motion: mean={s_zero['mean']:.2f} median={s_zero['median']:.2f} RMSE={s_zero['rmse']:.2f}")
        imp_cv = 100*(s_cv['median']-s_model['median'])/s_cv['median'] if s_cv['median'] else float("nan")
        imp_zero = 100*(s_zero['median']-s_model['median'])/s_zero['median'] if s_zero['median'] else float("nan")
        print(f"Improvement vs constant velocity: {imp_cv:+.1f}% (median)")
        print(f"Improvement vs zero motion: {imp_zero:+.1f}% (median)")

        for name, s in [("learned", s_model), ("constant_velocity", s_cv), ("zero_motion", s_zero)]:
            baseline_rows.append([k, name] + [s.get(kk, float("nan")) for kk in
                                 ["n","mean","median","rmse","std","min","p25","p50","p75","p90","p95","p99","max"]])
        all_metrics_rows.append([k, args.horizon, s_model["n"], s_model["mean"], s_model["median"],
                                s_model["rmse"], s_model["p90"], s_model["p95"],
                                cm["u_mae"], cm["v_mae"], cm["u_rmse"], cm["v_rmse"],
                                cm["u_signed_mean"], cm["v_signed_mean"],
                                imp_cv, imp_zero])

        # per-sample predictions + metadata
        recs = np.array([m.split(":")[0] for m in meta_test])
        tids = np.array([m.split(":")[1] for m in meta_test])
        frames = np.array([int(m.split(":")[2]) for m in meta_test])
        pred_future = U_test + pred_disp
        true_future = U_test + Y_test
        for i in range(len(meta_test)):
            pred_rows.append([k, recs[i], tids[i], frames[i], frames[i]+args.horizon,
                             f"{true_future[i,0]:.2f}", f"{true_future[i,1]:.2f}",
                             f"{pred_future[i,0]:.2f}", f"{pred_future[i,1]:.2f}",
                             f"{err_model[i]:.3f}", f"{err_cv[i]:.3f}", f"{err_zero[i]:.3f}"])

        # ---- Section 11: by recording ----
        for rec in sorted(set(recs)):
            sel = recs==rec
            sm, sc = full_stats(err_model[sel]), full_stats(err_cv[sel])
            by_rec_rows.append([k, rec, sm["n"], sm["mean"], sm["median"], sm["rmse"],
                               sm["p90"], sm["p95"], sc["median"]])
            print(f"  by recording {rec}: n={sm['n']} model_median={sm['median']:.2f} "
                 f"cv_median={sc['median']:.2f}")

        # ---- Section 12: by track ----
        for rec, tid in sorted(set(zip(recs, tids))):
            sel = (recs==rec) & (tids==tid)
            sm = full_stats(err_model[sel])
            sc_med = float(np.median(err_cv[sel])) if sel.sum() else float("nan")
            by_track_rows.append([k, rec, tid, sm["n"], sm["mean"], sm["median"], sm["rmse"],
                                 sm["p95"], float(np.median(err_zero[sel])) if sel.sum() else float("nan"), sc_med])

        # ---- Section 13: by object identity (manual mapping, NOT classify_cluster) ----
        idents = np.array([object_identity(recs[i], tids[i]) for i in range(len(recs))])
        for obj in ["ball","box"]:
            sel = idents==obj
            if sel.sum():
                sm = full_stats(err_model[sel])
                by_obj_rows.append([k, obj, sm["n"], sm["mean"], sm["median"], sm["rmse"], sm["p95"]])
                print(f"  by object {obj}: n={sm['n']} model_median={sm['median']:.2f}")

        # ---- Section 14: jump proximity ----
        for jr in suspicious_jumps:
            if jr["recording"] not in set(recs):
                continue
            ja, jb = int(jr["frame_a"]), int(jr["frame_b"])
            near = np.where((recs==jr["recording"]) & (tids==jr["track_id"]) &
                           (np.abs(frames - ja) <= 15))[0]
            if len(near):
                local_err = err_model[near]
                jump_rows.append([k, jr["recording"], jr["track_id"], ja, jb,
                                 jr["jump_distance_mm"], len(near),
                                 float(np.median(local_err)), float(s_model["median"])])

        # ---- Section 15: error vs motion characteristics ----
        speed_i, ddu_i, ddv_i = feature_names.index("speed"), feature_names.index("ddu"), feature_names.index("ddv")
        speed, accel = X_test[:,speed_i], np.hypot(X_test[:,ddu_i], X_test[:,ddv_i])
        def corr(a,b):
            return float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else float("nan")
        c_speed, c_accel = corr(speed, err_model), corr(accel, err_model)
        motion_rows.append([k, "speed", c_speed])
        motion_rows.append([k, "acceleration", c_accel])
        print(f"  error correlation: speed={c_speed:.3f} acceleration={c_accel:.3f}")

        # ---- Section 19: train vs test ----
        pred_train = tm2.predict_disp(X_train, model["W"], model["b"], model["mu"], model["sd"])
        err_train = euclid(pred_train, Y_train)
        s_train = full_stats(err_train)
        tvt_rows.append([k, s_train["n"], s_train["mean"], s_train["median"], s_train["rmse"],
                        s_model["n"], s_model["mean"], s_model["median"], s_model["rmse"]])
        print(f"  train: n={s_train['n']} median={s_train['median']:.2f}  vs  "
             f"test: n={s_model['n']} median={s_model['median']:.2f}")

        # ---- Section 20: residual diagnostics ----
        resid_rows.append([k, cm["u_signed_mean"], cm["v_signed_mean"], cm["u_rmse"], cm["v_rmse"]])

        # ---- Section 21: large-error cases (top 5%) ----
        thresh = pctl(err_model, 95)
        large_idx = np.where(err_model >= thresh)[0]
        for i in large_idx:
            avg_du_i, avg_dv_i = feature_names.index("avg_du"), feature_names.index("avg_dv")
            near_jump = any(abs(int(frames[i]) - int(jr["frame_a"])) <= 15 and
                           jr["recording"]==recs[i] for jr in suspicious_jumps)
            large_error_rows.append([k, recs[i], tids[i], frames[i], frames[i]+args.horizon,
                                    f"{true_future[i,0]:.2f}", f"{true_future[i,1]:.2f}",
                                    f"{pred_future[i,0]:.2f}", f"{pred_future[i,1]:.2f}",
                                    f"{err_model[i]:.2f}", f"{X_test[i,speed_i]:.2f}",
                                    f"{X_test[i,avg_du_i]:.2f}", f"{X_test[i,avg_dv_i]:.2f}",
                                    f"{X_test[i,ddu_i]:.2f}", f"{X_test[i,ddv_i]:.2f}",
                                    f"{err_cv[i]:.2f}", f"{err_zero[i]:.2f}", near_jump])

        per_k_results[k] = dict(err_model=err_model, err_cv=err_cv, err_zero=err_zero,
                               meta_test=meta_test, recs=recs, tids=tids, frames=frames,
                               U_test=U_test, Y_test=Y_test, pred_disp=pred_disp,
                               cv_disp=cv_disp, s_model=s_model, s_cv=s_cv, s_zero=s_zero)

    # ---- write CSVs ----
    save_csv(os.path.join(args.out, "trajectory_model_metrics.csv"), all_metrics_rows,
            ["k","horizon","n","mean_mm","median_mm","rmse_mm","p90_mm","p95_mm",
             "u_mae","v_mae","u_rmse","v_rmse","u_signed_mean","v_signed_mean",
             "improvement_vs_cv_pct","improvement_vs_zero_pct"])
    save_csv(os.path.join(args.out, "baseline_comparison.csv"), baseline_rows,
            ["k","method","n","mean_mm","median_mm","rmse_mm","std_mm","min_mm",
             "p25_mm","p50_mm","p75_mm","p90_mm","p95_mm","p99_mm","max_mm"])
    save_csv(os.path.join(args.out, "trajectory_predictions.csv"), pred_rows,
            ["k","recording","track_id","current_frame","target_frame",
             "true_u","true_v","pred_u","pred_v","model_err_mm","cv_err_mm","zero_err_mm"])
    save_csv(os.path.join(args.out, "performance_by_recording.csv"), by_rec_rows,
            ["k","recording","n","mean_mm","median_mm","rmse_mm","p90_mm","p95_mm","cv_median_mm"])
    save_csv(os.path.join(args.out, "performance_by_track.csv"), by_track_rows,
            ["k","recording","track_id","n","mean_mm","median_mm","rmse_mm","p95_mm",
             "zero_median_mm","cv_median_mm"])
    save_csv(os.path.join(args.out, "performance_by_object.csv"), by_obj_rows,
            ["k","object","n","mean_mm","median_mm","rmse_mm","p95_mm"])
    save_csv(os.path.join(args.out, "jump_error_analysis.csv"), jump_rows,
            ["k","recording","track_id","jump_frame_a","jump_frame_b","jump_distance_mm",
             "n_samples_within_15frames","local_median_err_mm","overall_median_err_mm"])
    save_csv(os.path.join(args.out, "motion_error_analysis.csv"), motion_rows,
            ["k","motion_variable","correlation_with_error"])
    save_csv(os.path.join(args.out, "train_vs_test.csv"), tvt_rows,
            ["k","n_train","train_mean_mm","train_median_mm","train_rmse_mm",
             "n_test","test_mean_mm","test_median_mm","test_rmse_mm"])
    save_csv(os.path.join(args.out, "residual_diagnostics.csv"), resid_rows,
            ["k","u_signed_mean_mm","v_signed_mean_mm","u_rmse_mm","v_rmse_mm"])
    save_csv(os.path.join(args.out, "large_error_cases.csv"), large_error_rows,
            ["k","recording","track_id","current_frame","target_frame","true_u","true_v",
             "pred_u","pred_v","error_mm","speed","avg_du","avg_dv","ddu","ddv",
             "cv_error_mm","zero_error_mm","near_suspicious_jump"])

    # ---- Section 9: common-target subset comparison ----
    heading("## COMMON-TARGET SUBSET COMPARISON (fair comparison across k)")
    common_rows = []
    if len(per_k_results) >= 2:
        key_sets = {}
        for k, r in per_k_results.items():
            key_sets[k] = set(zip(r["recs"].tolist(), r["tids"].tolist(), r["frames"].tolist()))
        common = set.intersection(*key_sets.values())
        print(f"Common (recording, track, current_frame) targets across all "
             f"loaded k values: {len(common)}")
        native_medians, common_medians = {}, {}
        for k, r in per_k_results.items():
            native_medians[k] = r["s_model"]["median"]
            mask = np.array([(rec,tid,fr) in common for rec,tid,fr in
                            zip(r["recs"], r["tids"], r["frames"])])
            common_err = r["err_model"][mask]
            cm_stats = full_stats(common_err)
            common_medians[k] = cm_stats["median"]
            print(f"  k={k}: native_n={r['s_model']['n']} native_median={native_medians[k]:.2f}  |  "
                 f"common_n={cm_stats['n']} common_median={cm_stats['median']:.2f}")
            common_rows.append([k, r["s_model"]["n"], native_medians[k], cm_stats["n"], cm_stats["median"]])
        best_native = min(native_medians, key=native_medians.get)
        best_common = min(common_medians, key=common_medians.get)
        print(f"\nBest k by native-set median: k={best_native} ({native_medians[best_native]:.2f}mm)")
        print(f"Best k by common-target median: k={best_common} ({common_medians[best_common]:.2f}mm)")
        note(f"Native test sets differ in size across k (larger k requires more "
            f"preceding history, so fewer usable targets near track starts). "
            f"Common-target subset removes this confound. "
            f"{'Same k wins both comparisons.' if best_native==best_common else f'DIFFERENT k wins: native favors k={best_native}, common-target favors k={best_common} -- the native-set comparison alone would have been misleading.'}")
    save_csv(os.path.join(args.out, "common_target_k_comparison.csv"), common_rows,
            ["k","native_n","native_median_mm","common_n","common_median_mm"])

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks_sorted = sorted(per_k_results.keys())
    fig, ax = plt.subplots(figsize=(8,5))
    for name, key in [("learned","err_model"),("constant_velocity","err_cv"),("zero_motion","err_zero")]:
        ax.plot(ks_sorted, [per_k_results[k][key].mean() for k in ks_sorted], "o-", label=name)
    ax.set_xlabel("k"); ax.set_ylabel("mean error (mm)"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title(f"Mean position error vs k (horizon={args.horizon})")
    fig.savefig(os.path.join(plots_dir, "mean_error_vs_k.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    for k in ks_sorted:
        r = per_k_results[k]
        fig, ax = plt.subplots(figsize=(8,5))
        bins = np.linspace(0, max(r["err_model"].max(), r["err_cv"].max(), r["err_zero"].max()), 50)
        ax.hist(r["err_zero"], bins=bins, alpha=0.4, label="zero motion", color="grey")
        ax.hist(r["err_cv"], bins=bins, alpha=0.5, label="constant velocity", color="orange")
        ax.hist(r["err_model"], bins=bins, alpha=0.6, label="learned model", color="tab:blue")
        ax.set_xlabel("error (mm)"); ax.set_ylabel("count"); ax.legend()
        ax.set_title(f"Error histogram, k={k}")
        fig.savefig(os.path.join(plots_dir, f"error_histogram_k{k}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8,5))
        for name, err in [("learned", r["err_model"]), ("constant_velocity", r["err_cv"]),
                          ("zero_motion", r["err_zero"])]:
            xs = np.sort(err); ys = np.arange(1,len(xs)+1)/len(xs)
            ax.plot(xs, ys, label=name)
        ax.set_xlabel("error (mm)"); ax.set_ylabel("cumulative fraction"); ax.legend()
        ax.set_title(f"Empirical CDF of error, k={k}")
        fig.savefig(os.path.join(plots_dir, f"error_cdf_k{k}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # error vs frame, per track
        fig, ax = plt.subplots(figsize=(10,4.5))
        for rec, tid in sorted(set(zip(r["recs"], r["tids"]))):
            sel = (r["recs"]==rec) & (r["tids"]==tid)
            order = np.argsort(r["frames"][sel])
            ax.plot(r["frames"][sel][order], r["err_model"][sel][order], ".-", ms=3,
                   label=f"{rec}:{tid}", alpha=0.7)
        ax.set_xlabel("current frame"); ax.set_ylabel("error (mm)")
        ax.set_title(f"Error over time, k={k}"); ax.legend(fontsize=7, ncol=2)
        fig.savefig(os.path.join(plots_dir, f"error_over_time_k{k}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # predicted-vs-true trajectory: best and worst track, using k=5 if available else first loaded
    ref_k = 5 if 5 in per_k_results else ks_sorted[0]
    r = per_k_results[ref_k]
    track_medians = {}
    for rec, tid in sorted(set(zip(r["recs"], r["tids"]))):
        sel = (r["recs"]==rec) & (r["tids"]==tid)
        track_medians[(rec,tid)] = float(np.median(r["err_model"][sel]))
    if track_medians:
        best_track = min(track_medians, key=track_medians.get)
        worst_track = max(track_medians, key=track_medians.get)
        for label, (rec, tid) in [("best", best_track), ("worst", worst_track)]:
            sel = (r["recs"]==rec) & (r["tids"]==tid)
            order = np.argsort(r["frames"][sel])
            true_fut = (r["U_test"]+r["Y_test"])[sel][order]
            pred_fut = (r["U_test"]+r["pred_disp"])[sel][order]
            cv_fut = (r["U_test"]+r["cv_disp"])[sel][order]
            fig, ax = plt.subplots(figsize=(7,6))
            ax.plot(true_fut[:,0], true_fut[:,1], "g.-", ms=4, label="true future", alpha=0.7)
            ax.plot(pred_fut[:,0], pred_fut[:,1], "b.-", ms=4, label="learned model predicted", alpha=0.7)
            ax.plot(cv_fut[:,0], cv_fut[:,1], "r.-", ms=4, label="constant velocity predicted", alpha=0.5)
            ax.set_xlabel("u (mm)"); ax.set_ylabel("v (mm)"); ax.legend(); ax.set_aspect("equal")
            ax.set_title(f"{label.upper()} track: {rec}:{tid}, k={ref_k} "
                        f"(median err={track_medians[(rec,tid)]:.1f}mm)")
            fig.savefig(os.path.join(plots_dir, f"trajectory_{label}_track.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

    # ---- report ----
    lines = ["# TRAJECTORY MODEL VALIDATION REPORT\n"]
    lines.append("**Split limitation, preserved as instructed:** the current "
                "trajectory split is a TIME split WITHIN each track. All 8 "
                "tracks contribute to both train and test. This evaluation "
                "therefore answers: **how well does the model predict LATER "
                "portions of the existing tracks from EARLIER portions used "
                "in training?** It does NOT yet prove generalization to "
                "completely unseen tracks. No split change was made here.\n\n")
    lines.extend(x if x.startswith("#") else x+"\n" for x in REPORT)

    lines.append("\n## Answers to the 12 report questions\n")
    if per_k_results:
        any_beats_zero = any(per_k_results[k]["s_model"]["median"] < per_k_results[k]["s_zero"]["median"] for k in per_k_results)
        any_beats_cv = any(per_k_results[k]["s_model"]["median"] < per_k_results[k]["s_cv"]["median"] for k in per_k_results)
        lines.append(f"1. Beats zero-motion: {'YES' if any_beats_zero else 'NO'} at the k values evaluated (see baseline_comparison.csv for exact numbers per k).\n")
        lines.append(f"2. Beats constant-velocity: {'YES' if any_beats_cv else 'NO'} (see baseline_comparison.csv).\n")
        lines.append("3. Magnitude of improvement: see 'improvement_vs_cv_pct'/'improvement_vs_zero_pct' columns in trajectory_model_metrics.csv.\n")
        lines.append("4-6. Does more history help / best k / common-target agreement: see COMMON-TARGET SUBSET COMPARISON section above.\n")
        lines.append("7. Ball vs box: see performance_by_object.csv.\n")
        lines.append("8. Hardest recordings/tracks: see performance_by_recording.csv and performance_by_track.csv.\n")
        lines.append("9. Large errors vs suspicious jumps: see jump_error_analysis.csv.\n")
        lines.append("10. Systematic u/v bias: see residual_diagnostics.csv (u_signed_mean/v_signed_mean; non-zero indicates directional bias).\n")
        lines.append("11. Train vs test: see train_vs_test.csv.\n")
        lines.append("12. Large-error tail reduction: compare p95/p99/max columns in baseline_comparison.csv between 'learned' and the two baselines.\n")

    report_path = os.path.join(args.out, "trajectory_model_validation_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines))

    print("\n" + "#"*70)
    print(f"Full report: {report_path}")
    print("#"*70)


if __name__ == "__main__":
    main()