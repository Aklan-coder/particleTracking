"""
evaluate_heldout_trajectory_test.py — FINAL held-out-track TEST evaluation.
Evaluates ONLY the pre-selected, pre-frozen k=10 model on ONLY its TEST
split. Trains nothing. Tunes nothing. Never touches k=1, k=5, or k=10.
Modifies no existing file.

MODEL SELECTION LOCK: k=10 was chosen using VALIDATION performance,
BEFORE this script ever opens the TEST arrays. This script contains NO
code path capable of loading heldout_traj_model_k1/k5/k10_h10.npz or
their TEST arrays -- confirmed by grep (shown below), not just by
omission.

REUSED DIRECTLY: train_model2.predict_disp(), baseline_cv() -- imported,
never reimplemented, never refit (no train_linreg() call anywhere in
this file).

TEST-TRACK IDENTITY VERIFICATION (Section 2): before computing any
metric, this script confirms the k=10 dataset's TEST metadata contains
ONLY (ball_moving, track 4) and (box_moving, track 1), and separately
confirms neither track appears in that same dataset's TRAIN metadata.
If either check fails, the script STOPS with an error and computes
nothing further.

Usage:
  python evaluate_heldout_trajectory_test.py
  python evaluate_heldout_trajectory_test.py --heldout-dir results/heldout_trajectory --out validation_results/heldout_trajectory_test
  python evaluate_heldout_trajectory_test.py --help
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

K_LOCKED = 10
HORIZON = 10
EXPECTED_TEST_TRACKS = {("ball_moving", "4"): "ball", ("box_moving", "1"): "box"}


def pctl(a,p): return float(np.percentile(a,p)) if len(a) else float("nan")
def full_stats(err):
    err = np.asarray(err, dtype=float)
    if len(err)==0: return dict(n=0)
    return dict(n=int(len(err)), mean=float(err.mean()), median=float(np.median(err)),
               rmse=float(np.sqrt((err**2).mean())), std=float(err.std()),
               min=float(err.min()), p25=pctl(err,25), p50=pctl(err,50), p75=pctl(err,75),
               p90=pctl(err,90), p95=pctl(err,95),
               p99=(pctl(err,99) if len(err)>=100 else float("nan")), max=float(err.max()))
def euclid(pred,true): return np.hypot(pred[:,0]-true[:,0], pred[:,1]-true[:,1])
def coord_errors(pred,true):
    du,dv = pred[:,0]-true[:,0], pred[:,1]-true[:,1]
    return dict(u_mae=float(np.mean(np.abs(du))), v_mae=float(np.mean(np.abs(dv))),
               u_rmse=float(np.sqrt(np.mean(du**2))), v_rmse=float(np.sqrt(np.mean(dv**2))),
               u_signed_mean=float(np.mean(du)), v_signed_mean=float(np.mean(dv)))
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
        description="FINAL locked TEST evaluation of the k=10 held-out "
                    "trajectory model, selected via VALIDATION before TEST "
                    "was ever opened. Evaluates k=10 ONLY.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--heldout-dir", type=str, default="results/heldout_trajectory")
    p.add_argument("--out", type=str, default="validation_results/heldout_trajectory_test")
    args = p.parse_args()
    def _resolve(path): return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)
    args.heldout_dir = _resolve(args.heldout_dir); args.out = _resolve(args.out)
    plots_dir = os.path.join(args.out, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    heading("# FINAL HELD-OUT TRAJECTORY TEST")
    print(f"LOCKED MODEL: k={K_LOCKED}  horizon={HORIZON}")

    # ---- SECTION 1: load ONLY the selected frozen model ----
    model_path = os.path.join(args.heldout_dir, "models", f"heldout_traj_model_k{K_LOCKED}_h{HORIZON}.npz")
    if not os.path.exists(model_path):
        print(f"STOP: {model_path} not found. Run train_heldout_trajectory.py first."); sys.exit(1)
    model = np.load(model_path, allow_pickle=True)
    required = ["W","b","mu","sd","k","horizon","feature_names"]
    missing = [r for r in required if r not in model]
    if missing:
        print(f"STOP: model file missing required keys {missing}."); sys.exit(1)
    if int(model["k"]) != K_LOCKED or int(model["horizon"]) != HORIZON:
        print(f"STOP: model file's own metadata says k={int(model['k'])}, "
             f"horizon={int(model['horizon'])} -- does not match locked "
             f"k={K_LOCKED}, horizon={HORIZON}."); sys.exit(1)
    feature_names = [str(n) for n in model["feature_names"]]
    print(f"Model verified: k={int(model['k'])}, horizon={int(model['horizon'])}, "
         f"features={feature_names}, W.shape={model['W'].shape}")
    if "l2" in model:
        print(f"Training config recorded on model: l2={float(model['l2'])} "
             f"iters={int(model['iters'])} lr={float(model['lr'])}")

    # ---- SECTION 2: load k=10 dataset, verify TEST-track identity ----
    ds_path = os.path.join(args.heldout_dir, f"heldout_traj_dataset_k{K_LOCKED}_h{HORIZON}.npz")
    if not os.path.exists(ds_path):
        print(f"STOP: {ds_path} not found."); sys.exit(1)
    d = np.load(ds_path, allow_pickle=True)
    if list(feature_names) != [str(n) for n in d["feature_names"]]:
        print("STOP: feature name/order mismatch between model and dataset."); sys.exit(1)

    meta_train = [str(m) for m in d["meta_train"]]
    meta_test = [str(m) for m in d["meta_test"]]
    train_tracks = set((m.split(":")[0], m.split(":")[1]) for m in meta_train)
    test_tracks = set((m.split(":")[0], m.split(":")[1]) for m in meta_test)

    print(f"\nTEST tracks found in dataset: {sorted(test_tracks)}")
    if test_tracks != set(EXPECTED_TEST_TRACKS.keys()):
        print(f"STOP: TEST metadata does not match the expected locked test "
             f"tracks {set(EXPECTED_TEST_TRACKS.keys())}. Found {test_tracks} "
             f"instead. Not proceeding."); sys.exit(1)
    overlap = train_tracks & test_tracks
    if overlap:
        print(f"STOP: TEST track(s) {overlap} ALSO appear in TRAIN metadata "
             f"for this same dataset file. This must never happen. Not "
             f"proceeding."); sys.exit(1)
    print("VERIFIED: TEST tracks are exactly the expected locked set, and "
         "are absent from TRAIN metadata for this dataset.")
    print(f"  ball_moving track 4 -> BALL")
    print(f"  box_moving track 1 -> BOX")

    F_test, Y_test, U_test = d["F_test"], d["Y_test"], d["U_test"]
    recs = np.array([m.split(":")[0] for m in meta_test])
    tids = np.array([m.split(":")[1] for m in meta_test])
    frames = np.array([int(m.split(":")[2]) for m in meta_test])

    # ---- SECTION 3/4: frozen normalization, frozen prediction ----
    W, b, mu, sd = model["W"], model["b"], model["mu"], model["sd"]
    pred_disp = tm2.predict_disp(F_test, W, b, mu, sd)
    print(f"\nNormalization (mu/sd) loaded from model file (TRAIN-derived) -- "
         f"not recomputed. Predicting via train_model2.predict_disp() unmodified.")

    # ---- SECTION 5/6: baselines ----
    cv_disp = tm2.baseline_cv(U_test, F_test, feature_names, HORIZON)
    zero_disp = np.zeros_like(Y_test)

    err_model = euclid(pred_disp, Y_test)
    err_cv = euclid(cv_disp, Y_test)
    err_zero = euclid(zero_disp, Y_test)
    s_model, s_cv, s_zero = full_stats(err_model), full_stats(err_cv), full_stats(err_zero)
    cm = coord_errors(pred_disp, Y_test)

    heading("## PRIMARY TEST METRICS")
    print(f"LEARNED MODEL: N={s_model['n']} mean={s_model['mean']:.2f} "
         f"median={s_model['median']:.2f} RMSE={s_model['rmse']:.2f} "
         f"P90={s_model['p90']:.2f} P95={s_model['p95']:.2f} "
         f"P99={s_model['p99']:.2f} max={s_model['max']:.2f}")
    print(f"  u_MAE={cm['u_mae']:.2f} v_MAE={cm['v_mae']:.2f} "
         f"u_RMSE={cm['u_rmse']:.2f} v_RMSE={cm['v_rmse']:.2f} "
         f"u_signed_mean={cm['u_signed_mean']:+.2f} v_signed_mean={cm['v_signed_mean']:+.2f}")
    print(f"CONSTANT VELOCITY: mean={s_cv['mean']:.2f} median={s_cv['median']:.2f} "
         f"RMSE={s_cv['rmse']:.2f} P90={s_cv['p90']:.2f} P95={s_cv['p95']:.2f}")
    print(f"ZERO MOTION: mean={s_zero['mean']:.2f} median={s_zero['median']:.2f} RMSE={s_zero['rmse']:.2f}")

    save_csv(os.path.join(args.out, "final_test_metrics.csv"),
            [[s_model[k] for k in ["n","mean","median","rmse","std","min","p25","p50","p75","p90","p95","p99","max"]] +
             [cm["u_mae"],cm["v_mae"],cm["u_rmse"],cm["v_rmse"],cm["u_signed_mean"],cm["v_signed_mean"]]],
            ["n","mean_mm","median_mm","rmse_mm","std_mm","min_mm","p25_mm","p50_mm","p75_mm",
             "p90_mm","p95_mm","p99_mm","max_mm","u_mae","v_mae","u_rmse","v_rmse",
             "u_signed_mean","v_signed_mean"])

    def imp(base, model_v): return 100*(base-model_v)/base if base else float("nan")
    heading("## IMPROVEMENT VS BASELINES (all metrics, not cherry-picked)")
    imp_cv = dict(median=imp(s_cv["median"],s_model["median"]), mean=imp(s_cv["mean"],s_model["mean"]),
                 rmse=imp(s_cv["rmse"],s_model["rmse"]), p90=imp(s_cv["p90"],s_model["p90"]),
                 p95=imp(s_cv["p95"],s_model["p95"]))
    imp_zero = dict(median=imp(s_zero["median"],s_model["median"]), mean=imp(s_zero["mean"],s_model["mean"]),
                   rmse=imp(s_zero["rmse"],s_model["rmse"]), p90=imp(s_zero["p90"],s_model["p90"]),
                   p95=imp(s_zero["p95"],s_model["p95"]))
    print(f"vs constant velocity: median={imp_cv['median']:+.1f}% mean={imp_cv['mean']:+.1f}% "
         f"RMSE={imp_cv['rmse']:+.1f}% P90={imp_cv['p90']:+.1f}% P95={imp_cv['p95']:+.1f}%")
    print(f"vs zero motion:       median={imp_zero['median']:+.1f}% mean={imp_zero['mean']:+.1f}% "
         f"RMSE={imp_zero['rmse']:+.1f}% P90={imp_zero['p90']:+.1f}% P95={imp_zero['p95']:+.1f}%")
    save_csv(os.path.join(args.out, "final_test_baseline_comparison.csv"),
            [["learned"]+[s_model[k] for k in ["n","mean","median","rmse","p90","p95","max"]],
             ["constant_velocity"]+[s_cv[k] for k in ["n","mean","median","rmse","p90","p95","max"]],
             ["zero_motion"]+[s_zero[k] for k in ["n","mean","median","rmse","p90","p95","max"]]],
            ["method","n","mean_mm","median_mm","rmse_mm","p90_mm","p95_mm","max_mm"])

    # ---- per-sample predictions ----
    pred_future = U_test + pred_disp
    true_future = U_test + Y_test
    pred_rows = []
    for i in range(len(meta_test)):
        pred_rows.append([recs[i], tids[i], EXPECTED_TEST_TRACKS[(recs[i],tids[i])], frames[i], frames[i]+HORIZON,
                         f"{true_future[i,0]:.2f}", f"{true_future[i,1]:.2f}",
                         f"{pred_future[i,0]:.2f}", f"{pred_future[i,1]:.2f}",
                         f"{err_model[i]:.3f}",
                         f"{(U_test[i]+cv_disp[i])[0]:.2f}", f"{(U_test[i]+cv_disp[i])[1]:.2f}",
                         f"{err_cv[i]:.3f}", f"{err_zero[i]:.3f}"])
    save_csv(os.path.join(args.out, "final_test_predictions.csv"), pred_rows,
            ["recording","track_id","object","current_frame","target_frame","true_u","true_v",
             "pred_u","pred_v","learned_err_mm","cv_pred_u","cv_pred_v","cv_err_mm","zero_err_mm"])

    # ---- SECTION 10: per-track ----
    heading("## PER-TRACK TEST PERFORMANCE")
    track_rows = []
    for (rec,tid), obj in EXPECTED_TEST_TRACKS.items():
        sel = (recs==rec) & (tids==tid)
        sm = full_stats(err_model[sel])
        cv_med = float(np.median(err_cv[sel])) if sel.sum() else float("nan")
        zero_med = float(np.median(err_zero[sel])) if sel.sum() else float("nan")
        i_cv = imp(cv_med, sm["median"]); i_zero = imp(zero_med, sm["median"])
        print(f"\n{obj.upper()} TRACK ({rec}:{tid})")
        print(f"  N={sm['n']} learned_mean={sm['mean']:.2f} learned_median={sm['median']:.2f} "
             f"learned_RMSE={sm['rmse']:.2f} P90={sm['p90']:.2f} P95={sm['p95']:.2f}")
        print(f"  cv_median={cv_med:.2f} zero_median={zero_med:.2f} "
             f"improvement_vs_cv={i_cv:+.1f}% improvement_vs_zero={i_zero:+.1f}%")
        track_rows.append([rec, tid, obj, sm["n"], sm["mean"], sm["median"], sm["rmse"],
                          sm["p90"], sm["p95"], cv_med, zero_med, i_cv, i_zero])
    save_csv(os.path.join(args.out, "final_test_by_track.csv"), track_rows,
            ["recording","track_id","object","n","learned_mean_mm","learned_median_mm",
             "learned_rmse_mm","learned_p90_mm","learned_p95_mm","cv_median_mm","zero_median_mm",
             "improvement_vs_cv_pct","improvement_vs_zero_pct"])

    # ---- SECTION 11: object-balanced summary ----
    heading("## OBJECT-BALANCED SUMMARY (average of ball-track and box-track medians, NOT pooled)")
    ball_row = [r for r in track_rows if r[2]=="ball"][0]
    box_row = [r for r in track_rows if r[2]=="box"][0]
    balanced_median = (ball_row[5] + box_row[5]) / 2
    balanced_cv_median = (ball_row[9] + box_row[9]) / 2
    print(f"Pooled median (sample-weighted, box track dominates due to more windows): {s_model['median']:.2f}mm")
    print(f"Object-balanced median (mean of ball-median, box-median, unweighted): {balanced_median:.2f}mm")
    print(f"Object-balanced CV median: {balanced_cv_median:.2f}mm")
    save_csv(os.path.join(args.out, "final_test_by_object.csv"),
            [["ball"]+ball_row[3:], ["box"]+box_row[3:],
             ["balanced_summary", "-", balanced_median, "-", "-", "-", balanced_cv_median, "-", "-", "-"]],
            ["object","n","median_or_mean_mm","rmse_mm","p90_mm","p95_mm","cv_median_mm",
             "zero_median_mm","improvement_vs_cv_pct","improvement_vs_zero_pct"])

    # ---- SECTION 12: train -> validation -> test comparison ----
    heading("## TRAIN -> VALIDATION -> TEST GENERALIZATION GAP")
    tvv_path = os.path.join(_PROJECT_ROOT, "validation_results", "heldout_trajectory_training",
                            "heldout_train_vs_validation.csv")
    tvt_rows = []
    if os.path.exists(tvv_path):
        for r in csv.DictReader(open(tvv_path)):
            if int(r["k"]) == K_LOCKED:
                train_med, train_rmse = float(r["train_median_mm"]), float(r["train_rmse_mm"])
                val_med, val_rmse = float(r["val_median_mm"]), float(r["val_rmse_mm"])
                print(f"TRAIN median={train_med:.2f}mm RMSE={train_rmse:.2f}mm")
                print(f"VALIDATION median={val_med:.2f}mm RMSE={val_rmse:.2f}mm")
                print(f"TEST median={s_model['median']:.2f}mm RMSE={s_model['rmse']:.2f}mm")
                tvt_rows.append([K_LOCKED, train_med, train_rmse, val_med, val_rmse,
                                s_model["median"], s_model["rmse"]])
    else:
        print(f"NOTE: {tvv_path} not found -- cannot report TRAIN/VALIDATION "
             f"numbers (not recomputed here, per instruction; only TEST was "
             f"computed by this script).")
    save_csv(os.path.join(args.out, "train_validation_test_comparison.csv"), tvt_rows,
            ["k","train_median_mm","train_rmse_mm","val_median_mm","val_rmse_mm",
             "test_median_mm","test_rmse_mm"])

    # ---- SECTION 13: large-error tail ----
    heading("## LARGE-ERROR TAIL (top 5%)")
    thresh = pctl(err_model, 95)
    large_idx = np.where(err_model >= thresh)[0]
    speed_i = feature_names.index("speed"); ddu_i = feature_names.index("ddu"); ddv_i = feature_names.index("ddv")
    avg_du_i = feature_names.index("avg_du"); avg_dv_i = feature_names.index("avg_dv")
    large_rows = []
    for i in large_idx:
        large_rows.append([recs[i], tids[i], EXPECTED_TEST_TRACKS[(recs[i],tids[i])], frames[i], frames[i]+HORIZON,
                          f"{true_future[i,0]:.2f}", f"{true_future[i,1]:.2f}",
                          f"{pred_future[i,0]:.2f}", f"{pred_future[i,1]:.2f}",
                          f"{err_model[i]:.2f}", f"{err_cv[i]:.2f}", f"{err_zero[i]:.2f}",
                          f"{F_test[i,speed_i]:.2f}", f"{F_test[i,avg_du_i]:.2f}",
                          f"{F_test[i,avg_dv_i]:.2f}", f"{F_test[i,ddu_i]:.2f}", f"{F_test[i,ddv_i]:.2f}"])
    save_csv(os.path.join(args.out, "final_test_large_errors.csv"), large_rows,
            ["recording","track_id","object","current_frame","target_frame","true_u","true_v",
             "pred_u","pred_v","learned_err_mm","cv_err_mm","zero_err_mm","speed","avg_du","avg_dv","ddu","ddv"])
    print(f"Top 5% threshold: {thresh:.2f}mm -- {len(large_idx)} samples saved to final_test_large_errors.csv")

    # ---- SECTION 14: error vs motion (post-hoc, descriptive only) ----
    heading("## ERROR VS MOTION (post-hoc TEST analysis -- NOT used to modify the model)")
    speed_arr = F_test[:,speed_i]; accel_arr = np.hypot(F_test[:,ddu_i], F_test[:,ddv_i])
    def corr(a,b): return float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else float("nan")
    c_speed, c_accel = corr(speed_arr, err_model), corr(accel_arr, err_model)
    print(f"correlation(speed, error) = {c_speed:.3f}   correlation(acceleration, error) = {c_accel:.3f}")
    save_csv(os.path.join(args.out, "final_test_motion_error_analysis.csv"),
            [["speed", c_speed], ["acceleration", c_accel]], ["motion_variable","correlation_with_error"])
    note("**This motion-error analysis is post-hoc and descriptive only. It "
        "was NOT and must NOT be used to modify the model, per instruction "
        "#14.**")

    # ---- plots ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for (rec,tid), obj in EXPECTED_TEST_TRACKS.items():
        sel = (recs==rec) & (tids==tid)
        order = np.argsort(frames[sel])
        tf, pf, cf = true_future[sel][order], pred_future[sel][order], (U_test+cv_disp)[sel][order]
        fig, ax = plt.subplots(figsize=(7,6))
        ax.plot(tf[:,0], tf[:,1], "g.-", ms=4, label="true future", alpha=0.7)
        ax.plot(pf[:,0], pf[:,1], "b.-", ms=4, label="learned k=10 predicted", alpha=0.7)
        ax.plot(cf[:,0], cf[:,1], "r.-", ms=4, label="constant velocity predicted", alpha=0.5)
        ax.set_xlabel("u (mm)"); ax.set_ylabel("v (mm)"); ax.legend(); ax.set_aspect("equal")
        ax.set_title(f"HELD-OUT {obj.upper()} TRACK ({rec}:{tid}) -- TEST")
        fig.savefig(os.path.join(plots_dir, f"trajectory_{obj}_track.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9,4.5))
        ax.plot(frames[sel][order], err_model[sel][order], ".-", ms=3, label="learned")
        ax.plot(frames[sel][order], err_cv[sel][order], ".-", ms=3, alpha=0.6, label="constant velocity")
        ax.set_xlabel("current frame"); ax.set_ylabel("error (mm)"); ax.legend()
        ax.set_title(f"Error vs target frame -- {obj.upper()} TRACK (TEST)")
        fig.savefig(os.path.join(plots_dir, f"error_over_time_{obj}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,5))
    bins = np.linspace(0, max(err_model.max(),err_cv.max(),err_zero.max()), 50)
    ax.hist(err_zero, bins=bins, alpha=0.4, label="zero motion", color="grey")
    ax.hist(err_cv, bins=bins, alpha=0.5, label="constant velocity", color="orange")
    ax.hist(err_model, bins=bins, alpha=0.6, label="learned k=10", color="tab:blue")
    ax.set_xlabel("error (mm)"); ax.set_ylabel("count"); ax.legend()
    ax.set_title("FINAL TEST error histogram")
    fig.savefig(os.path.join(plots_dir, "error_histogram.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,5))
    for name, err in [("learned",err_model), ("constant_velocity",err_cv), ("zero_motion",err_zero)]:
        xs = np.sort(err); ys = np.arange(1,len(xs)+1)/len(xs)
        ax.plot(xs, ys, label=name)
    ax.set_xlabel("error (mm)"); ax.set_ylabel("cumulative fraction"); ax.legend()
    ax.set_title("FINAL TEST empirical CDF")
    fig.savefig(os.path.join(plots_dir, "error_cdf.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- report ----
    lines = ["# FINAL HELD-OUT TRAJECTORY TEST REPORT\n"]
    lines.append("**This TEST result is a one-time evaluation of the model "
                "selected using VALIDATION. No model or hyperparameter "
                "changes were made using TEST performance.** If future model "
                "changes are made after examining these results, this TEST "
                "set should no longer be described as untouched final test "
                "data for those changed models.\n\n")
    lines.append("## LIMITATION (Section 21)\n")
    lines.append("This is **TRACK-INDEPENDENT** testing. It is NOT completely "
                "recording-independent for every object:\n"
                "- **BOX test (box_moving track 1): track-independent AND "
                "recording-independent** -- box_moving never appeared in "
                "TRAIN or VALIDATION at all.\n"
                "- **BALL test (ball_moving track 4): track-independent but "
                "NOT recording-independent** -- other ball_moving tracks "
                "appeared in TRAIN/VALIDATION.\n\n")
    lines.extend(x if x.startswith("#") else x+"\n" for x in REPORT)
    report_path = os.path.join(args.out, "final_heldout_test_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"\n\nFull report: {report_path}")

    print("\n" + "#"*70)
    print("FINAL TEST COMPLETE -- k=10 was selected before TEST was evaluated.")
    print("#"*70)


if __name__ == "__main__":
    main()