"""
validate_geometry_resolution.py — investigates WHY geometry-based ball-vs-box
identification degrades as cell size h increases, using ONLY real data and
the EXISTING production geometry/classification code. Does not retrain,
does not tune thresholds, does not use synthetic data. Modifies no file.

PRODUCTION FUNCTIONS REUSED, NOT REIMPLEMENTED:
  sweep.run_one()            -- the EXACT phase-averaged identification
                                 experiment that produced results/sweep_results*.csv.
                                 Called verbatim here to recompute/verify
                                 those numbers on real data.
  sweep.degraded_classify()  -- the exact cell-level classification
                                 function (wraps geometry.classify_cluster()
                                 with the h-scaled threshold sweep.py uses).
  sweep._phase_partitions()  -- the exact grid-alignment-averaging logic.
  geometry.ransac_sphere(), geometry.planes_fit() -- called directly (same
                                 threshold formula as degraded_classify) to
                                 get NUMERIC sphere/box residuals for
                                 distribution analysis -- classify_cluster()
                                 itself only returns a text 'detail' string
                                 with both numbers embedded, so calling the
                                 underlying fits directly (with the SAME
                                 threshold) avoids fragile string-parsing
                                 while reusing the identical fitting code.
  fit_static.object_points(), fit_static.load_table(), geometry.load_masked(),
  discretize.Partition -- unchanged.

MATCHED FRAMES ACROSS h: guaranteed structurally, not just claimed -- the
frame file list is built ONCE per recording (identical to how sweep.main()
does it) and reused for every h; only the Partition (grid) changes.

TWO DISTINCT ANALYSES IN THIS SCRIPT, clearly separated:
  (A) OFFICIAL VERIFICATION (Section 2): reuses sweep.run_one() exactly as
      written, phase-averaged (9 grid alignments per h, per sweep.py's own
      PHASES=3), to verify/reproduce the saved sweep_results numbers.
  (B) DETAILED PER-FRAME ANALYSIS (Sections 5-11): uses a SINGLE canonical
      grid phase (zero offset) per matched frame, so each real frame gets
      one interpretable geometric measurement per h. This deliberately
      differs from (A)'s phase-averaged statistic -- stated explicitly,
      not conflated.

Usage:
  python validate_geometry_resolution.py
  python validate_geometry_resolution.py --results-dir results --out validation_results/geometry_resolution
  python validate_geometry_resolution.py --help
"""
import argparse
import csv
import glob
import os
import re
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
from discretize import Partition
from fit_static import load_table, object_points
from geometry import load_masked, ransac_sphere, planes_fit
import sweep as sweep_mod   # reused module: run_one, degraded_classify, _phase_partitions, TRUTH, MIN_CELL_POINTS

H_TEST = [10.0, 20.0, 30.0, 40.0]
RECORDINGS = ["ball_static", "box_static"]


def pctl(a,p): return float(np.percentile(a,p)) if len(a) else float("nan")
def stats_block(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)] if len(arr) else arr
    if len(arr)==0: return dict(n=0, mean=np.nan, median=np.nan, std=np.nan, p5=np.nan, p25=np.nan, p75=np.nan, p95=np.nan)
    return dict(n=int(len(arr)), mean=float(arr.mean()), median=float(np.median(arr)),
               std=float(arr.std()), p5=pctl(arr,5), p25=pctl(arr,25), p75=pctl(arr,75), p95=pctl(arr,95))
def save_csv(path, rows, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w",newline="") as fh:
        w=csv.writer(fh); w.writerow(header); w.writerows(rows)
    return path

REPORT = []
def note(md): REPORT.append(md)
def heading(md):
    print("\n"+"="*78); print(md); print("="*78); REPORT.append(md)


# ===========================================================================
# SECTION 2: verify official sweep numbers by reusing sweep.run_one() verbatim
# ===========================================================================
def verify_official_results(results_dir, out_dir, every):
    heading("## OFFICIAL IDENTIFICATION NUMBERS -- reproduced via sweep.run_one() verbatim")
    bg = np.load(os.path.join(results_dir, "background_median.npy"))
    tf, z = load_table()

    saved = {}
    for fname in ["sweep_results_fit.csv", "sweep_results.csv"]:
        path = os.path.join(results_dir, fname)
        if os.path.exists(path):
            for r in csv.DictReader(open(path)):
                saved[(r["recording"], round(float(r["h"]),1))] = float(r["id_rate"])
            print(f"  loaded saved comparison values from {path}")
            break
    else:
        print("  NOTE: no existing sweep_results*.csv found -- nothing to compare "
             "freshly-computed numbers against; will still compute and report fresh.")

    rows, conf_rows = [], []
    for rec in RECORDINGS:
        files = sorted(glob.glob(os.path.join(_PROJECT_ROOT, "Data", "extracted", rec, "depth", "*.npy")))[::every]
        if not files:
            print(f"  {rec}: no raw depth frames found under Data/extracted/{rec}/depth -- "
                 f"cannot recompute. (Real depth frames required for this script; "
                 f"not present in this environment.)")
            continue
        print(f"\n  {rec}: {len(files)} sampled frames x {sweep_mod.PHASES**2} grid phases")
        for h in H_TEST:
            r = sweep_mod.run_one(rec, files, bg, tf, z, h)
            truth = sweep_mod.TRUTH[rec]
            print(f"    h={h}: n_trials={r['trials']} n_detected={r['detected']} "
                 f"n_correct={r['correct']} id_rate={r['id_rate']:.4f}")
            saved_val = saved.get((rec, h))
            if saved_val is not None:
                diff = abs(r["id_rate"] - saved_val)
                flag = "MATCH" if diff < 0.02 else f"DISCREPANCY ({diff:.3f})"
                print(f"      vs saved sweep_results: {saved_val:.4f}  [{flag}]")
            rows.append([rec, h, r["trials"], r["detected"], r["correct"], r["id_rate"],
                        r["det_rate"], saved_val if saved_val is not None else ""])
            conf_rows.append([rec, h, truth, r["pred_ball"], r["pred_box"], r["pred_unknown"], r["pred_other"]])

    save_csv(os.path.join(out_dir, "geometry_resolution_accuracy.csv"), rows,
            ["recording","h_mm","n_trials","n_detected","n_correct","id_rate",
             "det_rate","saved_sweep_id_rate"])
    save_csv(os.path.join(out_dir, "geometry_resolution_confusion.csv"), conf_rows,
            ["recording","h_mm","truth","pred_ball","pred_box","pred_unknown","pred_other"])
    note("Official identification numbers reproduced via the EXACT sweep.run_one() "
        "function on real data (phase-averaged, matches the production experiment). "
        "Any discrepancy vs previously saved results is reported above, not hidden.")
    return rows


# ===========================================================================
# SECTIONS 5-11: detailed single-phase per-matched-frame analysis
# ===========================================================================
def detailed_per_frame_analysis(results_dir, out_dir, every):
    heading("## DETAILED PER-MATCHED-FRAME ANALYSIS (single grid phase, zero offset)")
    bg = np.load(os.path.join(results_dir, "background_median.npy"))
    tf, z = load_table()

    spatial_rows, sphere_rows, box_rows, margin_rows, failure_rows = [], [], [], [], []
    feature_by_h = {rec: {h: [] for h in H_TEST} for rec in RECORDINGS}
    reference_h = 10.0
    delta_rows = []

    for rec in RECORDINGS:
        files = sorted(glob.glob(os.path.join(_PROJECT_ROOT, "Data", "extracted", rec, "depth", "*.npy")))[::every]
        if not files:
            continue
        truth = sweep_mod.TRUTH[rec]
        print(f"\n{rec} (truth={truth}): {len(files)} matched frames, reused identically across every h")

        ref_values = {}  # frame_index -> reference-h size, for delta computation
        for h in H_TEST:
            part = Partition((z["u_range"][0], z["u_range"][1]), (z["v_range"][0], z["v_range"][1]), h=h)
            thresh_mm = max(config.SPHERE_THRESH_MM, h/2.0)   # IDENTICAL formula to sweep.degraded_classify()
            sizes, sphere_rmses, box_rmses, margins, occupied_cells_list = [], [], [], [], []
            n_correct = n_total = 0

            for fi, f in enumerate(files):
                pts = object_points(load_masked(f), bg)
                if len(pts) < 10:
                    continue
                uvh = tf.to_table(pts)
                clusters, _ = part.clusters(uvh)
                if not clusters:
                    continue
                cl = clusters[0]
                cluster_uvh = uvh[cl["points_mask"]]

                g = part.grids(cluster_uvh)
                ii, jj = np.nonzero(g["count"] > 0)
                n_occ = len(ii)
                if n_occ == 0:
                    continue
                occupied_cells_list.append(n_occ)
                u_span = (jj.max()-jj.min()+1) if n_occ else 0
                v_span = (ii.max()-ii.min()+1) if n_occ else 0
                points_per_cell = float(len(cluster_uvh)) / n_occ

                # official decision -- reused verbatim
                res = sweep_mod.degraded_classify(cluster_uvh, part)
                kind = res.get("kind","unknown"); detail = res.get("detail","")
                n_total += 1
                correct = (kind == truth)
                n_correct += int(correct)

                # numeric residuals -- SAME fits, SAME threshold, direct call
                uc, vc = part.center_of(ii, jj)
                cell_pts = np.column_stack([uc, vc, g["mean_h"][ii, jj]])
                rmse_s = rmse_b = np.inf
                r_s = None
                try:
                    _, r_s, _, rmse_s = ransac_sphere(cell_pts, thresh_mm=thresh_mm)
                except (ValueError, np.linalg.LinAlgError):
                    pass
                try:
                    _, rmse_b, _ = planes_fit(cell_pts, thresh_mm=thresh_mm)
                except (ValueError, np.linalg.LinAlgError):
                    pass
                if np.isfinite(rmse_s): sphere_rmses.append(rmse_s)
                if np.isfinite(rmse_b): box_rmses.append(rmse_b)
                if r_s is not None: sizes.append(r_s)

                if np.isfinite(rmse_s) and np.isfinite(rmse_b):
                    winner, loser = min(rmse_s,rmse_b), max(rmse_s,rmse_b)
                    margin_ratio = loser/winner if winner>0 else np.inf
                    margins.append(margin_ratio)
                    margin_rows.append([rec, h, fi, kind, correct, margin_ratio, config.CLASS_MARGIN])

                feature_by_h[rec][h].append(dict(size=r_s, sphere_rmse=rmse_s if np.isfinite(rmse_s) else None,
                                                box_rmse=rmse_b if np.isfinite(rmse_b) else None,
                                                n_occ=n_occ, correct=correct, kind=kind))

                if h == reference_h and r_s is not None:
                    ref_values[fi] = r_s
                if h != reference_h and fi in ref_values and r_s is not None:
                    delta_rows.append([rec, h, fi, r_s - ref_values[fi]])

                if not correct:
                    reason = ("insufficient_cells" if n_occ < sweep_mod.MIN_CELL_POINTS else
                             "ambiguous_margin" if "margin too small" in detail else
                             "wrong_decision" if kind in ("ball","box") else
                             "fit_failed" if "both fits failed" in detail else "other")
                    failure_rows.append([rec, h, fi, truth, kind, n_occ,
                                        f"{rmse_s:.3f}" if np.isfinite(rmse_s) else "inf",
                                        f"{rmse_b:.3f}" if np.isfinite(rmse_b) else "inf",
                                        f"{margins[-1]:.3f}" if margins and len(margins)==n_total else "",
                                        reason])

            sp = stats_block(occupied_cells_list)
            ss = stats_block(sphere_rmses); sb = stats_block(box_rmses)
            sm = stats_block(margins); ssz = stats_block(sizes)
            id_rate_local = n_correct/max(n_total,1)
            print(f"  h={h}: n={n_total} id_rate={id_rate_local:.3f} "
                 f"occupied_cells[median={sp['median']:.1f}] "
                 f"sphere_rmse[median={ss['median']:.3f}] box_rmse[median={sb['median']:.3f}] "
                 f"margin_ratio[median={sm['median']:.3f}] CLASS_MARGIN={config.CLASS_MARGIN}")
            spatial_rows.append([rec, h, sp["n"], sp["mean"], sp["median"], sp["std"], sp["p5"], sp["p95"],
                                u_span, v_span, points_per_cell])
            sphere_rows.append([rec, h, ss["n"], ss["mean"], ss["median"], ss["p95"], ssz["median"], ssz["std"]])
            box_rows.append([rec, h, sb["n"], sb["mean"], sb["median"], sb["p95"]])

    save_csv(os.path.join(out_dir, "spatial_information_by_h.csv"), spatial_rows,
            ["recording","h_mm","n","mean_occupied_cells","median_occupied_cells",
             "std_occupied_cells","p5_occupied_cells","p95_occupied_cells",
             "last_frame_u_span_cells","last_frame_v_span_cells","points_per_cell_last_frame"])
    save_csv(os.path.join(out_dir, "sphere_geometry_by_h.csv"), sphere_rows,
            ["recording","h_mm","n","mean_sphere_rmse","median_sphere_rmse","p95_sphere_rmse",
             "median_radius_mm","std_radius_mm"])
    save_csv(os.path.join(out_dir, "box_geometry_by_h.csv"), box_rows,
            ["recording","h_mm","n","mean_box_rmse","median_box_rmse","p95_box_rmse"])
    save_csv(os.path.join(out_dir, "decision_margin_by_h.csv"), margin_rows,
            ["recording","h_mm","frame_index","predicted_kind","correct","margin_ratio","class_margin_threshold"])
    save_csv(os.path.join(out_dir, "geometry_failure_cases.csv"), failure_rows,
            ["recording","h_mm","frame_index","true_object","predicted_kind","occupied_cells",
             "sphere_rmse","box_rmse","margin_ratio","failure_reason"])
    save_csv(os.path.join(out_dir, "resolution_error_propagation.csv"), delta_rows,
            ["recording","h_mm","frame_index","delta_radius_vs_h10_mm"])

    return feature_by_h


# ===========================================================================
# SECTION 8: feature separation ball vs box, per h
# ===========================================================================
def feature_separation(feature_by_h, out_dir):
    heading("## GEOMETRY FEATURE SEPARATION: ball vs box, per h")
    rows = []
    for h in H_TEST:
        ball_feats = feature_by_h.get("ball_static", {}).get(h, [])
        box_feats = feature_by_h.get("box_static", {}).get(h, [])
        for feat_name in ["sphere_rmse", "box_rmse", "n_occ"]:
            ball_vals = [f[feat_name] for f in ball_feats if f[feat_name] is not None]
            box_vals = [f[feat_name] for f in box_feats if f[feat_name] is not None]
            sb, sx = stats_block(ball_vals), stats_block(box_vals)
            overlap = "N/A"
            if sb["n"] and sx["n"]:
                lo, hi = max(sb["p5"],sx["p5"]), min(sb["p95"],sx["p95"])
                overlap = "YES (P5-P95 ranges overlap)" if hi>lo else "NO (ranges separated)"
            print(f"  h={h} {feat_name}: ball[median={sb['median']:.3f}] "
                 f"box[median={sx['median']:.3f}]  overlap={overlap}")
            rows.append([h, feat_name, sb["n"], sb["median"], sb["p5"], sb["p95"],
                        sx["n"], sx["median"], sx["p5"], sx["p95"], overlap])
    save_csv(os.path.join(out_dir, "geometry_feature_separation.csv"), rows,
            ["h_mm","feature","ball_n","ball_median","ball_p5","ball_p95",
             "box_n","box_median","box_p5","box_p95","p5_p95_overlap"])


# ===========================================================================
# SECTION 12/13: threshold audit + sensitivity (no tuning, no optimization)
# ===========================================================================
def threshold_audit(out_dir):
    heading("## THRESHOLD AUDIT (Section 12) -- no thresholds changed")
    rows = [
        ["SPHERE_THRESH_MM (config)", config.SPHERE_THRESH_MM, "physical mm", "fixed, NOT h-dependent"],
        ["degraded_classify thresh_mm", "max(SPHERE_THRESH_MM, h/2.0)", "physical mm",
        "DELIBERATELY h-dependent by design (sweep.py docstring: residuals legitimately grow with h)"],
        ["MIN_CELL_POINTS (sweep.py local)", sweep_mod.MIN_CELL_POINTS, "cell count", "fixed cell-count threshold, NOT physical-size dependent"],
        ["config.MIN_CLUSTER_POINTS (different constant, full-res path)", config.MIN_CLUSTER_POINTS,
        "point count", "NOT used by degraded_classify -- a DIFFERENT, larger threshold used elsewhere in the full-resolution pipeline; confirm you are not conflating the two"],
        ["CLASS_MARGIN (config)", config.CLASS_MARGIN, "unitless ratio", "fixed, NOT h-dependent"],
    ]
    for r in rows:
        print(f"  {r[0]}: {r[1]}  [{r[2]}] -- {r[3]}")
    save_csv(os.path.join(out_dir, "threshold_audit.csv"), rows,
            ["threshold","value","units","h_dependence_note"])
    note("**One threshold is genuinely and deliberately h-dependent**: the "
        "RANSAC inlier band scales as max(4mm, h/2) in the degraded "
        "(cell-level) path. This is stated in sweep.py's own docstring as an "
        "intentional accommodation for growing quantization residuals at "
        "coarse h, not a bug. It is flagged here per your audit instruction, "
        "not silently accepted. `MIN_CELL_POINTS=4` (local to sweep.py) is "
        "a FIXED cell-count floor, distinct from `config.MIN_CLUSTER_POINTS=40` "
        "used by the full-resolution (non-degraded) pipeline -- confirmed "
        "these are two different constants for two different code paths, "
        "not the same threshold reused inconsistently.")


def threshold_sensitivity(margin_rows_path, out_dir):
    heading("## THRESHOLD SENSITIVITY DIAGNOSTIC (Section 13) -- diagnostic only, NOT tuning")
    if not os.path.exists(margin_rows_path):
        print("  margin data not available -- skipping."); return
    rows = list(csv.DictReader(open(margin_rows_path)))
    out_rows = []
    for h in H_TEST:
        hrows = [r for r in rows if float(r["h_mm"])==h]
        ratios = np.array([float(r["margin_ratio"]) for r in hrows if r["margin_ratio"]])
        cm = config.CLASS_MARGIN
        close_frac = float(np.mean(np.abs(ratios - cm) < 0.1*cm)) if len(ratios) else float("nan")
        print(f"  h={h}: {100*close_frac:.1f}% of samples have a margin ratio within "
             f"10% of the CLASS_MARGIN={cm} decision threshold "
             f"(higher = more brittle/sensitive to the exact threshold value)")
        out_rows.append([h, len(ratios), close_frac])
    save_csv(os.path.join(out_dir, "threshold_sensitivity.csv"), out_rows,
            ["h_mm","n_samples","fraction_within_10pct_of_class_margin"])
    note("This reports how many real samples sit close to the existing "
        "CLASS_MARGIN decision boundary at each h -- a HIGH fraction means "
        "the classification at that h is inherently brittle regardless of "
        "the exact threshold chosen. No new threshold was searched for or "
        "reported as improving accuracy.")


# ===========================================================================
# SECTION 17: geometry vs learned comparison (read-only, already-validated)
# ===========================================================================
def geometry_vs_learned(geo_rows, out_dir):
    heading("## GEOMETRY vs LEARNED CLASSIFIER (using ALREADY-VALIDATED learned results)")
    learned_path = os.path.join(_PROJECT_ROOT, "results", "phase2_results.csv")
    learned = {}
    if os.path.exists(learned_path):
        for r in csv.DictReader(open(learned_path)):
            if r.get("variant")=="full":
                learned[float(r["h_mm"])] = float(r["accuracy"]) if "accuracy" in r else None
        print(f"  loaded learned classifier accuracy from {learned_path} (NOT recomputed here)")
    else:
        print(f"  NOTE: {learned_path} not found -- geometry-only numbers reported, "
             f"no comparison possible without recomputing (not done here).")

    geo_by_h = {}
    for r in geo_rows:
        rec, h, id_rate = r[0], r[1], r[5]
        geo_by_h.setdefault(h, []).append(id_rate)
    rows = []
    for h in H_TEST:
        geo_avg = float(np.mean(geo_by_h.get(h, [np.nan])))
        learned_acc = learned.get(h)
        print(f"  h={h}: geometry_avg_id_rate={geo_avg:.4f}  "
             f"learned_accuracy={'N/A' if learned_acc is None else f'{learned_acc:.4f}'}")
        rows.append([h, geo_avg, learned_acc if learned_acc is not None else ""])
    save_csv(os.path.join(out_dir, "geometry_vs_learned_comparison.csv"), rows,
            ["h_mm","geometry_avg_id_rate","learned_classifier_accuracy"])
    note("**CAUTION, stated explicitly:** geometry numbers here come from "
        "`ball_static`/`box_static` (single-object recordings). The learned "
        "classifier's saved accuracy may come from a different evaluation "
        "set (check `results/phase2_results.csv`'s own methodology). **This "
        "comparison is CONTEXTUAL, not necessarily sample-matched or "
        "apples-to-apples** -- do not present it as a controlled comparison "
        "without verifying the underlying test populations match.")


# ===========================================================================
def main():
    p = argparse.ArgumentParser(
        description="Real-data investigation of geometry-based identification "
                    "vs cell size h. Reuses sweep.py's production functions. "
                    "No retraining, no synthetic data, no threshold tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--out", type=str, default="validation_results/geometry_resolution")
    p.add_argument("--every", type=int, default=20, help="frame sampling stride (matches sweep.py's own default)")
    args = p.parse_args()
    def _resolve(path): return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)
    args.results_dir = _resolve(args.results_dir); args.out = _resolve(args.out)
    plots_dir = os.path.join(args.out, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.isdir(os.path.join(_PROJECT_ROOT, "Data", "extracted")):
        print("NOTE: Data/extracted/ not found in this environment. This script "
             "requires real raw depth frames (ball_static, box_static) to run. "
             "Nothing computed.")
        return

    geo_rows = verify_official_results(args.results_dir, args.out, args.every)
    feature_by_h = detailed_per_frame_analysis(args.results_dir, args.out, args.every)
    feature_separation(feature_by_h, args.out)
    threshold_audit(args.out)
    threshold_sensitivity(os.path.join(args.out, "decision_margin_by_h.csv"), args.out)
    geometry_vs_learned(geo_rows, args.out)

    # ---- Section 18: identification vs tracking distinction ----
    heading("## IDENTIFICATION vs TRACKING -- explicit distinction")
    note("**This script investigates GEOMETRY-BASED SHAPE IDENTIFICATION "
        "under spatial coarsening only.** It does NOT measure temporal "
        "tracking robustness vs h -- that is a separate question, not "
        "addressed here, and failure of shape identification at a given h "
        "does not by itself imply tracking (position estimation across "
        "frames) also fails at that h.")

    # ---- plots ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    hs = H_TEST
    for rec in RECORDINGS:
        rrows = [r for r in geo_rows if r[0]==rec]
        if rrows:
            fig, ax = plt.subplots(figsize=(7,4.5))
            ax.plot([r[1] for r in rrows], [r[5] for r in rrows], "o-")
            ax.set_xlabel("h (mm)"); ax.set_ylabel("identification rate"); ax.set_ylim(-0.05,1.05)
            ax.set_title(f"{rec}: identification accuracy vs h")
            fig.savefig(os.path.join(plots_dir, f"accuracy_vs_h_{rec}.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

    report_path = os.path.join(args.out, "geometry_resolution_validation_report.md")
    lines = ["# GEOMETRY RESOLUTION VALIDATION REPORT\n"]
    lines.extend(x if x.startswith("#") else x+"\n" for x in REPORT)
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"\n\nFull report: {report_path}")


if __name__ == "__main__":
    main()