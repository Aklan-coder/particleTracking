"""
sweep_geometry_transition.py — locates the geometry-identification failure
transition between 20mm and 30mm at fine h steps, using ONLY real
ball_static/box_static data and the EXISTING production geometry pipeline.
No thresholds changed, no retraining, no synthetic data. Modifies no file.

PRODUCTION FUNCTIONS REUSED, NOT REIMPLEMENTED:
  sweep.run_one()             -- called verbatim per h, for the OFFICIAL
                                  phase-averaged accuracy numbers (exact
                                  same statistic as sweep.py's own output).
  sweep._phase_partitions()   -- the exact 9-phase grid-alignment set,
                                  reused directly for the detailed
                                  per-trial instrumentation loop below
                                  (run_one() itself does not expose
                                  per-trial spatial/fit detail, only
                                  aggregated rates, so this script adds an
                                  INSTRUMENTED outer loop using the SAME
                                  phases/frames/threshold -- not a
                                  different experiment, the same one with
                                  more logging).
  sweep.degraded_classify()   -- the exact classification decision,
                                  called identically in both the official
                                  and the instrumented loop.
  sweep.MIN_CELL_POINTS, sweep.TRUTH -- unchanged.
  geometry.ransac_sphere(), geometry.planes_fit() -- called directly with
                                  the SAME threshold formula degraded_classify()
                                  uses (max(SPHERE_THRESH_MM, h/2.0)), only
                                  to expose numeric residuals for the
                                  detailed CSVs (classify_cluster() itself
                                  only returns them embedded in a string).
  fit_static.object_points(), load_table(), geometry.load_masked(),
  discretize.Partition -- unchanged.

Does NOT run the learned classifier anywhere in this file (confirmed by
the absence of any import from train_model.py / eval_model.py / any
model_h*.npz file).

Usage:
  python sweep_geometry_transition.py
  python sweep_geometry_transition.py --results-dir results --out validation_results/geometry_transition --every 20
  python sweep_geometry_transition.py --help
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
from fit_static import load_table, object_points
from geometry import load_masked, ransac_sphere, planes_fit
import sweep as sweep_mod

H_TRANSITION = [20.0, 22.5, 25.0, 27.5, 30.0]
RECORDINGS = ["ball_static", "box_static"]


def pctl(a,p): return float(np.percentile(a,p)) if len(a) else float("nan")
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
# OFFICIAL accuracy numbers -- sweep.run_one() reused verbatim
# ===========================================================================
def official_accuracy(results_dir, out_dir, every):
    heading("## OFFICIAL PHASE-AVERAGED ACCURACY (sweep.run_one(), verbatim, 9 phases per h)")
    bg = np.load(os.path.join(results_dir, "background_median.npy"))
    tf, z = load_table()
    rows, files_by_rec = [], {}
    for rec in RECORDINGS:
        files = sorted(glob.glob(os.path.join(_PROJECT_ROOT, "Data", "extracted", rec, "depth", "*.npy")))[::every]
        files_by_rec[rec] = files
        if not files:
            print(f"  {rec}: no raw frames found -- skipping."); continue
        print(f"\n{rec}: {len(files)} frames x {sweep_mod.PHASES**2} phases (identical frame list across every h)")
        for h in H_TRANSITION:
            r = sweep_mod.run_one(rec, files, bg, tf, z, h)
            print(f"  h={h}: trials={r['trials']} detected={r['detected']} correct={r['correct']} "
                 f"id_rate={r['id_rate']:.4f} det_rate={r['det_rate']:.4f} cells_across={r['cells_across']:.1f}")
            rows.append([rec, h, r["trials"], r["detected"], r["correct"], r["id_rate"],
                        r["det_rate"], r["cells_across"]])
    save_csv(os.path.join(out_dir, "transition_accuracy.csv"), rows,
            ["recording","h_mm","n_trials","n_detected","n_correct","id_rate","det_rate","median_cells_across"])
    return rows, files_by_rec


# ===========================================================================
# Instrumented per-trial loop: same phases/frames/threshold, extra detail
# ===========================================================================
def instrumented_detail(results_dir, out_dir, files_by_rec):
    heading("## INSTRUMENTED PER-TRIAL DETAIL (same phases/frames/threshold as sweep.run_one())")
    bg = np.load(os.path.join(results_dir, "background_median.npy"))
    tf, z = load_table()

    by_object_rows, spatial_rows, fit_rows, failure_rows = [], [], [], []
    cellbin_records = []   # (n_occ, correct) across ALL h/objects, for the empirical binning

    for rec in RECORDINGS:
        files = files_by_rec.get(rec, [])
        if not files: continue
        truth = sweep_mod.TRUTH[rec]

        for h in H_TRANSITION:
            parts = sweep_mod._phase_partitions(z, h)   # SAME 9 phases as run_one()
            thresh_mm = max(config.SPHERE_THRESH_MM, h/2.0)
            n_trials = n_detected = n_correct = 0
            cells_list, margins, sphere_rmses, box_rmses = [], [], [], []
            n_sphere_valid = n_box_valid = 0

            for f in files:
                pts = object_points(load_masked(f), bg)
                if len(pts) < 10:
                    n_trials += len(parts)
                    continue
                uvh = tf.to_table(pts)
                for part in parts:
                    n_trials += 1
                    clusters, _ = part.clusters(uvh)
                    if not clusters:
                        continue
                    cl = clusters[0]
                    n_detected += 1
                    cluster_uvh = uvh[cl["points_mask"]]

                    res = sweep_mod.degraded_classify(cluster_uvh, part)
                    kind = res.get("kind","unknown")
                    correct = (kind == truth)
                    n_correct += int(correct)

                    g = part.grids(cluster_uvh)
                    ii, jj = np.nonzero(g["count"]>0)
                    n_occ = len(ii)
                    cells_list.append(n_occ)
                    cellbin_records.append((n_occ, correct))

                    rmse_s = rmse_b = np.inf
                    if n_occ > 0:
                        uc, vc = part.center_of(ii, jj)
                        cell_pts = np.column_stack([uc, vc, g["mean_h"][ii,jj]])
                        try:
                            _,_,_,rmse_s = ransac_sphere(cell_pts, thresh_mm=thresh_mm)
                            n_sphere_valid += 1
                        except (ValueError, np.linalg.LinAlgError): pass
                        try:
                            _,rmse_b,_ = planes_fit(cell_pts, thresh_mm=thresh_mm)
                            n_box_valid += 1
                        except (ValueError, np.linalg.LinAlgError): pass
                    if np.isfinite(rmse_s): sphere_rmses.append(rmse_s)
                    if np.isfinite(rmse_b): box_rmses.append(rmse_b)
                    if np.isfinite(rmse_s) and np.isfinite(rmse_b):
                        winner, loser = min(rmse_s,rmse_b), max(rmse_s,rmse_b)
                        margins.append(loser/winner if winner>0 else np.inf)

                    if not correct:
                        reason = ("insufficient_cells" if n_occ < sweep_mod.MIN_CELL_POINTS else
                                 "fit_failed" if not (np.isfinite(rmse_s) or np.isfinite(rmse_b)) else
                                 "ambiguous_margin" if kind=="unknown" else
                                 "wrong_decision")
                        failure_rows.append([rec, h, truth, kind, n_occ,
                                            f"{rmse_s:.3f}" if np.isfinite(rmse_s) else "inf",
                                            f"{rmse_b:.3f}" if np.isfinite(rmse_b) else "inf", reason])

            cells_arr = np.array(cells_list)
            margin_arr = np.array(margins)
            id_rate = n_correct/max(n_detected,1)
            sphere_valid_pct = 100*n_sphere_valid/max(n_detected,1)
            box_valid_pct = 100*n_box_valid/max(n_detected,1)
            valid_decision_pct = 100*sum(1 for i in range(len(cells_list)) if True)/max(n_detected,1)  # detected==attempted here

            print(f"\n{rec} h={h}: id_rate={id_rate:.4f}  "
                 f"cells[median={np.median(cells_arr) if len(cells_arr) else float('nan'):.1f} "
                 f"P25={pctl(cells_arr,25):.1f} P75={pctl(cells_arr,75):.1f}]  "
                 f"sphere_valid={sphere_valid_pct:.1f}%  box_valid={box_valid_pct:.1f}%  "
                 f"margin_median={np.median(margin_arr) if len(margin_arr) else float('nan'):.3f}")

            by_object_rows.append([rec, h, n_trials, n_detected, n_correct, id_rate,
                                  np.median(cells_arr) if len(cells_arr) else np.nan,
                                  pctl(cells_arr,25), pctl(cells_arr,75),
                                  sphere_valid_pct, np.median(sphere_rmses) if sphere_rmses else np.nan,
                                  box_valid_pct, np.median(box_rmses) if box_rmses else np.nan,
                                  np.median(margin_arr) if len(margin_arr) else np.nan])
            spatial_rows.append([rec, h, np.median(cells_arr) if len(cells_arr) else np.nan,
                                pctl(cells_arr,25), pctl(cells_arr,75), pctl(cells_arr,5), pctl(cells_arr,95)])
            fit_rows.append([rec, h, sphere_valid_pct, box_valid_pct,
                            np.median(sphere_rmses) if sphere_rmses else np.nan,
                            np.median(box_rmses) if box_rmses else np.nan])

    save_csv(os.path.join(out_dir, "transition_by_object.csv"), by_object_rows,
            ["recording","h_mm","n_trials","n_detected","n_correct","id_rate",
             "median_cells","p25_cells","p75_cells","sphere_valid_pct","median_sphere_rmse",
             "box_valid_pct","median_box_rmse","median_decision_margin"])
    save_csv(os.path.join(out_dir, "transition_spatial_support.csv"), spatial_rows,
            ["recording","h_mm","median_cells","p25_cells","p75_cells","p5_cells","p95_cells"])
    save_csv(os.path.join(out_dir, "transition_fit_validity.csv"), fit_rows,
            ["recording","h_mm","sphere_valid_pct","box_valid_pct","median_sphere_rmse","median_box_rmse"])
    save_csv(os.path.join(out_dir, "transition_failure_cases.csv"), failure_rows,
            ["recording","h_mm","truth","predicted_kind","occupied_cells","sphere_rmse","box_rmse","failure_reason"])
    return by_object_rows, cellbin_records


# ===========================================================================
# accuracy vs occupied-cell count, data-derived bins
# ===========================================================================
def accuracy_by_cells(cellbin_records, out_dir):
    heading("## IDENTIFICATION ACCURACY vs OCCUPIED-CELL COUNT (data-derived bins)")
    if not cellbin_records:
        print("  no records collected."); return
    cells = np.array([c for c,_ in cellbin_records])
    correct = np.array([c for _,c in cellbin_records])
    # data-derived bin edges: deciles of the observed cell-count distribution
    edges = np.unique(np.percentile(cells, np.linspace(0,100,11)))
    rows = []
    for i in range(len(edges)-1):
        lo, hi = edges[i], edges[i+1]
        sel = (cells>=lo) & (cells<=hi if i==len(edges)-2 else cells<hi)
        n = int(sel.sum())
        acc = float(correct[sel].mean()) if n else float("nan")
        print(f"  cells in [{lo:.1f},{hi:.1f}): n={n} accuracy={acc:.4f}")
        rows.append([lo, hi, n, acc])
    save_csv(os.path.join(out_dir, "accuracy_by_occupied_cells.csv"), rows,
            ["cell_count_bin_lo","cell_count_bin_hi","n","identification_accuracy"])
    note("Bins are DATA-DERIVED (deciles of the observed occupied-cell "
        "distribution across all matched trials), not a predetermined "
        "minimum-cell threshold chosen in advance. **Cell count alone is "
        "not claimed as causal** -- see transition_fit_validity.csv "
        "(sphere/box fit success rate) alongside this table before "
        "concluding cell count is the sole driver; fit validity and cell "
        "arrangement matter too, per your instruction.")


def main():
    p = argparse.ArgumentParser(
        description="Locate the geometry-identification failure transition "
                    "between 20-30mm using real data and the existing "
                    "production sweep.py pipeline. No learned classifier, "
                    "no retraining, no threshold changes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--out", type=str, default="validation_results/geometry_transition")
    p.add_argument("--every", type=int, default=20)
    args = p.parse_args()
    def _resolve(path): return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)
    args.results_dir = _resolve(args.results_dir); args.out = _resolve(args.out)
    plots_dir = os.path.join(args.out, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.isdir(os.path.join(_PROJECT_ROOT, "Data", "extracted")):
        print("NOTE: Data/extracted/ not found in this environment. Nothing computed.")
        return

    official_rows, files_by_rec = official_accuracy(args.results_dir, args.out, args.every)
    by_object_rows, cellbin_records = instrumented_detail(args.results_dir, args.out, files_by_rec)
    accuracy_by_cells(cellbin_records, args.out)

    # ---- transition table + step-to-step deltas ----
    heading("## TRANSITION TABLE + STEP-TO-STEP CHANGES")
    trans_rows = []
    by_h = {}
    for h in H_TRANSITION:
        ball = [r for r in by_object_rows if r[0]=="ball_static" and r[1]==h]
        box = [r for r in by_object_rows if r[0]=="box_static" and r[1]==h]
        if not ball or not box: continue
        ball_acc, box_acc = ball[0][5], box[0][5]
        balanced = (ball_acc+box_acc)/2
        by_h[h] = dict(ball_acc=ball_acc, box_acc=box_acc, balanced=balanced,
                       ball_cells=ball[0][6], box_cells=box[0][6],
                       sphere_valid=ball[0][9], box_valid=box[0][11])
        print(f"h={h}: ball_acc={ball_acc:.4f} box_acc={box_acc:.4f} balanced={balanced:.4f}")
        trans_rows.append([h, ball_acc, box_acc, balanced, ball[0][6], box[0][6],
                          ball[0][9], box[0][11]])
    save_csv(os.path.join(args.out, "transition_table.csv"), trans_rows,
            ["h_mm","ball_accuracy","box_accuracy","balanced_accuracy",
             "ball_median_cells","box_median_cells","sphere_fit_valid_pct","box_fit_valid_pct"])

    steps = list(zip(H_TRANSITION[:-1], H_TRANSITION[1:]))
    largest_drop = (None, -1)
    for h0, h1 in steps:
        if h0 not in by_h or h1 not in by_h: continue
        d_bal = by_h[h0]["balanced"] - by_h[h1]["balanced"]
        d_cells_ball = by_h[h0]["ball_cells"] - by_h[h1]["ball_cells"]
        d_cells_box = by_h[h0]["box_cells"] - by_h[h1]["box_cells"]
        d_sphere_valid = by_h[h0]["sphere_valid"] - by_h[h1]["sphere_valid"]
        print(f"  {h0}->{h1}: balanced_acc_drop={d_bal:+.4f}  "
             f"ball_cells_drop={d_cells_ball:+.2f}  box_cells_drop={d_cells_box:+.2f}  "
             f"sphere_valid_pct_drop={d_sphere_valid:+.2f}")
        if d_bal > largest_drop[1]:
            largest_drop = (f"{h0}->{h1}", d_bal)
    if largest_drop[0]:
        print(f"\nLargest balanced-accuracy drop: {largest_drop[0]} ({largest_drop[1]:+.4f})")
        note(f"**Largest single-step balanced-accuracy drop occurs at "
            f"{largest_drop[0]}mm ({largest_drop[1]:+.4f}).** See "
            f"transition_by_object.csv and transition_fit_validity.csv for "
            f"which geometric quantity (cell count, sphere/box fit validity, "
            f"decision margin) changes most sharply at that same step -- "
            f"reported as observed co-occurring changes, not asserted as "
            f"the single cause.")

    # ---- plots ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    hs = sorted(by_h.keys())
    if hs:
        fig, ax = plt.subplots(figsize=(7,4.5))
        ax.plot(hs, [by_h[h]["ball_acc"] for h in hs], "o-", label="ball")
        ax.plot(hs, [by_h[h]["box_acc"] for h in hs], "s-", label="box")
        ax.plot(hs, [by_h[h]["balanced"] for h in hs], "^--", label="balanced")
        ax.set_xlabel("h (mm)"); ax.set_ylabel("identification accuracy"); ax.legend(); ax.set_ylim(-0.05,1.05)
        ax.set_title("Geometry accuracy vs h, 20-30mm transition")
        fig.savefig(os.path.join(plots_dir, "accuracy_vs_h.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7,4.5))
        ax.plot(hs, [by_h[h]["ball_cells"] for h in hs], "o-", label="ball")
        ax.plot(hs, [by_h[h]["box_cells"] for h in hs], "s-", label="box")
        ax.set_xlabel("h (mm)"); ax.set_ylabel("median occupied cells"); ax.legend()
        ax.set_title("Occupied cells vs h")
        fig.savefig(os.path.join(plots_dir, "cells_vs_h.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7,4.5))
        ax.plot(hs, [by_h[h]["sphere_valid"] for h in hs], "o-", label="sphere fit valid %")
        ax.plot(hs, [by_h[h]["box_valid"] for h in hs], "s-", label="box fit valid %")
        ax.set_xlabel("h (mm)"); ax.set_ylabel("%"); ax.legend()
        ax.set_title("Fit validity vs h")
        fig.savefig(os.path.join(plots_dir, "fit_validity_vs_h.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    if cellbin_records:
        cells = np.array([c for c,_ in cellbin_records]); correct = np.array([c for _,c in cellbin_records])
        edges = np.unique(np.percentile(cells, np.linspace(0,100,11)))
        centers, accs = [], []
        for i in range(len(edges)-1):
            lo, hi = edges[i], edges[i+1]
            sel = (cells>=lo) & (cells<=hi if i==len(edges)-2 else cells<hi)
            if sel.sum():
                centers.append((lo+hi)/2); accs.append(correct[sel].mean())
        fig, ax = plt.subplots(figsize=(7,4.5))
        ax.plot(centers, accs, "o-")
        ax.set_xlabel("occupied cells (bin center)"); ax.set_ylabel("identification accuracy")
        ax.set_title("Accuracy vs occupied-cell count (data-derived bins)")
        fig.savefig(os.path.join(plots_dir, "accuracy_vs_cells.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---- report ----
    report_path = os.path.join(args.out, "geometry_transition_report.md")
    lines = ["# GEOMETRY TRANSITION REPORT (20-30mm, real data, unmodified production pipeline)\n"]
    lines.extend(x if x.startswith("#") else x+"\n" for x in REPORT)
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"\n\nFull report: {report_path}")


if __name__ == "__main__":
    main()