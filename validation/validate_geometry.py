"""
01_validate_geometry.py — real-data geometry validation tool.

Evaluates the EXISTING geometry implementation (geometry.py, fit_static.py)
against real recordings you point it at. Does not retrain anything, does
not alter your data, does not replace or tune any fitting algorithm.
Measures behavior only.

Imports, reused UNMODIFIED from your repository:
  geometry.py    : load_masked, backproject, ransac_sphere, planes_fit
  fit_static.py  : load_table, object_points

ONE exception, documented per your instruction #1 ("if an important
algorithm is embedded inside another script and cannot be imported, make
the smallest possible refactoring necessary and document it"):

  box_dimensions_from_points() below is EXTRACTED VERBATIM from the
  inline block inside fit_static.py's fit_box_frames() function (PCA on
  the table-frame footprint for length/width, 99th-percentile height).
  That logic was not previously exposed as an importable function -- it
  was written directly inside the per-recording loop. This is the exact
  same math your research results were generated with, character for
  character; nothing about the computation was changed. If you want a
  single source of truth going forward, the smallest real fix is to move
  this function into fit_static.py itself and have fit_box_frames() call
  it -- I have NOT made that edit to your repository; this script simply
  duplicates the formula and says so, so you can decide.

Usage:
  python 01_validate_geometry.py --reference PATH --ball PATH --box PATH

  With physical ground truth (optional -- omit to skip physical-accuracy
  numbers, fit-residual numbers are always computed regardless):
  python 01_validate_geometry.py --reference PATH --ball PATH --box PATH \\
      --ball-diameter-mm 60.66 \\
      --box-length-mm 133.1 --box-width-mm 57.3 --box-height-mm 35.3

  Other options:
    --results-dir results     (where background_median.npy / table_plane.npz live)
    --out validation_results  (where this script writes its output)
    --every 1                 (process every Nth frame; 1 = every frame)

  python 01_validate_geometry.py --help
"""
import argparse
import csv
import glob
import os
import sys
import traceback

# This script can live anywhere in the project tree (e.g.
# results/validation/). Make sure the project ROOT -- where config.py,
# geometry.py, fit_static.py actually live -- is importable regardless of
# where this file itself sits or where it's invoked from.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = _THIS_DIR
for _ in range(4):  # walk up until we find config.py, or give up after 4 levels
    if os.path.exists(os.path.join(_PROJECT_ROOT, "config.py")):
        break
    _PROJECT_ROOT = os.path.dirname(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

import config
from geometry import load_masked, backproject, ransac_sphere, planes_fit
from fit_static import load_table, object_points


# ===========================================================================
# Extracted-verbatim helper (see module docstring)
# ===========================================================================
def box_dimensions_from_points(pts, tf):
    """Oriented length/width (PCA on table-frame footprint) + height
    (99th-percentile above table plane). Copied verbatim from the inline
    block inside fit_static.py's fit_box_frames(); NOT a new algorithm."""
    uvh = tf.to_table(pts)
    uv = uvh[:, :2]
    c2 = uv.mean(axis=0)
    A = uv - c2
    cov = A.T @ A / len(A)
    w, V = np.linalg.eigh(cov)
    proj = A @ V
    lo, hi = np.percentile(proj[:, 1], [1, 99])
    length = float(hi - lo)
    lo, hi = np.percentile(proj[:, 0], [1, 99])
    width = float(hi - lo)
    height = float(np.percentile(uvh[:, 2], 99))
    return length, width, height, c2


# ===========================================================================
# Small utilities
# ===========================================================================
def pctl(arr, p):
    return float(np.percentile(arr, p)) if len(arr) else float("nan")


def stats_block(arr):
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return dict(n=0, mean=float("nan"), median=float("nan"), rmse=float("nan"),
                    std=float("nan"), p90=float("nan"), p95=float("nan"), max=float("nan"))
    return dict(n=int(len(arr)), mean=float(arr.mean()), median=float(np.median(arr)),
               rmse=float(np.sqrt((arr**2).mean())), std=float(arr.std()),
               p90=pctl(arr,90), p95=pctl(arr,95), max=float(arr.max()))


def save_csv(path, rows, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


def list_depth_files(folder):
    """Matches this repo's existing convention: <folder>/depth/*.npy,
    or <folder>/*.npy directly if no 'depth' subfolder exists."""
    if os.path.isdir(os.path.join(folder, "depth")):
        folder = os.path.join(folder, "depth")
    return sorted(glob.glob(os.path.join(folder, "*.npy")))


# ===========================================================================
# PLANE VALIDATION (real reference frames, existing plane from table_plane.npz)
# ===========================================================================
def validate_plane(reference_dir, plane_n, plane_d, out_dir, every=1):
    print("\n" + "="*70)
    print("PLANE VALIDATION (real reference frames)")
    print("="*70)
    if not reference_dir:
        print("  --reference not provided -- skipping plane validation.")
        return None, 0, 0, 0
    files = list_depth_files(reference_dir)[::every]
    if not files:
        print(f"  No .npy frames found under {reference_dir} -- skipping.")
        return None, 0, 0, 0

    rows, fail_rows, all_resid = [], [], []
    for fi, f in enumerate(files):
        try:
            depth = load_masked(f)
            pts, _ = backproject(depth)
            if len(pts) < 10:
                fail_rows.append([f, "plane", "too few valid points", len(pts)])
                continue
            resid = pts @ plane_n + plane_d
            abs_resid = np.abs(resid)
            inl = abs_resid < config.PLANE_THRESH_MM
            s = stats_block(abs_resid)
            rows.append([os.path.basename(f), len(pts), int(inl.sum()),
                        f"{100*inl.mean():.2f}", f"{s['mean']:.4f}", f"{s['median']:.4f}",
                        f"{s['rmse']:.4f}", f"{s['std']:.4f}", f"{s['p90']:.4f}",
                        f"{s['p95']:.4f}", f"{s['max']:.4f}"])
            all_resid.append(abs_resid)
        except Exception as e:
            fail_rows.append([f, "plane", f"{type(e).__name__}: {e}", -1])
        if (fi+1) % 100 == 0:
            print(f"  ...{fi+1}/{len(files)}")

    save_csv(os.path.join(out_dir, "plane_per_frame.csv"), rows,
             ["frame","n_points","n_inliers","inlier_pct","mean_abs_resid_mm",
              "median_resid_mm","rmse_mm","std_mm","p90_mm","p95_mm","max_mm"])

    n_attempt, n_ok, n_fail = len(files), len(rows), len(fail_rows)
    if all_resid:
        agg = np.concatenate(all_resid)
        s = stats_block(agg)
        summary = [["frames_attempted", n_attempt], ["frames_succeeded", n_ok],
                   ["frames_failed", n_fail],
                   ["failure_pct", 100*n_fail/max(n_attempt,1)],
                   ["plane_normal_x", float(plane_n[0])], ["plane_normal_y", float(plane_n[1])],
                   ["plane_normal_z", float(plane_n[2])], ["plane_d", float(plane_d)],
                   ["mean_abs_resid_mm", s["mean"]], ["median_resid_mm", s["median"]],
                   ["rmse_mm", s["rmse"]], ["std_mm", s["std"]],
                   ["p90_mm", s["p90"]], ["p95_mm", s["p95"]], ["max_mm", s["max"]]]
        save_csv(os.path.join(out_dir, "plane_summary.csv"), summary, ["stat","value"])
        print(f"  frames: attempted={n_attempt} succeeded={n_ok} failed={n_fail} "
             f"({100*n_fail/max(n_attempt,1):.1f}% failure)")
        print(f"  abs residual: mean={s['mean']:.4f}mm median={s['median']:.4f}mm "
             f"rmse={s['rmse']:.4f}mm p95={s['p95']:.4f}mm max={s['max']:.4f}mm")
        return s, n_attempt, n_ok, n_fail
    else:
        print("  No successful plane residuals computed.")
        return None, n_attempt, n_ok, n_fail


# ===========================================================================
# BALL / SPHERE VALIDATION
# ===========================================================================
def validate_ball(ball_dir, bg, tf, out_dir, every=1, true_diameter_mm=None):
    print("\n" + "="*70)
    print("BALL / SPHERE VALIDATION (real ball frames)")
    print("="*70)
    if not ball_dir:
        print("  --ball not provided -- skipping ball validation.")
        return None, 0, 0, 0, []
    files = list_depth_files(ball_dir)[::every]
    if not files:
        print(f"  No .npy frames found under {ball_dir} -- skipping.")
        return None, 0, 0, 0, []

    rows, fail_rows, all_resid, diameters = [], [], [], []
    for fi, f in enumerate(files):
        try:
            pts = object_points(load_masked(f), bg)
            if len(pts) < config.MIN_CLUSTER_POINTS:
                fail_rows.append([f, "ball", "too few object points", len(pts)])
                continue
            c, r, inl, rmse = ransac_sphere(pts)
            resid = np.abs(np.linalg.norm(pts[inl] - c, axis=1) - r)
            s = stats_block(resid)
            diam = 2 * r
            diameters.append(diam)
            all_resid.append(resid)
            row = [os.path.basename(f), len(pts), f"{c[0]:.3f}", f"{c[1]:.3f}", f"{c[2]:.3f}",
                  f"{r:.4f}", f"{diam:.4f}", int(inl.sum()), f"{s['mean']:.4f}",
                  f"{s['median']:.4f}", f"{s['rmse']:.4f}", f"{s['p90']:.4f}",
                  f"{s['p95']:.4f}", f"{s['max']:.4f}", "success"]
            rows.append(row)
        except (ValueError, np.linalg.LinAlgError) as e:
            fail_rows.append([f, "ball", f"{type(e).__name__}: {e}", -1])
        except Exception as e:
            fail_rows.append([f, "ball", f"UNEXPECTED {type(e).__name__}: {e}", -1])
        if (fi+1) % 100 == 0:
            print(f"  ...{fi+1}/{len(files)}")

    save_csv(os.path.join(out_dir, "ball_per_frame.csv"), rows,
             ["frame","n_points","center_x","center_y","center_z","radius_mm","diameter_mm",
              "n_inliers","mean_abs_resid_mm","median_resid_mm","rmse_mm","p90_mm","p95_mm",
              "max_mm","status"])

    n_attempt, n_ok, n_fail = len(files), len(rows), len(fail_rows)
    summary = [["frames_attempted", n_attempt], ["frames_succeeded", n_ok],
              ["frames_failed", n_fail], ["failure_pct", 100*n_fail/max(n_attempt,1)]]
    if all_resid:
        agg = np.concatenate(all_resid)
        s = stats_block(agg)
        diam_arr = np.array(diameters)
        summary += [["mean_abs_resid_mm", s["mean"]], ["median_resid_mm", s["median"]],
                   ["rmse_mm", s["rmse"]], ["std_mm", s["std"]], ["p90_mm", s["p90"]],
                   ["p95_mm", s["p95"]], ["max_mm", s["max"]],
                   ["mean_estimated_diameter_mm", float(diam_arr.mean())],
                   ["std_estimated_diameter_mm", float(diam_arr.std())]]
        print(f"  frames: attempted={n_attempt} succeeded={n_ok} failed={n_fail} "
             f"({100*n_fail/max(n_attempt,1):.1f}% failure)")
        print(f"  surface residual: mean={s['mean']:.4f}mm median={s['median']:.4f}mm "
             f"rmse={s['rmse']:.4f}mm p95={s['p95']:.4f}mm max={s['max']:.4f}mm")
        print(f"  estimated diameter: mean={diam_arr.mean():.3f}mm std={diam_arr.std():.4f}mm")
        if true_diameter_mm is not None:
            err = np.abs(diam_arr - true_diameter_mm)
            pct = 100 * err / true_diameter_mm
            se = stats_block(err)
            summary += [["true_diameter_mm", true_diameter_mm],
                       ["mean_abs_physical_error_mm", se["mean"]],
                       ["median_abs_physical_error_mm", se["median"]],
                       ["rmse_physical_error_mm", se["rmse"]],
                       ["p95_physical_error_mm", se["p95"]],
                       ["max_physical_error_mm", se["max"]],
                       ["mean_pct_error", float(pct.mean())],
                       ["max_pct_error", float(pct.max())]]
            print(f"  PHYSICAL ground truth diameter={true_diameter_mm}mm: "
                 f"mean_err={se['mean']:.3f}mm ({pct.mean():.2f}%) "
                 f"max_err={se['max']:.3f}mm ({pct.max():.2f}%)")
        else:
            print("  Physical ground-truth accuracy not calculated because "
                 "--ball-diameter-mm was not supplied.")
            summary.append(["physical_ground_truth", "NOT SUPPLIED"])
        save_csv(os.path.join(out_dir, "ball_summary.csv"), summary, ["stat","value"])
        return s, n_attempt, n_ok, n_fail, diameters
    else:
        save_csv(os.path.join(out_dir, "ball_summary.csv"), summary, ["stat","value"])
        print("  No successful ball fits.")
        return None, n_attempt, n_ok, n_fail, []


# ===========================================================================
# BOX / CUBOID VALIDATION
# ===========================================================================
def validate_box(box_dir, bg, tf, out_dir, every=1,
                 true_L=None, true_W=None, true_H=None):
    print("\n" + "="*70)
    print("BOX / CUBOID VALIDATION (real box frames)")
    print("="*70)
    if not box_dir:
        print("  --box not provided -- skipping box validation.")
        return None, 0, 0, 0, []
    files = list_depth_files(box_dir)[::every]
    if not files:
        print(f"  No .npy frames found under {box_dir} -- skipping.")
        return None, 0, 0, 0, []

    rows, fail_rows, all_resid, dims_list = [], [], [], []
    for fi, f in enumerate(files):
        try:
            pts = object_points(load_masked(f), bg)
            if len(pts) < config.MIN_CLUSTER_POINTS:
                fail_rows.append([f, "box", "too few object points", len(pts)])
                continue
            planes, rmse, coverage = planes_fit(pts)
            L, W, H, c2 = box_dimensions_from_points(pts, tf)
            dims_list.append((L, W, H))
            all_resid.append(rmse)
            rows.append([os.path.basename(f), len(pts), f"{c2[0]:.3f}", f"{c2[1]:.3f}",
                        f"{L:.4f}", f"{W:.4f}", f"{H:.4f}", len(planes), f"{coverage:.4f}",
                        f"{rmse:.4f}", "success"])
        except (ValueError, np.linalg.LinAlgError) as e:
            fail_rows.append([f, "box", f"{type(e).__name__}: {e}", -1])
        except Exception as e:
            fail_rows.append([f, "box", f"UNEXPECTED {type(e).__name__}: {e}", -1])
        if (fi+1) % 100 == 0:
            print(f"  ...{fi+1}/{len(files)}")

    save_csv(os.path.join(out_dir, "box_per_frame.csv"), rows,
             ["frame","n_points","center_u","center_v","length_mm","width_mm","height_mm",
              "n_planes","coverage","planes_rmse_mm","status"])

    n_attempt, n_ok, n_fail = len(files), len(rows), len(fail_rows)
    summary = [["frames_attempted", n_attempt], ["frames_succeeded", n_ok],
              ["frames_failed", n_fail], ["failure_pct", 100*n_fail/max(n_attempt,1)]]
    if dims_list:
        dims_arr = np.array(dims_list)
        rmse_arr = np.array(all_resid)
        s_rmse = stats_block(rmse_arr)
        summary += [["mean_planes_rmse_mm", s_rmse["mean"]], ["median_planes_rmse_mm", s_rmse["median"]],
                   ["p90_planes_rmse_mm", s_rmse["p90"]], ["p95_planes_rmse_mm", s_rmse["p95"]],
                   ["max_planes_rmse_mm", s_rmse["max"]],
                   ["mean_length_mm", float(dims_arr[:,0].mean())], ["std_length_mm", float(dims_arr[:,0].std())],
                   ["mean_width_mm", float(dims_arr[:,1].mean())], ["std_width_mm", float(dims_arr[:,1].std())],
                   ["mean_height_mm", float(dims_arr[:,2].mean())], ["std_height_mm", float(dims_arr[:,2].std())]]
        print(f"  frames: attempted={n_attempt} succeeded={n_ok} failed={n_fail} "
             f"({100*n_fail/max(n_attempt,1):.1f}% failure)")
        print(f"  planes_rmse: mean={s_rmse['mean']:.4f}mm median={s_rmse['median']:.4f}mm "
             f"p95={s_rmse['p95']:.4f}mm max={s_rmse['max']:.4f}mm")
        print(f"  NOTE: a prior synthetic diagnostic found ~1.32mm RMSE even at "
             f"ZERO synthetic noise for this box-fitting method (RANSAC edge "
             f"cross-contamination in planes_fit's 4mm threshold). Compare the "
             f"real median above ({s_rmse['median']:.4f}mm) against that floor "
             f"yourself -- this script does not editorialize on it.")
        print(f"  estimated dims: L={dims_arr[:,0].mean():.2f}+/-{dims_arr[:,0].std():.3f}  "
             f"W={dims_arr[:,1].mean():.2f}+/-{dims_arr[:,1].std():.3f}  "
             f"H={dims_arr[:,2].mean():.2f}+/-{dims_arr[:,2].std():.3f} mm")
        if true_L is not None and true_W is not None and true_H is not None:
            true_vals = np.array([true_L, true_W, true_H])
            errs = np.abs(dims_arr - true_vals)
            pct = 100 * errs / true_vals
            for i, name in enumerate(["length","width","height"]):
                se = stats_block(errs[:,i])
                summary += [[f"true_{name}_mm", true_vals[i]],
                           [f"mean_abs_{name}_error_mm", se["mean"]],
                           [f"median_abs_{name}_error_mm", se["median"]],
                           [f"rmse_{name}_error_mm", se["rmse"]],
                           [f"p95_{name}_error_mm", se["p95"]],
                           [f"max_{name}_error_mm", se["max"]],
                           [f"mean_pct_error_{name}", float(pct[:,i].mean())]]
                print(f"  PHYSICAL {name}: true={true_vals[i]}mm mean_err="
                     f"{se['mean']:.3f}mm ({pct[:,i].mean():.2f}%) "
                     f"max_err={se['max']:.3f}mm ({pct[:,i].max():.2f}%)")
        else:
            print("  Physical ground-truth accuracy not calculated because "
                 "--box-length-mm/--box-width-mm/--box-height-mm were not "
                 "all supplied.")
            summary.append(["physical_ground_truth", "NOT SUPPLIED"])
        save_csv(os.path.join(out_dir, "box_summary.csv"), summary, ["stat","value"])
        return s_rmse, n_attempt, n_ok, n_fail, dims_list
    else:
        save_csv(os.path.join(out_dir, "box_summary.csv"), summary, ["stat","value"])
        print("  No successful box fits.")
        return None, n_attempt, n_ok, n_fail, []


# ===========================================================================
# PLOTS
# ===========================================================================
def make_plots(out_plots_dir, plane_resid, ball_resid, ball_diams, true_diam,
               box_rmse_arr, box_dims, true_LWH):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_plots_dir, exist_ok=True)

    if plane_resid is not None and len(plane_resid):
        fig, ax = plt.subplots(figsize=(7,4.5))
        ax.hist(plane_resid, bins=50, color="tab:blue", alpha=0.8)
        ax.set_xlabel("point-to-plane residual (mm)"); ax.set_ylabel("count")
        ax.set_title("Plane fit: residual distribution (real reference frames)")
        fig.savefig(os.path.join(out_plots_dir, "plane_residual_distribution.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    if ball_resid is not None and len(ball_resid):
        fig, ax = plt.subplots(figsize=(7,4.5))
        ax.hist(ball_resid, bins=50, color="tab:green", alpha=0.8)
        ax.set_xlabel("sphere surface residual (mm)"); ax.set_ylabel("count")
        ax.set_title("Ball fit: residual distribution (real ball frames)")
        fig.savefig(os.path.join(out_plots_dir, "ball_residual_distribution.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    if ball_diams:
        fig, ax = plt.subplots(figsize=(8,4.5))
        ax.plot(ball_diams, ".", ms=3, alpha=0.6, color="tab:green")
        if true_diam is not None:
            ax.axhline(true_diam, color="black", ls="--", label=f"true diameter = {true_diam}mm")
            ax.legend()
        ax.set_xlabel("frame index (processed order)"); ax.set_ylabel("estimated diameter (mm)")
        ax.set_title("Ball: estimated diameter across real frames")
        fig.savefig(os.path.join(out_plots_dir, "ball_diameter_across_frames.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    if box_rmse_arr is not None and len(box_rmse_arr):
        fig, ax = plt.subplots(figsize=(7,4.5))
        ax.hist(box_rmse_arr, bins=50, color="tab:red", alpha=0.8)
        ax.set_xlabel("plane-set (box) fitting RMSE (mm)"); ax.set_ylabel("count")
        ax.set_title("Box fit: RMSE distribution (real box frames)")
        fig.savefig(os.path.join(out_plots_dir, "box_rmse_distribution.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    if box_dims:
        arr = np.array(box_dims)
        fig, axes = plt.subplots(1, 3, figsize=(13,4))
        names = ["length","width","height"]
        for i, ax in enumerate(axes):
            ax.plot(arr[:,i], ".", ms=3, alpha=0.6, color="tab:red")
            if true_LWH and true_LWH[i] is not None:
                ax.axhline(true_LWH[i], color="black", ls="--")
            ax.set_title(names[i]); ax.set_xlabel("frame index"); ax.set_ylabel("mm")
        fig.suptitle("Box: estimated dimensions across real frames")
        fig.savefig(os.path.join(out_plots_dir, "box_dimensions_across_frames.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    p = argparse.ArgumentParser(
        description="Real-data geometry validation for the existing "
                    "research pipeline (does NOT retrain or alter anything).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--reference", type=str, default=None,
                   help="Path to reference/background recording (folder containing "
                        "depth/*.npy, or *.npy directly). Used for plane validation.")
    p.add_argument("--ball", type=str, default=None,
                   help="Path to a ball recording (folder containing depth/*.npy).")
    p.add_argument("--box", type=str, default=None,
                   help="Path to a box recording (folder containing depth/*.npy).")
    p.add_argument("--ball-diameter-mm", type=float, default=None,
                   help="True physical ball diameter in mm, if known. "
                        "Omit to skip physical-accuracy numbers (fit-residual "
                        "numbers are always computed regardless).")
    p.add_argument("--box-length-mm", type=float, default=None)
    p.add_argument("--box-width-mm", type=float, default=None)
    p.add_argument("--box-height-mm", type=float, default=None)
    p.add_argument("--results-dir", type=str, default="results",
                   help="Where background_median.npy and table_plane.npz already live.")
    p.add_argument("--out", type=str, default="validation_results",
                   help="Where this script writes its output.")
    p.add_argument("--every", type=int, default=1,
                   help="Process every Nth frame (1 = every frame).")
    args = p.parse_args()

    # Resolve any relative paths against the PROJECT ROOT (not the current
    # working directory), so this works the same regardless of which folder
    # you happen to run the command from.
    def _resolve(path):
        if path is None or os.path.isabs(path):
            return path
        return os.path.join(_PROJECT_ROOT, path)
    args.reference = _resolve(args.reference)
    args.ball = _resolve(args.ball)
    args.box = _resolve(args.box)
    args.results_dir = _resolve(args.results_dir)
    args.out = _resolve(args.out)

    out_dir = os.path.join(args.out, "geometry")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    print("UNITS/COORDINATES (as implemented in your existing code, checked "
         "not assumed):")
    print(f"  depth input: uint16 millimeters (per geometry.load_masked)")
    print(f"  backproject(): pinhole model, output (x,y,z) in mm, camera frame "
         f"(config.FX={config.FX}, FY={config.FY}, CX={config.CX}, CY={config.CY})")
    print(f"  TableFrame.to_table(): output (u,v,height) in mm, plane-relative")
    print(f"  sphere: radius_mm reported directly; diameter = 2*radius, computed here")
    print(f"  box: length/width from PCA on table-frame (u,v) footprint (mm), "
         f"height = 99th-percentile above plane (mm) -- ordering is "
         f"length=major axis, width=minor axis, NOT tied to any fixed "
         f"world X/Y; height is always the plane-normal direction")
    print(f"  PLANE_THRESH_MM={config.PLANE_THRESH_MM}  SPHERE_THRESH_MM={config.SPHERE_THRESH_MM}  "
         f"MIN_CLUSTER_POINTS={config.MIN_CLUSTER_POINTS}")

    # load existing saved state (already-fit plane, already-built background)
    bg_path = os.path.join(args.results_dir, "background_median.npy")
    plane_path = os.path.join(args.results_dir, "table_plane.npz")
    if not os.path.exists(bg_path) or not os.path.exists(plane_path):
        print(f"\nERROR: {bg_path} and/or {plane_path} not found. "
             f"Run build_reference.py first (this script does not build "
             f"the background model itself -- it VALIDATES the existing one).")
        sys.exit(1)
    bg = np.load(bg_path)
    # fit_static.load_table() reads config.RESULTS_DIR directly (not a
    # parameter) -- override the module constant so --results-dir actually
    # takes effect, without editing fit_static.py itself.
    config.RESULTS_DIR = args.results_dir
    tf, z = load_table()
    plane_n, plane_d = tf.n, tf.d

    plane_resid, n_pa, n_po, n_pf = validate_plane(
        args.reference, plane_n, plane_d, out_dir, args.every)
    ball_resid, n_ba, n_bo, n_bf, ball_diams = validate_ball(
        args.ball, bg, tf, out_dir, args.every, args.ball_diameter_mm)
    box_rmse, n_xa, n_xo, n_xf, box_dims = validate_box(
        args.box, bg, tf, out_dir, args.every,
        args.box_length_mm, args.box_width_mm, args.box_height_mm)

    # combined failure counts across all three components (per-frame failure
    # reasons already saved inline by each validate_* call where applicable)
    save_csv(os.path.join(out_dir, "geometry_failures_summary.csv"),
            [["plane", n_pa, n_po, n_pf, 100*n_pf/max(n_pa,1)],
             ["ball", n_ba, n_bo, n_bf, 100*n_bf/max(n_ba,1)],
             ["box", n_xa, n_xo, n_xf, 100*n_xf/max(n_xa,1)]],
            ["component","attempted","succeeded","failed","failure_pct"])

    make_plots(plots_dir, None, None, ball_diams,
              args.ball_diameter_mm, None, box_dims,
              (args.box_length_mm, args.box_width_mm, args.box_height_mm))

    # ---- Markdown report ----
    lines = ["# GEOMETRY VALIDATION REPORT (real data)\n"]
    lines.append(f"Reference dir: `{args.reference}`  \n"
                f"Ball dir: `{args.ball}`  \n"
                f"Box dir: `{args.box}`  \n"
                f"Every Nth frame processed: {args.every}\n")
    lines.append("\n## PLANE\n")
    if plane_resid:
        lines.append(f"- Frames: attempted={n_pa}, succeeded={n_po}, failed={n_pf} "
                    f"({100*n_pf/max(n_pa,1):.1f}% failure)\n"
                    f"- Residual (mm): mean={plane_resid['mean']:.4f}, "
                    f"median={plane_resid['median']:.4f}, rmse={plane_resid['rmse']:.4f}, "
                    f"p95={plane_resid['p95']:.4f}, max={plane_resid['max']:.4f}\n")
    else:
        lines.append("- Not evaluated (no --reference provided or no successful frames).\n")
    lines.append("\n## BALL\n")
    if ball_resid:
        lines.append(f"- Frames: attempted={n_ba}, succeeded={n_bo}, failed={n_bf} "
                    f"({100*n_bf/max(n_ba,1):.1f}% failure)\n"
                    f"- Surface residual (mm): mean={ball_resid['mean']:.4f}, "
                    f"median={ball_resid['median']:.4f}, rmse={ball_resid['rmse']:.4f}, "
                    f"p95={ball_resid['p95']:.4f}, max={ball_resid['max']:.4f}\n"
                    f"- Estimated diameter: mean={np.mean(ball_diams):.3f}mm, "
                    f"std={np.std(ball_diams):.4f}mm\n")
        if args.ball_diameter_mm:
            lines.append(f"- Physical ground truth diameter: {args.ball_diameter_mm}mm "
                        f"(see ball_summary.csv for full error breakdown)\n")
        else:
            lines.append("- Physical ground-truth accuracy not calculated because "
                        "physical dimensions were not supplied.\n")
    else:
        lines.append("- Not evaluated (no --ball provided or no successful frames).\n")
    lines.append("\n## BOX\n")
    if box_rmse:
        lines.append(f"- Frames: attempted={n_xa}, succeeded={n_xo}, failed={n_xf} "
                    f"({100*n_xf/max(n_xa,1):.1f}% failure)\n"
                    f"- Plane-set fitting RMSE (mm): mean={box_rmse['mean']:.4f}, "
                    f"median={box_rmse['median']:.4f}, p95={box_rmse['p95']:.4f}, "
                    f"max={box_rmse['max']:.4f}\n"
                    f"- NOTE: a prior synthetic diagnostic found ~1.32mm RMSE at ZERO "
                    f"synthetic noise for this method. Compare against the real median "
                    f"above yourself.\n")
        if all(v is not None for v in [args.box_length_mm, args.box_width_mm, args.box_height_mm]):
            lines.append("- Physical ground truth supplied — see box_summary.csv for "
                        "full per-dimension error breakdown.\n")
        else:
            lines.append("- Physical ground-truth accuracy not calculated because "
                        "physical dimensions were not supplied.\n")
    else:
        lines.append("- Not evaluated (no --box provided or no successful frames).\n")

    report_path = os.path.join(out_dir, "geometry_validation_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines))

    # ---- terminal summary ----
    print("\n" + "#"*70)
    print("# GEOMETRY VALIDATION SUMMARY")
    print("#"*70)
    print("\nPLANE")
    if plane_resid:
        print(f"  Frames evaluated:  {n_po}/{n_pa}")
        print(f"  Median residual:   {plane_resid['median']:.4f} mm")
        print(f"  RMSE:              {plane_resid['rmse']:.4f} mm")
        print(f"  95th percentile:   {plane_resid['p95']:.4f} mm")
    else:
        print("  not evaluated")
    print("\nBALL")
    if ball_resid:
        print(f"  Frames evaluated:      {n_bo}/{n_ba}")
        print(f"  Median surface resid:  {ball_resid['median']:.4f} mm")
        print(f"  RMSE:                  {ball_resid['rmse']:.4f} mm")
        print(f"  Estimated diameter:    {np.mean(ball_diams):.3f} mm")
        print(f"  Physical diameter:     {args.ball_diameter_mm if args.ball_diameter_mm else 'not supplied'}")
    else:
        print("  not evaluated")
    print("\nBOX")
    if box_rmse:
        print(f"  Frames evaluated:      {n_xo}/{n_xa}")
        print(f"  Median fitting resid:  {box_rmse['median']:.4f} mm")
        arr = np.array(box_dims)
        print(f"  Estimated dimensions:  L={arr[:,0].mean():.2f} W={arr[:,1].mean():.2f} H={arr[:,2].mean():.2f} mm")
        print(f"  Physical dimensions:   L={args.box_length_mm} W={args.box_width_mm} H={args.box_height_mm}")
    else:
        print("  not evaluated")
    print(f"\nFull report: {report_path}")


if __name__ == "__main__":
    main()