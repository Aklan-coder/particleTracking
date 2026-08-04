"""
fit_static.py — Shape fits on the static recordings.

Purpose (three results in one script):
  1. INTRINSICS VALIDATION: the fitted ball diameter vs the ruler-measured
     diameter is the one-time check of the nominal focal length. A
     consistent ratio between them IS the focal-length correction factor.
  2. FIT PRECISION: the scatter of fitted centers across ~1000 frames of an
     unmoving object is the empirical measurement noise — it feeds the
     Kalman filter's R matrix (real numbers, not guesses).
  3. OBJECT GROUND TRUTH: mean ball radius and box dimensions become the
     known shapes used by the shape-conditioned motion models.

Outputs: results/ball_static_fits.csv, results/box_static_fits.csv,
         results/static_summary.txt

Usage:  python3 fit_static.py            (expects Data/extracted/...)
        python3 fit_static.py --every 5  (fit every 5th frame; default 5)
"""
import csv
import glob
import os
import sys

import numpy as np

import config
from geometry import (TableFrame, backproject, load_masked, planes_fit,
                      ransac_sphere)


def load_table():
    z = np.load(os.path.join(config.RESULTS_DIR, "table_plane.npz"))
    tf = TableFrame(z["n"], float(z["d"]), origin=z["origin"])
    tf.e1, tf.e2 = z["e1"], z["e2"]   # exact axes from build_reference
    return tf, z


def object_points(depth, bg):
    """Background-subtract one frame against the median model.

    A pixel is 'object' if valid in both frame and model and closer to the
    camera than the background by more than BG_DIFF_MM (objects sit ON the
    table, hence strictly closer).
    """
    valid = (depth > 0) & (bg > 0)
    obj_mask = valid & (bg - depth > config.BG_DIFF_MM)
    d = np.where(obj_mask, depth, 0.0)
    pts, _ = backproject(d)
    return pts


def fit_ball_frames(files, bg, tf, writer):
    rs, centers = [], []
    for k, f in enumerate(files):
        pts = object_points(load_masked(f), bg)
        if len(pts) < config.MIN_CLUSTER_POINTS:
            continue
        try:
            c, r, inl, rmse = ransac_sphere(pts)
        except (ValueError, np.linalg.LinAlgError):
            continue
        uvh = tf.to_table(c)[0]
        writer.writerow([os.path.basename(f), f"{c[0]:.2f}", f"{c[1]:.2f}",
                         f"{c[2]:.2f}", f"{uvh[0]:.2f}", f"{uvh[1]:.2f}",
                         f"{uvh[2]:.2f}", f"{r:.3f}", f"{rmse:.3f}",
                         int(inl.sum())])
        rs.append(r)
        centers.append(uvh)
        if (k + 1) % 40 == 0:
            print(f"  ...{k + 1}/{len(files)}")
    return np.array(rs), np.array(centers)


def fit_box_frames(files, bg, tf, writer):
    """Box static fits: dimensions from the table-frame footprint + height.

    Dimensions via PCA of the footprint (principal axes of the points
    projected on the table): length/width = extent along the two in-plane
    principal directions, height = max height above plane. PCA gives the
    box's orientation without assuming axis alignment.
    """
    dims, centers = [], []
    for k, f in enumerate(files):
        pts = object_points(load_masked(f), bg)
        if len(pts) < config.MIN_CLUSTER_POINTS:
            continue
        try:
            _, rmse, coverage = planes_fit(pts)
        except (ValueError, np.linalg.LinAlgError):
            continue
        uvh = tf.to_table(pts)
        uv = uvh[:, :2]
        c2 = uv.mean(axis=0)
        # PCA on the footprint for oriented length/width.
        A = uv - c2
        cov = A.T @ A / len(A)
        w, V = np.linalg.eigh(cov)
        proj = A @ V             # coordinates along principal axes
        # Extents via 1st-99th percentiles, NOT min/max: absolute extremes
        # are hijacked by single flying pixels at the object boundary (the
        # measured cause of a +/-17 mm width instability on real data),
        # while percentiles are robust to lone outliers and near-identical
        # for clean data.
        lo, hi = np.percentile(proj[:, 1], [1, 99])
        length = float(hi - lo)                       # major axis
        lo, hi = np.percentile(proj[:, 0], [1, 99])
        width = float(hi - lo)
        height = float(np.percentile(uvh[:, 2], 99))
        writer.writerow([os.path.basename(f), f"{c2[0]:.2f}", f"{c2[1]:.2f}",
                         f"{length:.2f}", f"{width:.2f}", f"{height:.2f}",
                         f"{rmse:.3f}", f"{coverage:.2f}"])
        dims.append((length, width, height))
        centers.append(c2)
        if (k + 1) % 40 == 0:
            print(f"  ...{k + 1}/{len(files)}")
    return np.array(dims), np.array(centers)


def main(every=5):
    bg = np.load(os.path.join(config.RESULTS_DIR, "background_median.npy"))
    tf, _ = load_table()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    summary = []

    # ---- ball ----
    files = sorted(glob.glob("Data/extracted/ball_static/depth/*.npy"))[::every]
    print(f"ball_static: fitting {len(files)} frames...")
    with open(os.path.join(config.RESULTS_DIR, "ball_static_fits.csv"),
              "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "cx_mm", "cy_mm", "cz_mm", "u_mm", "v_mm",
                    "h_mm", "radius_mm", "rmse_mm", "inliers"])
        rs, cs = fit_ball_frames(files, bg, tf, w)
    if len(rs):
        s = (f"BALL: {len(rs)} fits | radius {rs.mean():.2f} ± "
             f"{rs.std():.3f} mm  ->  DIAMETER {2*rs.mean():.2f} ± "
             f"{2*rs.std():.3f} mm\n"
             f"      center scatter (std u, v, h): "
             f"{np.round(cs.std(axis=0), 3)} mm\n"
             f"      >>> compare DIAMETER to the ruler measurement; the "
             f"ratio ruler/fit is the focal-length correction. <<<")
        print(s)
        summary.append(s)

    # ---- box ----
    files = sorted(glob.glob("Data/extracted/box_static/depth/*.npy"))[::every]
    print(f"box_static: fitting {len(files)} frames...")
    with open(os.path.join(config.RESULTS_DIR, "box_static_fits.csv"),
              "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "u_mm", "v_mm", "length_mm", "width_mm",
                    "height_mm", "planes_rmse_mm", "coverage"])
        dims, cs = fit_box_frames(files, bg, tf, w)
    if len(dims):
        m, sd = dims.mean(axis=0), dims.std(axis=0)
        s = (f"BOX:  {len(dims)} fits | L x W x H = "
             f"{m[0]:.1f}±{sd[0]:.2f} x {m[1]:.1f}±{sd[1]:.2f} x "
             f"{m[2]:.1f}±{sd[2]:.2f} mm\n"
             f"      center scatter (std u, v): {np.round(cs.std(axis=0), 3)} mm")
        print(s)
        summary.append(s)

    with open(os.path.join(config.RESULTS_DIR, "static_summary.txt"), "w") as fh:
        fh.write("\n".join(summary) + "\n")
    print("saved results/*_static_fits.csv and results/static_summary.txt")


if __name__ == "__main__":
    every = 1
    if "--every" in sys.argv:
        every = int(sys.argv[sys.argv.index("--every") + 1])
    main(every)