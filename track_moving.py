"""
track_moving.py — The phase-1 baseline: per-frame partition -> cluster ->
classify (no prior) -> Kalman tracking, on one moving recording.

Per-frame pipeline (the partition IS the segmentation, per the professor):
  1. background-subtract against the median reference model
  2. back-project surviving pixels to 3D, transform to table coordinates
  3. discretize into the m x n grid (cell size = config.H_BASELINE)
  4. connected components of occupied cells -> clusters (full-res points kept)
  5. classify each cluster by competing geometric hypotheses
     (ball / box / unknown — 'unknown' plotted as grey S, per convention)
  6. associate detections to Kalman tracks; update

Outputs per recording under results/:
  <name>_tracks.csv      one row per confirmed track per frame
  <name>_cells.npz       per-frame cell value matrices (count/mean/max/var)
                         — the professor's non-binary cells, archived as the
                         future phase-2 feature bank
  <name>_trajectory.png  static trajectory figure (O blue / X red / S grey)
  <name>_tracking.gif    animated overlay: depth video + live markers +
                         short trails (the 'show the professor' artifact)

Usage:  python3 track_moving.py ball_moving
        python3 track_moving.py ball_and_box_moving --gif-every 3
"""
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")           # headless-safe; files only
import matplotlib.pyplot as plt
import numpy as np

import config
from discretize import Partition
from fit_static import load_table, object_points
from geometry import backproject, classify_cluster, load_masked
from merge_split import default_body_radius, split_merged
from tracking import Tracker


def measurement_std():
    """Kalman R from the MEASURED static-fit scatter if available; falls
    back to a conservative 2 mm before fit_static.py has been run."""
    path = os.path.join(config.RESULTS_DIR, "ball_static_fits.csv")
    try:
        u, v = np.loadtxt(path, delimiter=",", skiprows=1,
                          usecols=(4, 5), unpack=True)
        return float(max(np.std(u), np.std(v), 0.3))
    except Exception:
        return 2.0


def process(name, gif_every=3, h=None):
    depth_dir = f"Data/extracted/{name}/depth"
    files = sorted(glob.glob(os.path.join(depth_dir, "*.npy")))
    if not files:
        raise SystemExit(f"no frames in {depth_dir}")
    bg = np.load(os.path.join(config.RESULTS_DIR, "background_median.npy"))
    tf, z = load_table()
    part = Partition(z["u_range"], z["v_range"], h=h)
    print(f"{name}: {len(files)} frames | grid {part.m} x {part.n} cells "
          f"@ h={part.h:.0f} mm")

    tracker = Tracker(measurement_std())
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    rows = []
    cell_archive = {}
    snapshots = []     # (frame_idx, depth, detections) for the GIF

    for fi, f in enumerate(files):
        depth = load_masked(f)
        pts = object_points(depth, bg)
        detections = []
        # Predictions of currently confirmed tracks (for merge splitting):
        # where does each existing object EXPECT to be this frame, and how
        # big is its body? Computed BEFORE tracker.step (which re-predicts).
        preds = []
        for t in tracker.tracks:
            if not t.confirmed:
                continue
            pu, pv = (t.kf.F @ t.kf.x)[:2]
            preds.append(dict(uv=(pu, pv), kind=t.kind,
                              size=default_body_radius(t.kind)))
        if len(pts) >= config.MIN_CLUSTER_POINTS:
            uvh = tf.to_table(pts)
            clusters, grids = part.clusters(uvh)
            # Archive the non-binary cell matrices for this frame.
            cell_archive[f"f{fi:06d}"] = np.stack(
                [grids["count"], grids["mean_h"],
                 grids["max_h"], grids["var_h"]]).astype(np.float32)
            for cl in clusters:
                cpts = pts[cl["points_mask"]]
                cuvh = uvh[cl["points_mask"]]
                # ---- MERGE SPLITTING (shape-aware, prediction-guided) ----
                # If this single cluster covers the predicted bodies of 2+
                # confirmed tracks, it is two touching objects seen as one
                # blob. Divide its points by nearest predicted BODY so both
                # tracks keep receiving measurements (identity survives
                # contact). Purely geometric; see merge_split.py.
                overlapping = [p for p in preds
                               if np.min(np.linalg.norm(
                                   cuvh[:, :2] - np.asarray(p["uv"]),
                                   axis=1)) < p["size"] + part.h]
                if len(overlapping) >= 2:
                    parts3d = split_merged(cuvh, cpts, overlapping)
                    for (sub_pts, sub_uvh), pinfo in zip(parts3d,
                                                         overlapping):
                        if len(sub_pts) < 10:
                            continue
                        res = classify_cluster(sub_pts)
                        cuv = tf.to_table(res["center"])[0][:2]
                        ci, cj = part.cell_of(sub_uvh)
                        ok = ((ci >= 0) & (ci < part.m) &
                              (cj >= 0) & (cj < part.n))
                        cells = np.unique(
                            np.column_stack([ci[ok], cj[ok]]), axis=0)
                        detections.append(dict(
                            uv=cuv, kind=res["kind"], size=res["size"],
                            rmse=res["rmse"], n_cells=len(cells),
                            cells=cells))
                    continue
                # ---- normal (unmerged) path ----
                res = classify_cluster(cpts)
                cuv = tf.to_table(res["center"])[0][:2]
                # cell indices occupied by THIS cluster (for the grid view
                # of the GIF: each object's cells painted in its class
                # color, per the professor's symbol/color convention)
                ci, cj = part.cell_of(uvh[cl["points_mask"]])
                ok = (ci >= 0) & (ci < part.m) & (cj >= 0) & (cj < part.n)
                cells = np.unique(np.column_stack([ci[ok], cj[ok]]), axis=0)
                detections.append(dict(uv=cuv, kind=res["kind"],
                                       size=res["size"], rmse=res["rmse"],
                                       n_cells=cl["n_cells"], cells=cells))
        confirmed = tracker.step(fi, detections)
        for t in confirmed:
            fr, u, v, du, dv, kind_now = t.history[-1]
            if fr != fi:
                continue
            rows.append([fi, f"{fi*config.DT:.3f}", t.id, t.kind, kind_now,
                         f"{u:.2f}", f"{v:.2f}", f"{du:.1f}", f"{dv:.1f}"])
        if fi % gif_every == 0:
            snapshots.append((fi, depth.copy(),
                              [(d["uv"], d["kind"], d.get("cells"))
                               for d in detections]))
        if (fi + 1) % 200 == 0:
            print(f"  ...{fi + 1}/{len(files)} | active tracks: "
                  f"{len(tracker.tracks)}")

    # ---- save tracks CSV ----
    csv_path = os.path.join(config.RESULTS_DIR, f"{name}_tracks.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "t_s", "track_id", "track_kind", "kind_now",
                    "u_mm", "v_mm", "du_mm_s", "dv_mm_s"])
        w.writerows(rows)
    print(f"saved {csv_path} ({len(rows)} rows)")

    # ---- save cell archive ----
    npz_path = os.path.join(config.RESULTS_DIR, f"{name}_cells.npz")
    np.savez_compressed(npz_path, **cell_archive)
    print(f"saved {npz_path} ({len(cell_archive)} frames of cell matrices)")

    # ---- trajectory figure ----
    fig, ax = plt.subplots(figsize=(8, 6))
    data = np.array([[float(r[5]), float(r[6])] for r in rows])
    kinds = [r[3] for r in rows]
    for kind, st in config.STYLE.items():
        sel = np.array([k == kind for k in kinds])
        if sel.any():
            ax.scatter(data[sel, 0], data[sel, 1], s=14, alpha=0.6, **st)
    ax.set_xlabel("u (mm, table frame)")
    ax.set_ylabel("v (mm, table frame)")
    ax.set_title(f"{name}: tracked trajectories "
                 f"(grid {part.m}x{part.n} @ {part.h:.0f} mm)")
    ax.legend()
    ax.set_aspect("equal")
    fig_path = os.path.join(config.RESULTS_DIR, f"{name}_trajectory.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {fig_path}")

    # ---- animated GIF (depth view + grid partition view) ----
    make_gif(name, snapshots, tf, part)


KIND_CELL_COLOR = {"ball": "tab:blue", "box": "tab:red",
                   "unknown": "grey"}


def make_gif(name, snapshots, tf, part, trail=12):
    """Two-panel animation, per the professor's framing:

      LEFT  — what the CAMERA sees: depth video + class markers (O/X/S).
      RIGHT — what the GRID (simulated radar array) sees: the m x n
              partition with each cluster's occupied cells painted in its
              class color (ball = blue, box = red, unknown = grey) and the
              class symbol at the tracked position.

    Also saves a single-frame PNG snapshot (mid-recording) of the same
    two-panel view for reports.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import ListedColormap

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.8, 4.8))
    img = axL.imshow(np.where(snapshots[0][1] > 0, snapshots[0][1], np.nan),
                     cmap="turbo")
    axL.set_title(f"{name} — camera view")
    scatters = {k: axL.scatter([], [], s=90, linewidths=2, **st)
                for k, st in config.STYLE.items()}
    trail_sc = axL.scatter([], [], s=8, c="white", alpha=0.5)
    recent = []

    # Grid panel: categorical image over cells.
    # 0 empty, 1 ball, 2 box, 3 unknown.
    cmap = ListedColormap(["#f0f0f0", "tab:blue", "tab:red", "grey"])
    KIND_CODE = {"ball": 1, "box": 2, "unknown": 3}
    grid_img = axR.imshow(np.zeros((part.n, part.m)), cmap=cmap,
                          vmin=0, vmax=3, origin="lower",
                          interpolation="nearest")
    # Draw the PARTITION itself: a line at every cell boundary, so the
    # panel reads as space divided into m x n cells (the professor's
    # framing), not as free-floating colored blobs. Minor ticks at
    # half-integers land exactly on imshow cell edges.
    axR.set_xticks(np.arange(-0.5, part.m, 1), minor=True)
    axR.set_yticks(np.arange(-0.5, part.n, 1), minor=True)
    axR.grid(which="minor", color="white", linewidth=0.6)
    axR.tick_params(which="minor", length=0)
    axR.set_title(f"grid view — {part.m} x {part.n} cells @ "
                  f"{part.h:.0f} mm")
    axR.set_xlabel("u cell index")
    axR.set_ylabel("v cell index")
    grid_marks = {k: axR.scatter([], [], s=140, linewidths=2.5, **st)
                  for k, st in config.STYLE.items()}

    def uv_to_pixel(uv):
        cam = tf.to_camera(np.array([[uv[0], uv[1], 30.0]]))[0]
        u_px = cam[0] * config.FX / cam[2] + config.CX
        v_px = cam[1] * config.FY / cam[2] + config.CY
        return u_px, v_px

    def uv_to_cell(uv):
        i = (uv[0] - part.u0) / part.h - 0.5
        j = (uv[1] - part.v0) / part.h - 0.5
        return i, j

    def update(k):
        fi, depth, dets = snapshots[k]
        img.set_data(np.where(depth > 0, depth, np.nan))
        pts = {key: [] for key in config.STYLE}
        cell_field = np.zeros((part.n, part.m))
        gpts = {key: [] for key in config.STYLE}
        for uv, kind, cells in dets:
            pts.setdefault(kind, []).append(uv_to_pixel(uv))
            recent.append(uv_to_pixel(uv))
            if cells is not None and len(cells):
                cell_field[cells[:, 1], cells[:, 0]] = KIND_CODE.get(
                    kind, 3)
            gpts.setdefault(kind, []).append(uv_to_cell(uv))
        del recent[:-trail * 3]
        for key, sc in scatters.items():
            arr = np.array(pts[key]) if pts[key] else np.empty((0, 2))
            sc.set_offsets(arr)
        for key, sc in grid_marks.items():
            arr = np.array(gpts[key]) if gpts[key] else np.empty((0, 2))
            sc.set_offsets(arr)
        trail_sc.set_offsets(np.array(recent) if recent
                             else np.empty((0, 2)))
        grid_img.set_data(cell_field)
        axL.set_xlabel(f"frame {fi}")
        return [img, trail_sc, grid_img,
                *scatters.values(), *grid_marks.values()]

    # PNG snapshot: pick a mid-recording frame that has detections.
    snap_k = next((k for k in range(len(snapshots) // 2,
                                    len(snapshots))
                   if snapshots[k][2]), len(snapshots) // 2)
    update(snap_k)
    png_path = os.path.join(config.RESULTS_DIR, f"{name}_grid.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"saved {png_path}")

    anim = FuncAnimation(fig, update, frames=len(snapshots), blit=False)
    gif_path = os.path.join(config.RESULTS_DIR, f"{name}_tracking.gif")
    anim.save(gif_path, writer=PillowWriter(fps=10))
    plt.close(fig)
    print(f"saved {gif_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python3 track_moving.py <recording_name> "
                         "[--gif-every N] [--h MM]")
    name = sys.argv[1]
    gif_every = (int(sys.argv[sys.argv.index("--gif-every") + 1])
                 if "--gif-every" in sys.argv else 3)
    h = (float(sys.argv[sys.argv.index("--h") + 1])
         if "--h" in sys.argv else None)
    process(name, gif_every=gif_every, h=h)