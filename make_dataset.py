"""
make_dataset.py — Phase 2, step 1: build per-resolution training datasets
from the recordings, with geometry-produced labels.

For every frame of the four single-object recordings, the object's cell
grid (the professor's 'matrix of values') is extracted at each training
resolution and stored with its free label (recording name -> class).

Example = a fixed-size patch of per-cell features centered on the object:
  channel 0: occupancy (0/1)
  channel 1: mean height above table (mm)
  channel 2: height variance (mm^2)  <- the dome-vs-plateau signature
The patch is PATCH x PATCH cells (object-centered), so the model sees the
local pattern; total occupied-cell count is stored separately so we can
test 'pattern only' vs 'pattern + size' (how much of identification is
shape vs mere size?).

Split: BY TIME, per recording - first 70% of frames -> train, last 30%
-> test. Never random: consecutive frames are near-duplicates and a
random split would leak them across the boundary.

Output: results/dataset_h{h}.npz with arrays
  X_train/X_test  (N, 3, PATCH, PATCH) float32
  n_train/n_test  (N,) occupied-cell counts
  y_train/y_test  (N,) 0 = ball, 1 = box
  meta            recording name + frame index per example

Usage: python3 make_dataset.py            (h = 10, 20, 30, 40)
       python3 make_dataset.py --every 2  (sample every 2nd frame)
"""
import glob
import os
import sys

import numpy as np

import config
from discretize import Partition
from fit_static import load_table, object_points
from geometry import load_masked

H_TRAIN = [10.0, 20.0, 30.0, 40.0]
PATCH = 13            # cells per side of the object-centered patch
TRAIN_FRAC = 0.70
RECORDINGS = {"ball_static": 0, "ball_moving": 0,
              "box_static": 1, "box_moving": 1}


def patch_for(cluster_uvh, part):
    """Object-centered PATCH x PATCH feature patch from one cluster."""
    g = part.grids(cluster_uvh)
    occ = g["count"] > 0
    ii, jj = np.nonzero(occ)
    if len(ii) == 0:
        return None, 0
    ci, cj = int(round(ii.mean())), int(round(jj.mean()))
    half = PATCH // 2
    out = np.zeros((3, PATCH, PATCH), dtype=np.float32)
    for pi in range(PATCH):
        for pj in range(PATCH):
            gi, gj = ci - half + pi, cj - half + pj
            if 0 <= gi < part.m and 0 <= gj < part.n and occ[gi, gj]:
                out[0, pi, pj] = 1.0
                out[1, pi, pj] = g["mean_h"][gi, gj]
                out[2, pi, pj] = g["var_h"][gi, gj]
    return out, int(occ.sum())


def main(every=1):
    bg = np.load(os.path.join(config.RESULTS_DIR, "background_median.npy"))
    tf, z = load_table()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    for h in H_TRAIN:
        part = Partition(z["u_range"], z["v_range"], h=h)
        Xtr, ntr, ytr, Xte, nte, yte = [], [], [], [], [], []
        meta_tr, meta_te = [], []
        for rec, label in RECORDINGS.items():
            files = sorted(glob.glob(
                f"Data/extracted/{rec}/depth/*.npy"))[::every]
            cut = int(TRAIN_FRAC * len(files))   # time split point
            for k, f in enumerate(files):
                pts = object_points(load_masked(f), bg)
                if len(pts) < 10:
                    continue
                uvh = tf.to_table(pts)
                clusters, _ = part.clusters(uvh)
                if not clusters:
                    continue
                cl = clusters[0]
                patch, ncells = patch_for(uvh[cl["points_mask"]], part)
                if patch is None:
                    continue
                is_train = k < cut               # by time, never random
                (Xtr if is_train else Xte).append(patch)
                (ntr if is_train else nte).append(ncells)
                (ytr if is_train else yte).append(label)
                (meta_tr if is_train else meta_te).append(f"{rec}:{k}")
            print(f"  h={h:.0f}  {rec}: done "
                  f"({cut} train cut of {len(files)})", flush=True)
        out = os.path.join(config.RESULTS_DIR, f"dataset_h{int(h)}.npz")
        np.savez_compressed(
            out,
            X_train=np.array(Xtr), n_train=np.array(ntr),
            y_train=np.array(ytr), meta_train=np.array(meta_tr),
            X_test=np.array(Xte), n_test=np.array(nte),
            y_test=np.array(yte), meta_test=np.array(meta_te))
        print(f"saved {out}: train {len(ytr)} (ball "
              f"{ytr.count(0)}, box {ytr.count(1)}), "
              f"test {len(yte)}", flush=True)


if __name__ == "__main__":
    every = (int(sys.argv[sys.argv.index("--every") + 1])
             if "--every" in sys.argv else 1)
    main(every)