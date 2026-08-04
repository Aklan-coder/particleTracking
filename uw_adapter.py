"""
uw_adapter.py — Convert UW RGB-D Object Dataset frames into our cell-patch
format, so models trained on OUR recordings can be tested on THEIR objects
(cross-dataset generalization exam).

UW format per frame (e.g. ball_1_1_100_*):
  *_depthcrop.png  16-bit PNG, depth in mm, cropped around the object
  *_maskcrop.png   binary mask: which crop pixels belong to the object
  *_loc.txt        "x,y" top-left of the crop in the full 640x480 frame
UW camera intrinsics (Kinect-class, 640x480): fx = fy = 570.3,
cx = 320, cy = 240. loc offsets restore absolute pixel coordinates.

IMPORTANT HONEST DIFFERENCE: our features use height above the TABLE;
UW crops contain no table. We use camera-relief instead: h = (far surface)
- (point depth), i.e. how much each point bulges toward the camera
relative to the object's farthest surface. A ball produces a dome of
relief; a flat box face produces a plateau — the same shape signature,
but ABSOLUTE height values will not transfer (our ball 'height' peaked at
~68 mm above table; UW relief peaks near the object's own radius). That
mismatch is not a bug: exposing which learned features survive a domain
shift is the point of the exam.

Usage: python3 uw_adapter.py ~/Downloads/rgbd-dataset
Output: results/uw_dataset_h{10,20,30,40}.npz  (X, n, y, meta)
"""
import glob
import os
import sys

import numpy as np

import config
from make_dataset import PATCH

UW_FX = UW_FY = 570.3
UW_CX, UW_CY = 320.0, 240.0
H_LIST = [10.0, 20.0, 30.0, 40.0]
LABEL = {"ball": 0, "cereal_box": 1}
EVERY = 5          # every 5th frame is plenty (turntable frames repeat)


def read_png16(path):
    """Minimal 16-bit grayscale PNG reader via matplotlib (no cv2 need)."""
    import matplotlib.image as mpimg
    img = mpimg.imread(path)
    if img.dtype == np.uint8:
        return img.astype(np.uint16)
    if img.dtype in (np.uint16,):
        return img
    # matplotlib normalizes 16-bit PNGs to float 0..1 -> rescale
    return (img * 65535.0 + 0.5).astype(np.uint16)


def frame_points(depth_path):
    base = depth_path[:-len("_depthcrop.png")]
    mask = read_png16(base + "_maskcrop.png") > 0
    depth = read_png16(depth_path).astype(float)
    with open(base + "_loc.txt") as fh:
        x0, y0 = (int(t) for t in fh.read().strip().split(","))
    v, u = np.nonzero(mask & (depth > 0))
    if len(v) < 30:
        return None
    z = depth[v, u]
    x = (u + x0 - UW_CX) * z / UW_FX
    y = (v + y0 - UW_CY) * z / UW_FY
    # camera-relief 'height': bulge toward camera vs the far surface
    h = np.percentile(z, 99) - z
    # local table-like coordinates: (x, y) in mm, relief as height
    return np.column_stack([x, y, h])


def patch_from_points(uvh, h_cell):
    """Object-centered PATCH x PATCH cell patch, same channels as
    make_dataset (occupancy, mean relief, relief variance)."""
    u0, v0 = uvh[:, 0].min(), uvh[:, 1].min()
    i = ((uvh[:, 0] - u0) / h_cell).astype(int)
    j = ((uvh[:, 1] - v0) / h_cell).astype(int)
    m, n = i.max() + 1, j.max() + 1
    flat = i * n + j
    size = m * n
    cnt = np.bincount(flat, minlength=size).astype(float)
    sh = np.bincount(flat, weights=uvh[:, 2], minlength=size)
    sh2 = np.bincount(flat, weights=uvh[:, 2] ** 2, minlength=size)
    with np.errstate(divide="ignore", invalid="ignore"):
        mh = np.where(cnt > 0, sh / cnt, 0.0)
        vh = np.where(cnt > 0, np.maximum(sh2 / cnt - mh ** 2, 0), 0.0)
    occ = (cnt > 0).reshape(m, n)
    mh, vh = mh.reshape(m, n), vh.reshape(m, n)
    ii, jj = np.nonzero(occ)
    ci, cj = int(round(ii.mean())), int(round(jj.mean()))
    half = PATCH // 2
    out = np.zeros((3, PATCH, PATCH), dtype=np.float32)
    for pi in range(PATCH):
        for pj in range(PATCH):
            gi, gj = ci - half + pi, cj - half + pj
            if 0 <= gi < m and 0 <= gj < n and occ[gi, gj]:
                out[0, pi, pj] = 1.0
                out[1, pi, pj] = mh[gi, gj]
                out[2, pi, pj] = vh[gi, gj]
    return out, int(occ.sum())


def main(root):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    for h in H_LIST:
        X, N, Y, META = [], [], [], []
        for cat, label in LABEL.items():
            files = sorted(glob.glob(
                os.path.join(root, cat, "*", "*_depthcrop.png")))[::EVERY]
            kept = 0
            for f in files:
                uvh = frame_points(f)
                if uvh is None:
                    continue
                patch, ncells = patch_from_points(uvh, h)
                X.append(patch)
                N.append(ncells)
                Y.append(label)
                META.append(os.path.basename(f))
                kept += 1
            print(f"h={h:.0f}  {cat}: {kept} frames", flush=True)
        out = os.path.join(config.RESULTS_DIR,
                           f"uw_dataset_h{int(h)}.npz")
        np.savez_compressed(out, X=np.array(X), n=np.array(N),
                            y=np.array(Y), meta=np.array(META))
        print(f"saved {out} ({len(Y)} examples)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else os.path.expanduser("~/Downloads/rgbd-dataset"))