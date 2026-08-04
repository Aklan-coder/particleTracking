"""
build_reference.py — One-time processing of the reference (empty scene)
recording into two reusable artifacts:

  results/background_median.npy — per-pixel MEDIAN depth over all reference
      frames. Median (not mean) is robust to transient artifacts, and
      averaging ~800 frames drives the model's noise far below the 0.77 mm
      per-frame floor: a near-noiseless portrait of the empty scene.
  results/table_plane.npz — the definitive table plane (n, d) + the table's
      extent in table coordinates (u/v ranges), which fixes the partition
      bounds so cell indices mean the same thing in every recording.

Every later script consumes these instead of recomputing.

Usage:  python3 build_reference.py [Data/extracted/reference/depth]
"""
import glob
import os
import sys

import numpy as np

import config
from geometry import TableFrame, backproject, load_masked, ransac_plane


def main(ref_dir):
    files = sorted(glob.glob(os.path.join(ref_dir, "*.npy")))
    if not files:
        raise SystemExit(f"no .npy frames found in {ref_dir}")
    print(f"building background model from {len(files)} reference frames...")

    # Stack in manageable chunks to bound memory (825 * 240*320 * 8B ~ 0.5GB
    # would be fine, but stay frugal and future-proof).
    sample = load_masked(files[0])
    stack = np.zeros((len(files),) + sample.shape, dtype=np.uint16)
    for k, f in enumerate(files):
        stack[k] = load_masked(f).astype(np.uint16)
        if (k + 1) % 200 == 0:
            print(f"  ...{k + 1}/{len(files)}")

    # Median over frames; pixels invalid in >half the frames stay 0.
    # np.median over uint16 with zeros: mask zeros as NaN first.
    stackf = stack.astype(float)
    stackf[stackf == 0] = np.nan
    with np.errstate(all="ignore"):
        bg = np.nanmedian(stackf, axis=0)
    bg = np.nan_to_num(bg, nan=0.0)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    np.save(os.path.join(config.RESULTS_DIR, "background_median.npy"), bg)
    print("saved results/background_median.npy")

    # Definitive table plane from the background model itself (objects: none).
    pts, _ = backproject(bg)
    n, d, inl = ransac_plane(pts)
    frame = TableFrame(n, d)
    uvh = frame.to_table(pts[inl])
    u_range = (float(uvh[:, 0].min()), float(uvh[:, 0].max()))
    v_range = (float(uvh[:, 1].min()), float(uvh[:, 1].max()))
    np.savez(os.path.join(config.RESULTS_DIR, "table_plane.npz"),
             n=n, d=d, u_range=u_range, v_range=v_range,
             origin=frame.origin, e1=frame.e1, e2=frame.e2)
    print(f"saved results/table_plane.npz")
    print(f"  plane normal: {np.round(n, 4)}   d = {d:.1f} mm")
    print(f"  plane inliers: {int(inl.sum())}/{len(pts)} points")
    print(f"  table extent: u {u_range[0]:.0f}..{u_range[1]:.0f} mm, "
          f"v {v_range[0]:.0f}..{v_range[1]:.0f} mm")
    span_u = u_range[1] - u_range[0]
    span_v = v_range[1] - v_range[0]
    h = config.H_BASELINE
    print(f"  baseline partition (h={h:.0f} mm): "
          f"{int(np.ceil(span_u/h))} x {int(np.ceil(span_v/h))} cells")


if __name__ == "__main__":
    ref = sys.argv[1] if len(sys.argv) > 1 else "Data/extracted/reference/depth"
    main(ref)
