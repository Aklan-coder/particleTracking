"""
heightfit.py — fair-metric classification for cell-level (radar-like) data.

THE PROBLEM THIS FIXES
----------------------
In the degraded (cell) view, each occupied cell contributes one point
(u_center, v_center, mean_height). The two coordinates have wildly
different error bars:
  * u, v are QUANTIZED: the true surface patch lies anywhere in the cell,
    so cell centers carry up to h/2 of positional error.
  * mean height is ACCURATE (~1 mm): averaging hundreds of pixels crushes
    the 0.77 mm sensor noise.
The original classifier compares a 3D radial sphere residual (which eats
the full u,v quantization error) against near-horizontal plane residuals
(which are essentially height-only, i.e. nearly noise-free). Planes are
judged on the clean axis, spheres on the dirty one — so at coarse h a
ball's dome loses to a piecewise-planar fit even when its height profile
is clearly curved.

THE FIX
-------
Score BOTH hypotheses on the same axis — the accurate one: "given the
known beam position (cell center), what height does the fitted model
predict, and how far is the measured mean height from that?" This is
exactly the physical measurement an m x n beam array returns: height per
beam. Every occupied cell is scored (no inlier cherry-picking; a model
that cannot explain a cell pays for it), residuals are capped at h so a
single rim cell cannot dominate, and the existing CLASS_MARGIN decision
rule is kept so the change is the METRIC only, not the decision logic.

Pure geometry, no catalog dimensions — the fair-metric counterpart to the
original classify_cluster, and the uncon­strained baseline against
identify.classify_smart (which adds known-catalog constraints).
"""
import numpy as np

import config
from geometry import ransac_sphere, planes_fit

CAP_FRAC = getattr(config, "SPHERE_MIN_CAP_FRAC", 0.25)
VERT_NZ = 0.2      # |n_h| below this => near-vertical plane; use 3D distance


def _sphere_height_residuals(cell_pts, c, r, cap):
    """Vertical mismatch between measured cell heights and the height of
    the fitted sphere's TOP surface at each cell center. Cells outside the
    sphere's footprint (where the model predicts no surface) pay `cap`."""
    du = cell_pts[:, 0] - c[0]
    dv = cell_pts[:, 1] - c[1]
    d2 = du * du + dv * dv
    res = np.full(len(cell_pts), cap, dtype=float)
    inside = d2 <= r * r
    if np.any(inside):
        h_pred = c[2] + np.sqrt(np.maximum(r * r - d2[inside], 0.0))
        res[inside] = np.abs(cell_pts[inside, 2] - h_pred)
    return np.minimum(res, cap)


def _planes_height_residuals(cell_pts, planes, cap):
    """Each cell is scored against its best plane. Non-vertical planes are
    scored by vertical (height) mismatch at the cell center; near-vertical
    planes (a box side face) predict no height, so their points are scored
    by perpendicular 3D distance instead."""
    N = len(cell_pts)
    best = np.full(N, cap, dtype=float)
    for (n, d) in planes:
        if abs(n[2]) >= VERT_NZ:
            h_pred = -(d + n[0] * cell_pts[:, 0] + n[1] * cell_pts[:, 1]) / n[2]
            r = np.abs(cell_pts[:, 2] - h_pred)
        else:
            r = np.abs(cell_pts @ n + d)
        best = np.minimum(best, r)
    return np.minimum(best, cap)


def classify_height(cell_pts, h):
    """Ball / box / unknown from cell-level points, both hypotheses scored
    on the height-consistency metric. Returns a dict compatible with
    classify_cluster's (kind, detail, sphere_rmse, box_rmse, margin)."""
    N = len(cell_pts)
    if N < 4:
        return dict(kind="unknown", detail="too few points",
                    sphere_rmse=np.inf, box_rmse=np.inf, margin=0.0)

    fit_band = max(config.SPHERE_THRESH_MM, h / 2.0)   # for RANSAC support
    cap = max(h, 10.0)                                  # residual cap (mm)

    # ---- sphere hypothesis: 3D fit (for the model), height-scored -------
    rmse_s = np.inf
    reason = ""
    try:
        c_s, r_s, _, _ = ransac_sphere(cell_pts, thresh_mm=fit_band)
        extent = float(np.linalg.norm(cell_pts.max(axis=0) - cell_pts.min(axis=0)))
        c0 = cell_pts.mean(axis=0)
        _, _, vt = np.linalg.svd(cell_pts - c0, full_matrices=False)
        thickness = float(np.ptp((cell_pts - c0) @ vt[-1]))
        if r_s > extent:
            reason = "radius too large"
        elif thickness < CAP_FRAC * r_s:
            reason = "shallow cap"
        else:
            res = _sphere_height_residuals(cell_pts, c_s, r_s, cap)
            rmse_s = float(np.sqrt((res ** 2).mean()))
            reason = "ok"
    except (ValueError, np.linalg.LinAlgError) as e:
        reason = f"sphere fit failed: {type(e).__name__}"

    # ---- box hypothesis: plane-set fit, height-scored -------------------
    rmse_b = np.inf
    try:
        planes, _, _ = planes_fit(cell_pts, thresh_mm=fit_band)
        res = _planes_height_residuals(cell_pts, planes, cap)
        rmse_b = float(np.sqrt((res ** 2).mean()))
    except (ValueError, np.linalg.LinAlgError):
        pass

    margin = (abs(rmse_b - rmse_s)
              if np.isfinite(rmse_s) and np.isfinite(rmse_b) else 0.0)

    if np.isinf(rmse_s) and np.isinf(rmse_b):
        return dict(kind="unknown", detail="both fits failed",
                    sphere_rmse=rmse_s, box_rmse=rmse_b, margin=margin,
                    sphere_reason=reason)
    if rmse_s * config.CLASS_MARGIN < rmse_b:
        return dict(kind="ball", detail=f"sphere {rmse_s:.2f} vs box {rmse_b:.2f}",
                    sphere_rmse=rmse_s, box_rmse=rmse_b, margin=margin,
                    sphere_reason=reason)
    if rmse_b * config.CLASS_MARGIN < rmse_s:
        return dict(kind="box", detail=f"box {rmse_b:.2f} vs sphere {rmse_s:.2f}",
                    sphere_rmse=rmse_s, box_rmse=rmse_b, margin=margin,
                    sphere_reason=reason)
    return dict(kind="unknown", detail=f"margin too small ({rmse_s:.2f} vs {rmse_b:.2f})",
                sphere_rmse=rmse_s, box_rmse=rmse_b, margin=margin,
                sphere_reason=reason)