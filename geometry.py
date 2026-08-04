"""
geometry.py — Geometric core: back-projection, robust model fitting,
shape classification, and the table coordinate frame.

Design commitments (agreed with the professor):
  * Purely geometric. No learned components anywhere in this module.
  * Classification is by COMPETING HYPOTHESES: fit both a sphere and a
    plane-set (box) model to a cluster and compare residuals. Thresholds
    appear only where principled and are defined in config.py.
  * 'unknown' is a first-class outcome, not an error: if neither model
    wins by a clear margin the cluster is honestly ambiguous.
"""
import numpy as np

import config


# ===========================================================================
# Frame loading / back-projection
# ===========================================================================
def load_masked(path_or_array):
    """Load a depth frame (uint16 mm) and apply the ROI as a MASK.

    Pixels outside the ROI are set to 0, which downstream code treats
    identically to sensor-invalid pixels. Coordinates are never altered,
    so the intrinsics in config.py remain valid (mask, don't crop).
    """
    d = (np.load(path_or_array) if isinstance(path_or_array, str)
         else path_or_array).astype(float)
    m = np.zeros_like(d)
    r = config.ROI
    m[r["y1"]:r["y2"], r["x1"]:r["x2"]] = 1.0
    return d * m


def backproject(depth_mm):
    """Depth image -> (N,3) point cloud in mm, camera coordinates.

    Pinhole model: a pixel (u, v) with depth z maps to
        x = (u - cx) * z / fx,   y = (v - cy) * z / fy.
    Only valid (z > 0) pixels produce points.
    Returns (points, pixel_indices) where pixel_indices = (v_array, u_array)
    so callers can map 3D points back to their source pixels if needed.
    """
    v, u = np.nonzero(depth_mm > 0)
    z = depth_mm[v, u]
    x = (u - config.CX) * z / config.FX
    y = (v - config.CY) * z / config.FY
    return np.column_stack([x, y, z]), (v, u)


# ===========================================================================
# Robust plane fitting (RANSAC + least-squares refinement)
# ===========================================================================
def ransac_plane(pts, n_iter=300, thresh_mm=None, rng=None):
    """Fit the dominant plane in a point cloud.

    Returns (n, d, inlier_mask) with the plane defined by n.p + d = 0,
    n a unit normal. RANSAC makes the fit robust to the objects sitting on
    the table (they are outliers to the plane); an SVD refit on inliers
    then averages pixel noise down.
    """
    thresh_mm = thresh_mm or config.PLANE_THRESH_MM
    rng = rng or np.random.default_rng(0)
    N = len(pts)
    best_count, best_inl = -1, None
    for _ in range(n_iter):
        i = rng.choice(N, 3, replace=False)
        p0, p1, p2 = pts[i]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-9:                    # degenerate (collinear) sample
            continue
        n /= norm
        inl = np.abs((pts - p0) @ n) < thresh_mm
        c = int(inl.sum())
        if c > best_count:
            best_count, best_inl = c, inl
    # Least-squares refinement: plane through inlier centroid, normal =
    # smallest singular vector of the centered inliers.
    P = pts[best_inl]
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[-1]
    d = -float(n @ c)
    # SIGN CONVENTION (bug fix from real data): RANSAC/SVD return the
    # normal with arbitrary sign. We enforce that the normal points TOWARD
    # the camera (origin): the camera images the table's top surface, so
    # 'above the table' must mean 'toward the camera'. Without this, height
    # above the plane comes out negated on roughly half of datasets
    # (symptom: a box fitted with height = -10 mm).
    if d < 0:                       # origin on negative side -> flip
        n, d = -n, -d
    inl = np.abs(pts @ n + d) < thresh_mm
    return n, d, inl


# ===========================================================================
# Sphere fitting
# ===========================================================================
def sphere_ls(pts):
    """Algebraic least-squares sphere fit (exact linear formulation).

    Solves |p - c|^2 = r^2 rewritten as a linear system in (c, r^2 - |c|^2).
    Fast and adequate as a RANSAC inner solver and final refiner.
    """
    A = np.column_stack([2.0 * pts, np.ones(len(pts))])
    b = (pts ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r = float(np.sqrt(max(sol[3] + c @ c, 0.0)))
    return c, r


def ransac_sphere(pts, n_iter=400, thresh_mm=None, rng=None):
    """Robust sphere fit. Returns (center, radius, inlier_mask, rmse).

    RANSAC exists here to reject 'flying pixels' — the boundary artifacts
    measured on this sensor (rim pixels flip by tens of mm between frames).
    A radius sanity window rejects degenerate giant-sphere fits (a plane is
    a sphere of infinite radius; without the window RANSAC happily 'fits'
    residual table points that way).
    """
    thresh_mm = thresh_mm or config.SPHERE_THRESH_MM
    rng = rng or np.random.default_rng(1)
    N = len(pts)
    if N < 8:
        raise ValueError("too few points for a sphere fit")
    best_count, best_inl = -1, None
    for _ in range(n_iter):
        i = rng.choice(N, 4, replace=False)
        try:
            c, r = sphere_ls(pts[i])
        except np.linalg.LinAlgError:
            continue
        if not (5.0 < r < 500.0):          # 1 cm .. 1 m object diameter
            continue
        inl = np.abs(np.linalg.norm(pts - c, axis=1) - r) < thresh_mm
        k = int(inl.sum())
        if k > best_count:
            best_count, best_inl = k, inl
    if best_inl is None or best_inl.sum() < 8:
        raise ValueError("sphere RANSAC found no acceptable model")
    c, r = sphere_ls(pts[best_inl])
    inl = np.abs(np.linalg.norm(pts - c, axis=1) - r) < thresh_mm
    c, r = sphere_ls(pts[inl])             # second refinement pass
    resid = np.linalg.norm(pts[inl] - c, axis=1) - r
    rmse = float(np.sqrt((resid ** 2).mean()))
    return c, r, inl, rmse


# ===========================================================================
# Plane-set ("box surface") fitting
# ===========================================================================
def planes_fit(pts, max_planes=3, thresh_mm=None, min_frac=0.15, rng=None):
    """Fit up to `max_planes` planes sequentially (RANSAC each, remove
    inliers, repeat). Returns (planes, rmse) where rmse is computed over all
    points assigned to some plane, distance to their own plane.

    Rationale: a depth camera sees 1-3 faces of a box; box-ness is
    'points lie on a small set of planes'. This is the box hypothesis in the
    competing-model classifier. We do not require face orthogonality — that
    would be a refinement, and the residual comparison works without it.
    """
    thresh_mm = thresh_mm or config.SPHERE_THRESH_MM   # same band => fair fight
    rng = rng or np.random.default_rng(2)
    remaining = pts.copy()
    assigned_resid = []
    planes = []
    for _ in range(max_planes):
        if len(remaining) < max(20, min_frac * len(pts)):
            break
        n, d, inl = ransac_plane(remaining, n_iter=200,
                                 thresh_mm=thresh_mm, rng=rng)
        if inl.sum() < max(20, min_frac * len(pts)):
            break
        planes.append((n, d))
        assigned_resid.append(np.abs(remaining[inl] @ n + d))
        remaining = remaining[~inl]
    if not planes:
        raise ValueError("no planar structure found")
    resid = np.concatenate(assigned_resid)
    coverage = resid.size / len(pts)       # fraction of points explained
    rmse = float(np.sqrt((resid ** 2).mean()))
    # Penalize low coverage: unexplained points count at the threshold value,
    # so a model that ignores half the cluster cannot win on cherry-picked
    # residuals alone.
    n_unassigned = len(pts) - resid.size
    if n_unassigned:
        pen = np.full(n_unassigned, thresh_mm)
        rmse = float(np.sqrt(np.concatenate([resid, pen]) ** 2).mean())
    return planes, rmse, coverage


# ===========================================================================
# Competing-hypotheses classifier
# ===========================================================================
def classify_cluster(pts, thresh_mm=None, min_points=None):
    """Decide ball / box / unknown for one cluster of 3D points.

    Method (purely geometric, per project commitment):
      1. Fit the sphere hypothesis  -> rmse_sphere
      2. Fit the plane-set (box) hypothesis -> rmse_box
      3. The winner must beat the loser by CLASS_MARGIN (rmse ratio),
         else the cluster is honestly 'unknown'.

    `thresh_mm` / `min_points` default to config values for full-resolution
    clusters; sweep.py overrides them for the DEGRADED (cell-level) path,
    where residuals legitimately grow with the quantization scale h and a
    coarse object may consist of only a handful of cells.

    Returns dict(kind, center, size, rmse, detail) where
      * ball: size = radius, center = sphere center
      * box:  size = None here (dimensions estimated elsewhere),
              center = cluster centroid
      * unknown: centroid + both rmses for the record.
    """
    min_points = min_points or config.MIN_CLUSTER_POINTS
    if len(pts) < min_points:
        return dict(kind="unknown", center=pts.mean(axis=0), size=None,
                    rmse=None, detail="too few points")

    rmse_s = rmse_b = np.inf
    sphere = None
    try:
        c_s, r_s, _, rmse_s = ransac_sphere(pts, thresh_mm=thresh_mm)
        # Shallow-cap rejection: if the fitted radius exceeds the cluster's
        # own spatial extent, the visible surface is a nearly flat cap —
        # geometrically indistinguishable from a plane — so it must not
        # count as sphere evidence. (Without this, a coarsely quantized
        # flat box top fits a huge-radius sphere and misclassifies.)
        extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        if r_s <= extent:
            sphere = (c_s, r_s)
        else:
            rmse_s = np.inf
    except (ValueError, np.linalg.LinAlgError):
        pass
    try:
        _, rmse_b, _ = planes_fit(pts, thresh_mm=thresh_mm)
    except (ValueError, np.linalg.LinAlgError):
        pass

    if np.isinf(rmse_s) and np.isinf(rmse_b):
        return dict(kind="unknown", center=pts.mean(axis=0), size=None,
                    rmse=None, detail="both fits failed")
    if rmse_s * config.CLASS_MARGIN < rmse_b:
        c_s, r_s = sphere
        return dict(kind="ball", center=c_s, size=r_s, rmse=rmse_s,
                    detail=f"sphere {rmse_s:.2f} vs box {rmse_b:.2f}")
    if rmse_b * config.CLASS_MARGIN < rmse_s:
        return dict(kind="box", center=pts.mean(axis=0), size=None,
                    rmse=rmse_b,
                    detail=f"box {rmse_b:.2f} vs sphere {rmse_s:.2f}")
    return dict(kind="unknown", center=pts.mean(axis=0), size=None,
                rmse=min(rmse_s, rmse_b),
                detail=f"margin too small ({rmse_s:.2f} vs {rmse_b:.2f})")


# ===========================================================================
# Table coordinate frame
# ===========================================================================
class TableFrame:
    """Orthonormal 2D coordinate system on the fitted table plane.

    Motion physically happens ON the table, so tracking/prediction run in
    (u, v) table coordinates; height above the plane is retained as a
    physics consistency check (e.g. a rolling ball's center must sit one
    radius above the plane).
    """

    def __init__(self, n, d, origin=None):
        n = np.asarray(n, float)
        self.n = n / np.linalg.norm(n)
        self.d = float(d)
        # Origin: the plane point closest to the camera origin by default.
        self.origin = (origin if origin is not None
                       else -self.d * self.n)
        # Build in-plane axes: e1 = any direction orthogonal to n.
        a = np.array([1.0, 0.0, 0.0])
        if abs(self.n @ a) > 0.9:
            a = np.array([0.0, 1.0, 0.0])
        self.e1 = np.cross(self.n, a)
        self.e1 /= np.linalg.norm(self.e1)
        self.e2 = np.cross(self.n, self.e1)

    def to_table(self, pts):
        """Points (N,3) camera coords -> (N,3): (u, v, height above plane)."""
        pts = np.atleast_2d(pts)
        rel = pts - self.origin
        u = rel @ self.e1
        v = rel @ self.e2
        h = pts @ self.n + self.d
        return np.column_stack([u, v, h])

    def to_camera(self, uvh):
        """Inverse of to_table."""
        uvh = np.atleast_2d(uvh)
        return (self.origin
                + uvh[:, :1] * self.e1
                + uvh[:, 1:2] * self.e2
                + (uvh[:, 2:3] - self.d - self.origin @ self.n) * self.n
                + (self.origin @ self.n) * 0)  # origin lies on plane: h0 = 0

    def as_dict(self):
        return dict(n=self.n, d=self.d, origin=self.origin,
                    e1=self.e1, e2=self.e2)