"""
discretize.py — The professor's mutable m x n spatial partition.

Central design directive: the grid simulates a future radar array. Cell size
h is a free parameter ('mutable m x n'); the baseline uses config.H_BASELINE
and sweep.py varies h to locate the identification/tracking resolution
limits ("a grid of at least m x n cells is required to identify objects of
size k").

Second directive honored here: cells are NOT binary. Each occupied cell
stores a small matrix of values (point count, mean/max height above the
table, height variance) — the information a richer sensor would return and
the feature bank for the later (phase-2) learned model. Saving these costs
nothing now and avoids reprocessing everything later.
"""
import numpy as np
from scipy import ndimage

import config


class Partition:
    """An m x n grid over the table plane at cell size h (mm).

    Bounds are fixed once (from the reference recording's table extent) so
    that cell indices mean the same thing in every frame of every recording.
    """

    def __init__(self, u_range, v_range, h=None):
        self.h = float(h if h is not None else config.H_BASELINE)
        self.u0, self.u1 = map(float, u_range)
        self.v0, self.v1 = map(float, v_range)
        self.m = int(np.ceil((self.u1 - self.u0) / self.h))  # cells along u
        self.n = int(np.ceil((self.v1 - self.v0) / self.h))  # cells along v

    # -- mapping ----------------------------------------------------------
    def cell_of(self, uv):
        """Table (u, v) -> integer cell indices (i, j). May fall outside."""
        i = np.floor((uv[:, 0] - self.u0) / self.h).astype(int)
        j = np.floor((uv[:, 1] - self.v0) / self.h).astype(int)
        return i, j

    def center_of(self, i, j):
        """Cell indices -> (u, v) of the cell center."""
        return (self.u0 + (np.asarray(i) + 0.5) * self.h,
                self.v0 + (np.asarray(j) + 0.5) * self.h)

    # -- core operation ----------------------------------------------------
    def grids(self, uvh):
        """Points in table coords -> per-cell value matrices.

        Returns dict of (m, n) arrays:
          count    — points per cell (occupancy = count > 0)
          mean_h   — mean height above table of the cell's points
          max_h    — max height (a box face and a ball crown differ here)
          var_h    — height variance (curvature signature: 'cells are not
                     binary' — a sphere-occupied cell has spread, a flat
                     box face has almost none)
        """
        i, j = self.cell_of(uvh)
        ok = (i >= 0) & (i < self.m) & (j >= 0) & (j < self.n)
        i, j, h = i[ok], j[ok], uvh[ok, 2]
        flat = i * self.n + j
        size = self.m * self.n

        count = np.bincount(flat, minlength=size).astype(float)
        sum_h = np.bincount(flat, weights=h, minlength=size)
        sum_h2 = np.bincount(flat, weights=h * h, minlength=size)
        max_h = np.zeros(size)
        np.maximum.at(max_h, flat, h)

        with np.errstate(divide="ignore", invalid="ignore"):
            mean_h = np.where(count > 0, sum_h / count, 0.0)
            var_h = np.where(count > 0,
                             np.maximum(sum_h2 / count - mean_h ** 2, 0.0),
                             0.0)
        shape = (self.m, self.n)
        return dict(count=count.reshape(shape),
                    mean_h=mean_h.reshape(shape),
                    max_h=max_h.reshape(shape),
                    var_h=var_h.reshape(shape))

    # -- segmentation on the grid -------------------------------------------
    def clusters(self, uvh, min_cells=2):
        """Connected components of occupied cells -> point clusters.

        THE PARTITION IS THE SEGMENTATION (professor's framing): objects are
        groups of touching occupied cells; each cluster returns the ORIGINAL
        full-resolution points that landed in its cells, so downstream
        classification retains all geometric detail.
        8-connectivity so diagonal contact still joins cells.
        """
        g = self.grids(uvh)
        occupied = g["count"] > 0
        lab, k = ndimage.label(occupied, structure=np.ones((3, 3)))
        i, j = self.cell_of(uvh)
        ok = (i >= 0) & (i < self.m) & (j >= 0) & (j < self.n)
        point_label = np.zeros(len(uvh), dtype=int)
        point_label[ok] = lab[i[ok], j[ok]]
        out = []
        for lbl in range(1, k + 1):
            n_cells = int((lab == lbl).sum())
            if n_cells < min_cells:
                continue                    # single-cell speckle -> ignore
            out.append(dict(points_mask=point_label == lbl,
                            n_cells=n_cells, label=lbl))
        # Largest first: convenient for callers expecting main objects early.
        out.sort(key=lambda c: -c["n_cells"])
        return out, g
