"""
merge_split.py — Shape-aware, prediction-guided splitting of merged clusters.

Problem: when two tracked objects touch, their occupied cells form ONE
connected component in the partition -> one detection -> one track starves
and identity breaks after TRACK_DEATH_FRAMES.

Fix (purely geometric, per project commitment): if a cluster overlaps the
predicted positions of MULTIPLE confirmed tracks, divide its POINTS between
those tracks by distance to each track's predicted BODY (not center):

  * ball body: circle of the track's known radius at the predicted center
    -> distance = |dist_to_center - radius|  (0 inside the shell)
  * box body: the track's known footprint (L x W oriented rectangle is
    approximated by its circumscribed radius for robustness; points inside
    that radius have distance 0 to the body, else distance beyond it)

Distance-to-BODY (not center) matters because the box is large: its far
cells sit closer to the ball's center POINT than to the box's own center,
so a naive Voronoi split steals them. Body distance assigns them correctly.

Each sub-cluster is then re-classified and reported as its own detection,
so both Kalman filters receive measurements through the contact and no
identity is lost.
"""
import numpy as np

import config


def body_distance(uv_pts, track_kind, pred_uv, size_mm):
    """Distance (mm) from table points to a track's predicted BODY.

    size_mm: characteristic body radius for the track's class —
      ball: fitted sphere radius; box: half-diagonal of the footprint.
    Inside the body -> 0.
    """
    d_center = np.linalg.norm(uv_pts - pred_uv, axis=1)
    return np.maximum(d_center - size_mm, 0.0)


def split_merged(uv_pts, pts3d, tracks_info):
    """Divide one merged cluster's points among >=2 predicted tracks.

    tracks_info: list of dicts with keys
        uv    — predicted (u, v) of the track this frame
        kind  — 'ball' / 'box' / 'unknown'
        size  — body radius in mm (see body_distance)
    Returns: list of (points3d_subset, uv_subset) per track, same order.
    Points are assigned to the track with the smallest body distance.
    Empty assignments return empty arrays (that track gets no measurement
    this frame and coasts — correct behavior if it truly left the blob).
    """
    D = np.stack([body_distance(uv_pts[:, :2], t["kind"],
                                np.asarray(t["uv"], float),
                                float(t["size"]))
                  for t in tracks_info])          # (n_tracks, n_points)
    owner = np.argmin(D, axis=0)
    out = []
    for ti in range(len(tracks_info)):
        sel = owner == ti
        out.append((pts3d[sel], uv_pts[sel]))
    return out


def default_body_radius(kind, ball_radius_mm=34.2,
                        box_lw_mm=(144.6, 62.4)):
    """Characteristic body radius per class, from the measured statics.

    ball: its fitted radius. box: half-diagonal of the L x W footprint
    (circumscribed circle — deliberately generous so real box points are
    never pushed out of their own body).
    """
    if kind == "ball":
        return ball_radius_mm
    L, W = box_lw_mm
    return 0.5 * float(np.hypot(L, W))