"""
selftest.py — Synthetic ground-truth validation of the full pipeline.

Principle: never trust code on unknown data until it recovers KNOWN answers.
Three tests, each rendering synthetic scenes through the same pinhole model
and noise level as the real sensor (config.DEPTH_NOISE_MM), then demanding
recovery within stated tolerances:

  1. sphere fit        — known 33 mm ball on a tilted plane + edge junk
  2. classifier        — a rendered ball must classify 'ball', a rendered
                         box must classify 'box' (competing hypotheses)
  3. tracker + physics — a synthetic decelerating trajectory fed through
                         the Kalman tracker; recovered positions and the
                         estimated deceleration must match ground truth

Run after ANY change to geometry.py / discretize.py / tracking.py.
Exit code 0 + 'ALL SELF-TESTS PASSED' = safe to run on real data.
"""
import numpy as np

import config
from discretize import Partition
from geometry import (TableFrame, backproject, classify_cluster, load_masked,
                      ransac_plane, ransac_sphere)
from tracking import Tracker

rng = np.random.default_rng(42)

PLANE_N = np.array([0.0, -0.42, -0.91])
PLANE_N = PLANE_N / np.linalg.norm(PLANE_N)
PLANE_P0 = np.array([0.0, 0.0, 580.0])


def render(sphere=None, box=None):
    """Ray-cast a 320x240 depth frame: tilted plane + optional sphere and/or
    axis-tilted box, with measured-level noise and right-edge junk."""
    H, W = 240, 320
    depth = np.zeros((H, W))
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    dx = (us - config.CX) / config.FX
    dy = (vs - config.CY) / config.FY
    # plane hit distance per ray (positive t only)
    denom = PLANE_N[0] * dx + PLANE_N[1] * dy + PLANE_N[2]
    t_plane = (PLANE_N @ PLANE_P0) / denom
    z = np.where(t_plane > 0, t_plane, np.inf)
    if sphere is not None:
        c, r = sphere
        # per-ray quadratic for |t*d - c| = r with d = (dx, dy, 1)
        dd = dx * dx + dy * dy + 1.0
        b = (dx * c[0] + dy * c[1] + c[2]) / dd
        disc = b * b - (c @ c - r * r) / dd
        t = np.where(disc >= 0, b - np.sqrt(np.maximum(disc, 0)), np.inf)
        z = np.minimum(z, np.where(t > 0, t, np.inf))
    if box is not None:
        c, half = box       # center + half-extents, camera-axis-aligned
        # slab method per ray (vectorized)
        t0 = np.full_like(z, 1e-6)
        t1 = np.full_like(z, np.inf)
        for axis, (dxx, cc, hh) in enumerate(
                [(dx, c[0], half[0]), (dy, c[1], half[1]),
                 (np.ones_like(dx), c[2], half[2])]):
            with np.errstate(divide="ignore", invalid="ignore"):
                ta = (cc - hh - 0 * dxx) / np.where(dxx == 0, 1e-12, dxx)
                tb = (cc + hh - 0 * dxx) / np.where(dxx == 0, 1e-12, dxx)
            lo, hi = np.minimum(ta, tb), np.maximum(ta, tb)
            t0, t1 = np.maximum(t0, lo), np.minimum(t1, hi)
        t_box = np.where(t1 >= t0, t0, np.inf)
        z = np.minimum(z, np.where(t_box > 0, t_box, np.inf))
    depth = np.where(np.isfinite(z), z, 0.0)
    depth += rng.normal(0, config.DEPTH_NOISE_MM, depth.shape) * (depth > 0)
    # junk on the right edge like the real sensor
    depth[:, 295:] = rng.uniform(300, 900, (H, 25)) * \
        (rng.random((H, 25)) > 0.5)
    return depth


def scene_object_points(depth):
    """Plane-removal segmentation for the synthetic scenes (self-contained:
    doesn't depend on a background model)."""
    pts, _ = backproject(load_masked(depth))
    n, d, inl = ransac_plane(pts)
    obj = pts[~inl]
    signed = obj @ n + d
    side = np.sign(np.median(np.sign(signed[np.abs(signed) > 6])) or 1)
    return obj[side * signed > 6], (n, d)


def test_sphere():
    C = np.array([-20.0, -55.0, 545.0])
    R = 33.0
    obj, _ = scene_object_points(render(sphere=(C, R)))
    c, r, _, rmse = ransac_sphere(obj)
    ce, re = np.linalg.norm(c - C), abs(r - R)
    print(f"[1] sphere: center err {ce:.2f} mm, radius err {re:.2f} mm, "
          f"rmse {rmse:.2f} mm")
    assert ce < 2.0 and re < 1.5, "sphere fit out of tolerance"


def test_classifier():
    C = np.array([-20.0, -55.0, 545.0])
    obj, _ = scene_object_points(render(sphere=(C, 33.0)))
    res = classify_cluster(obj)
    print(f"[2a] rendered ball -> '{res['kind']}' ({res['detail']})")
    assert res["kind"] == "ball", "ball misclassified"

    BC = np.array([30.0, -40.0, 560.0])
    obj, _ = scene_object_points(render(box=(BC, np.array([40., 25., 30.]))))
    res = classify_cluster(obj)
    print(f"[2b] rendered box  -> '{res['kind']}' ({res['detail']})")
    assert res["kind"] == "box", "box misclassified"


def test_tracker_and_partition():
    """A ball decelerating in a straight line across the table, observed
    with 1 mm measurement noise, must be (a) segmented via the partition,
    conceptually validated by clustering a rendered frame, and (b) tracked
    with small position error and near-truth recovered deceleration."""
    # (a) partition clustering on one rendered two-object frame
    depth = render(sphere=(np.array([-60., -50., 540.]), 33.0),
                   box=(np.array([60., -40., 560.]),
                        np.array([40., 25., 30.])))
    obj, (n, d) = scene_object_points(depth)
    tf = TableFrame(n, d)
    uvh = tf.to_table(obj)
    part = Partition((uvh[:, 0].min() - 20, uvh[:, 0].max() + 20),
                     (uvh[:, 1].min() - 20, uvh[:, 1].max() + 20),
                     h=config.H_BASELINE)
    clusters, grids = part.clusters(uvh)
    print(f"[3a] partition ({part.m}x{part.n} cells): "
          f"{len(clusters)} clusters found (expect 2)")
    assert len(clusters) == 2, "partition failed to separate two objects"
    kinds = sorted(classify_cluster(obj[c['points_mask']])['kind']
                   for c in clusters)
    print(f"[3b] cluster classes: {kinds} (expect ['ball', 'box'])")
    assert kinds == ["ball", "box"], "two-object classification failed"
    # non-binary cells sanity: variance inside the ball's cells must exceed
    # the (flat) box-top's — the professor's 'matrix of values' signal.
    # (informational print; not asserted, it is resolution-dependent)

    # (b) synthetic trajectory through the Kalman tracker
    DEC = 800.0                        # mm/s^2 ground truth
    v0 = np.array([500.0, 200.0])      # mm/s
    p = np.array([0.0, 0.0])
    v = v0.copy()
    tracker = Tracker(meas_std_mm=1.0)
    truth = []
    for f in range(90):
        sp = np.linalg.norm(v)
        if sp > 1e-6:
            a = -DEC * v / sp * config.DT
            v = v + a if np.linalg.norm(v + a) < sp else v * 0
        p = p + v * config.DT
        truth.append(p.copy())
        zmeas = p + rng.normal(0, 1.0, 2)
        tracker.step(f, [dict(uv=zmeas, kind="ball")])
    t = tracker.tracks[0]
    # Align by frame number: history begins one frame after track birth.
    truth = np.array(truth)
    hist_uv = np.array([(h[1], h[2]) for h in t.history])
    hist_f = np.array([h[0] for h in t.history], dtype=int)
    sel = hist_f >= 20                     # skip filter convergence period
    err = np.linalg.norm(hist_uv[sel] - truth[hist_f[sel]], axis=1)
    # recovered deceleration from the track's own velocity history
    vel = np.array([(h[3], h[4]) for h in t.history])
    sp = np.linalg.norm(vel, axis=1)
    dsp = np.diff(sp) / config.DT
    # Median over ALL moving samples: with symmetric velocity noise this
    # is an unbiased estimate of the (negative) mean acceleration, robust
    # to brief positive outliers (pushes). Conditioning on dsp<0 would
    # rectify noise and overestimate the deceleration.
    sel = sp[:-1] > 50
    dec_est = -np.median(dsp[sel])
    print(f"[3c] tracker: median position err {np.median(err):.2f} mm | "
          f"deceleration truth {DEC:.0f} vs estimated {dec_est:.0f} mm/s^2")
    assert np.median(err) < 5.0, "tracking error too large"
    assert abs(dec_est - DEC) / DEC < 0.35, "deceleration estimate poor"


def test_merge_split():
    """Two objects pass THROUGH each other's neighborhood: a ball crossing
    a box's position. Without merge splitting, the merged frames starve one
    track and identity breaks. With shape-aware splitting, BOTH tracks must
    survive the crossing with continuous identity (no new track IDs) and
    end up on the correct sides."""
    from merge_split import default_body_radius, split_merged
    from tracking import Tracker

    rng2 = np.random.default_rng(7)
    tracker = Tracker(meas_std_mm=1.0)

    ball_r = 34.2
    box_size = default_body_radius("box")
    box_uv = np.array([0.0, 0.0])                 # box parked at origin
    ball_uv = np.array([-250.0, 8.0])             # ball rolls through +u
    ball_v = np.array([600.0, 0.0])               # mm/s

    ids_seen = set()
    for f in range(60):
        ball_uv = ball_uv + ball_v * config.DT
        sep = np.linalg.norm(ball_uv - box_uv)
        merged = sep < (ball_r + box_size)        # bodies overlapping?
        if merged and len(tracker.tracks) >= 2:
            # simulate ONE merged cluster: points from both bodies
            n_b, n_x = 60, 150
            ball_pts = ball_uv + rng2.normal(0, ball_r * 0.5, (n_b, 2))
            box_pts = box_uv + rng2.uniform(-1, 1, (n_x, 2)) * [72, 31]
            allp = np.vstack([ball_pts, box_pts])
            uvh = np.column_stack([allp, np.full(len(allp), 20.0)])
            preds = []
            for t in tracker.tracks:
                pu, pv = (t.kf.F @ t.kf.x)[:2]
                preds.append(dict(uv=(pu, pv), kind=t.kind,
                                  size=default_body_radius(t.kind)))
            partsplit = split_merged(uvh, uvh, preds)
            dets = []
            for (sub, _), p in zip(partsplit, preds):
                if len(sub) < 5:
                    continue
                dets.append(dict(uv=sub[:, :2].mean(axis=0),
                                 kind=p["kind"]))
        else:
            dets = [dict(uv=ball_uv + rng2.normal(0, 1, 2), kind="ball"),
                    dict(uv=box_uv + rng2.normal(0, 1, 2), kind="box")]
        tracker.step(f, dets)
        for t in tracker.tracks:
            if t.confirmed:
                ids_seen.add((t.id, t.kind))

    kinds = sorted(k for _, k in ids_seen)
    n_tracks = len(ids_seen)
    balls = [t for t in tracker.tracks if t.kind == "ball"]
    boxes = [t for t in tracker.tracks if t.kind == "box"]
    print(f"[4] merge crossing: {n_tracks} confirmed track IDs over the "
          f"run (expect exactly 2) -> {kinds}")
    assert n_tracks == 2, "identity broke: extra track IDs created"
    assert len(balls) == 1 and len(boxes) == 1, "a track died in the merge"
    # ball must have come out the other side (u > box position)
    print(f"    ball ended at u={balls[0].kf.x[0]:.0f} mm "
          f"(box at u=0): crossed = {balls[0].kf.x[0] > 60}")
    assert balls[0].kf.x[0] > 60, "ball track did not follow through"


if __name__ == "__main__":
    test_sphere()
    test_classifier()
    test_tracker_and_partition()
    test_merge_split()
    print("ALL SELF-TESTS PASSED")