"""
make_dataset2.py — Multi-frame trajectory dataset builder.

Companion to make_dataset.py, but for the professor's follow-up question:
does using a WINDOW of the k most recent frames (instead of a single
current frame) help predict where the object goes next?

make_dataset.py:  one frame       -> one patch  -> class label (ball/box)
make_dataset2.py: k recent frames -> one window  -> future (u, v) position

Source data: the SAME tracked positions predict.py already uses
(results/<recording>_tracks.csv, produced by track_moving.py). No new
recordings needed — this reuses your existing tracking output.

Window features (explainable, matching the project's design commitment —
no raw-window black box):
  u, v            current position (mm, last frame in the window)
  du, dv          current velocity (mm/s, last frame in the window,
                     already smoothed by the Kalman filter)
  speed           current speed magnitude (mm/s)
  avg_du, avg_dv  average velocity across the whole window
  ddu, ddv        velocity CHANGE across the window (acceleration proxy):
                     (velocity at window end - velocity at window start)
                     / window time span

Target: displacement (delta_u, delta_v) = position at (t + HORIZON) minus
position at t. Predicting a displacement, not an absolute position, keeps
the regression well-scaled regardless of where on the table the window
sits (same reasoning as predicting velocity/acceleration in predict.py).

Rules kept from make_dataset.py:
  - split BY TIME (never random) — first TRAIN_FRAC of each recording's
    frames -> train, rest -> test
  - a window must never cross a track boundary (no window spans a
    track's death and a different track's birth — that would look like a
    teleport, not real motion)
  - MIN_SPEED filtering matches predict.py: parked stretches carry no
    motion information and would flatter every model equally

Output: results/traj_dataset_k{K}_h{HORIZON}.npz with arrays
  X_train/X_test        (N, 9) float32   engineered window features
  Y_train/Y_test         (N, 2) float32   target displacement (du, dv) mm
  U_train/U_test          (N, 2) float32   current (u, v) — to reconstruct
                                            absolute predicted position
  meta_train/meta_test    (N,)   strings   "recording:track_id:frame"

Usage: python3 make_dataset2.py                  (K=5, HORIZON=10)
       python3 make_dataset2.py --k 5 --h 10
"""
import csv
import os
import sys

import numpy as np

import config

RECORDINGS = ["ball_moving", "box_moving", "ball_and_box_moving"]
TRAIN_FRAC = 0.70      # same time-split convention as make_dataset.py
MIN_SPEED = 30.0       # mm/s — same "parked" cutoff as predict.py


def load_tracks(name):
    """Same loader as predict.py: one array per track, sorted by frame."""
    path = os.path.join(config.RESULTS_DIR, f"{name}_tracks.csv")
    rows = list(csv.DictReader(open(path)))
    tracks = {}
    for r in rows:
        tracks.setdefault(int(r["track_id"]), []).append(r)
    out = []
    for tid, rs in tracks.items():
        rs.sort(key=lambda r: int(r["frame"]))
        arr = np.array([[int(r["frame"]), float(r["u_mm"]), float(r["v_mm"]),
                         float(r["du_mm_s"]), float(r["dv_mm_s"])]
                        for r in rs])
        out.append(dict(id=tid, arr=arr, recording=name))
    return out


def window_features(a, i, k):
    """Build the engineered feature vector for the window ending at row i
    (i.e. rows i-k+1 .. i), from a track's (frame, u, v, du, dv) array."""
    win = a[i - k + 1:i + 1]
    f0, u0, v0, du0, dv0 = win[0]
    f1, u1, v1, du1, dv1 = win[-1]
    dt_span = (f1 - f0) * config.DT
    if dt_span < 1e-9:
        dt_span = config.DT
    avg_du = win[:, 3].mean()
    avg_dv = win[:, 4].mean()
    ddu = (du1 - du0) / dt_span
    ddv = (dv1 - dv0) / dt_span
    speed = float(np.hypot(du1, dv1))
    return np.array([u1, v1, du1, dv1, speed, avg_du, avg_dv, ddu, ddv],
                    dtype=np.float32)


FEATURE_NAMES = ["u", "v", "du", "dv", "speed",
                 "avg_du", "avg_dv", "ddu", "ddv"]


def build_examples(tracks, k, horizon):
    """Slide a k-frame window over every track; one example per valid
    window end-point. Window and target must stay within the SAME track
    (never cross a track boundary — see module docstring)."""
    X, Y, U, meta, frame_idx = [], [], [], [], []
    for t in tracks:
        a = t["arr"]
        frames = a[:, 0].astype(int)
        index = {f: i for i, f in enumerate(frames)}
        n = len(a)
        for i in range(k - 1, n):
            f_now = int(a[i, 0])
            # window must be CONSECUTIVE frames within this track (no gaps
            # from missed detections — a gap would silently compress real
            # elapsed time into the window without the model knowing).
            if int(a[i, 0]) - int(a[i - k + 1, 0]) != k - 1:
                continue
            j = index.get(f_now + horizon)
            if j is None:
                continue                      # future truth unavailable
            du_now, dv_now = a[i, 3], a[i, 4]
            if np.hypot(du_now, dv_now) < MIN_SPEED:
                continue                      # parked — see MIN_SPEED
            feat = window_features(a, i, k)
            u_now, v_now = a[i, 1], a[i, 2]
            u_future, v_future = a[j, 1], a[j, 2]
            X.append(feat)
            Y.append([u_future - u_now, v_future - v_now])
            U.append([u_now, v_now])
            meta.append(f"{t['recording']}:{t['id']}:{f_now}")
            frame_idx.append(f_now)
    return X, Y, U, meta, frame_idx


def main(k=5, horizon=10):
    Xtr, Ytr, Utr, meta_tr = [], [], [], []
    Xte, Yte, Ute, meta_te = [], [], [], []

    for rec in RECORDINGS:
        path = os.path.join(config.RESULTS_DIR, f"{rec}_tracks.csv")
        if not os.path.exists(path):
            print(f"  skip {rec}: {path} not found")
            continue
        tracks = load_tracks(rec)
        for t in tracks:
            frames = t["arr"][:, 0]
            cut_frame = frames[0] + TRAIN_FRAC * (frames[-1] - frames[0])
            train_arr = t["arr"][t["arr"][:, 0] <= cut_frame]
            test_arr = t["arr"][t["arr"][:, 0] > cut_frame]
            if len(train_arr) >= k:
                X, Y, U, meta, _ = build_examples(
                    [dict(id=t["id"], arr=train_arr, recording=rec)],
                    k, horizon)
                Xtr += X; Ytr += Y; Utr += U; meta_tr += meta
            if len(test_arr) >= k:
                X, Y, U, meta, _ = build_examples(
                    [dict(id=t["id"], arr=test_arr, recording=rec)],
                    k, horizon)
                Xte += X; Yte += Y; Ute += U; meta_te += meta
        print(f"  {rec}: {len(tracks)} track(s) processed")

    out = os.path.join(config.RESULTS_DIR,
                       f"traj_dataset_k{k}_h{horizon}.npz")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    np.savez_compressed(
        out,
        X_train=np.array(Xtr, dtype=np.float32),
        Y_train=np.array(Ytr, dtype=np.float32),
        U_train=np.array(Utr, dtype=np.float32),
        meta_train=np.array(meta_tr),
        X_test=np.array(Xte, dtype=np.float32),
        Y_test=np.array(Yte, dtype=np.float32),
        U_test=np.array(Ute, dtype=np.float32),
        meta_test=np.array(meta_te),
        feature_names=np.array(FEATURE_NAMES),
        k=k, horizon=horizon)
    print(f"\nsaved {out}")
    print(f"  window k={k} frames, predicting +{horizon} frames ahead")
    print(f"  train windows: {len(Ytr)} | test windows: {len(Yte)}")


if __name__ == "__main__":
    k = (int(sys.argv[sys.argv.index("--k") + 1])
         if "--k" in sys.argv else 5)
    horizon = (int(sys.argv[sys.argv.index("--h") + 1])
              if "--h" in sys.argv else 10)
    main(k, horizon)