"""
tracking.py — Kalman filtering and multi-object track management.
"""
import numpy as np

import config


class Kalman2D:
    """Constant-velocity Kalman filter in table coordinates (mm, mm/s)."""

    def __init__(self, u, v, meas_std_mm):
        self.x = np.array([u, v, 0.0, 0.0])
        self.P = np.diag([meas_std_mm ** 2, meas_std_mm ** 2,
                          400.0 ** 2, 400.0 ** 2])
        dt = config.DT
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], float)
        sa = 1500.0
        q = sa ** 2
        self.Q = q * np.array(
            [[dt ** 4 / 4, 0, dt ** 3 / 2, 0],
             [0, dt ** 4 / 4, 0, dt ** 3 / 2],
             [dt ** 3 / 2, 0, dt ** 2, 0],
             [0, dt ** 3 / 2, 0, dt ** 2]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], float)
        self.R = np.eye(2) * meas_std_mm ** 2

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, z):
        z = np.asarray(z, float)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def gate_distance(self, z):
        S = self.H @ self.P @ self.H.T + self.R
        d = np.asarray(z, float) - self.H @ self.x
        return float(np.sqrt(d @ np.linalg.inv(S) @ d))


class Track:
    _next_id = 1

    def __init__(self, u, v, meas_std_mm, kind, frame):
        self.id = Track._next_id
        Track._next_id += 1
        self.kf = Kalman2D(u, v, meas_std_mm)
        self.votes = {"ball": 0, "box": 0, "unknown": 0}
        self.votes[kind] += 1
        self.born = frame
        self.hits = 1
        self.misses = 0
        self.confirmed = False
        self.history = []

    @property
    def kind(self):
        b, x = self.votes["ball"], self.votes["box"]
        if b == 0 and x == 0:
            return "unknown"
        return "ball" if b >= x else "box"

    def step_hit(self, frame, z, kind):
        self.kf.update(z)
        self.votes[kind] += 1
        self.hits += 1
        self.misses = 0
        if self.hits >= config.TRACK_BIRTH_FRAMES:
            self.confirmed = True
        u, v, du, dv = self.kf.x
        self.history.append((frame, u, v, du, dv, kind))

    def step_miss(self, frame):
        self.misses += 1
        u, v, du, dv = self.kf.x
        self.history.append((frame, u, v, du, dv, "missed"))

    @property
    def dead(self):
        return self.misses > config.TRACK_DEATH_FRAMES


class Tracker:
    def __init__(self, meas_std_mm):
        self.meas_std = meas_std_mm
        self.tracks = []

    def step(self, frame, detections):
        for t in self.tracks:
            t.kf.predict()

        pairs = []
        for ti, t in enumerate(self.tracks):
            for di, det in enumerate(detections):
                eucl = np.linalg.norm(t.kf.x[:2] - det["uv"])
                if eucl > config.GATE_MM:
                    continue
                pairs.append((t.kf.gate_distance(det["uv"]), ti, di))
        pairs.sort()

        used_t, used_d = set(), set()
        for dist, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            self.tracks[ti].step_hit(frame, detections[di]["uv"],
                                     detections[di]["kind"])
            used_t.add(ti)
            used_d.add(di)

        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.step_miss(frame)

        for di, det in enumerate(detections):
            if di not in used_d:
                self.tracks.append(Track(det["uv"][0], det["uv"][1],
                                         self.meas_std, det["kind"], frame))

        self.tracks = [t for t in self.tracks if not t.dead]
        return [t for t in self.tracks if t.confirmed]
