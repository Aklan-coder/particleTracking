"""
predict.py — Phase-1 headline experiment: shape-conditioned prediction vs a
shape-agnostic baseline, scored on the recorded tracks.

Three predictors, run side by side on every track (purely geometric —
friction coefficients are ESTIMATED FROM THE TRACKS by fitting observed
decelerations, which is parameter estimation from physics, not learning):

  cv       constant velocity (shape-blind null model)
  rolling  ball model: motion on the plane, small rolling-resistance
           deceleration opposing velocity
  sliding  box model: Coulomb friction — larger constant deceleration
           opposing velocity, velocity clamps to zero (boxes stop)

Both physics models are the same mathematical family (decelerate at rate a,
stop at v=0); the SHAPE decides which deceleration applies. Predictions at
+1/+5/+10 frames are scored against what actually happened.

Output: results/prediction_errors.csv, results/prediction_curves.png,
        results/prediction_summary.txt

Usage:  python3 predict.py ball_moving box_moving ball_and_box_moving
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

HORIZONS = [1, 5, 10]          # frames ahead
MIN_SPEED = 30.0               # mm/s: below this the object is 'parked';
                               # parked stretches carry no motion info and
                               # would flatter every predictor equally.


def load_tracks(name):
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
        kind = rs[len(rs) // 2]["track_kind"]   # stable majority label
        out.append(dict(id=tid, kind=kind, arr=arr, recording=name))
    return out


def estimate_deceleration(tracks, kind):
    """Median deceleration (mm/s^2) observed on moving segments of tracks of
    one kind — the friction parameter, estimated from the data itself."""
    decs = []
    for t in tracks:
        if t["kind"] != kind:
            continue
        a = t["arr"]
        sp = np.hypot(a[:, 3], a[:, 4])
        dsp = np.diff(sp) / config.DT
        # Median of speed-change over ALL moving samples (not only slowing
        # ones): with symmetric noise this is unbiased, and it is robust to
        # brief push events, which appear as positive outliers. Conditioning
        # on dsp < 0 would rectify noise and overestimate friction (bug
        # caught by selftest.py test 3c).
        sel = sp[:-1] > MIN_SPEED
        if sel.sum() > 5:
            decs.append(max(-np.median(dsp[sel]), 0.0))
    return float(np.median(decs)) if decs else 0.0


def predict_state(u, v, du, dv, k, decel):
    """Propagate k frames with constant deceleration `decel` opposing the
    velocity direction; velocity clamps at zero (object stops).
    decel = 0 reduces exactly to constant velocity."""
    dt = config.DT
    sp = float(np.hypot(du, dv))
    if sp < 1e-9:
        return u, v
    t = k * dt
    t_stop = sp / decel if decel > 0 else np.inf
    te = min(t, t_stop)
    dist = sp * te - 0.5 * decel * te ** 2
    return u + du / sp * dist, v + dv / sp * dist


def main(names):
    all_tracks = []
    for n in names:
        all_tracks += load_tracks(n)
    dec_ball = estimate_deceleration(all_tracks, "ball")
    dec_box = estimate_deceleration(all_tracks, "box")
    print(f"estimated decelerations: ball {dec_ball:.0f} mm/s^2 "
          f"(rolling), box {dec_box:.0f} mm/s^2 (sliding)")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out = open(os.path.join(config.RESULTS_DIR, "prediction_errors.csv"),
               "w", newline="")
    w = csv.writer(out)
    w.writerow(["recording", "track_id", "kind", "frame", "horizon",
                "err_cv_mm", "err_shape_mm"])

    errs = {}          # (kind, horizon, model) -> list of errors
    for t in all_tracks:
        a = t["arr"]
        decel = dec_ball if t["kind"] == "ball" else dec_box
        frames = a[:, 0].astype(int)
        index = {f: i for i, f in enumerate(frames)}
        for i in range(len(a)):
            f0, u, v, du, dv = a[i]
            if np.hypot(du, dv) < MIN_SPEED:
                continue                       # parked: skip (see MIN_SPEED)
            for k in HORIZONS:
                j = index.get(int(f0) + k)
                if j is None:
                    continue                   # truth unavailable (track gap)
                ut, vt = a[j, 1], a[j, 2]
                pu, pv = predict_state(u, v, du, dv, k, 0.0)
                e_cv = float(np.hypot(pu - ut, pv - vt))
                pu, pv = predict_state(u, v, du, dv, k, decel)
                e_sh = float(np.hypot(pu - ut, pv - vt))
                w.writerow([t["recording"], t["id"], t["kind"], int(f0), k,
                            f"{e_cv:.2f}", f"{e_sh:.2f}"])
                errs.setdefault((t["kind"], k, "cv"), []).append(e_cv)
                errs.setdefault((t["kind"], k, "shape"), []).append(e_sh)
    out.close()

    # ---- summary + figure ----
    lines = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, kind in zip(axes, ["ball", "box"]):
        st = config.STYLE[kind]
        for model, ls in [("cv", "--"), ("shape", "-")]:
            med = [np.median(errs.get((kind, k, model), [np.nan]))
                   for k in HORIZONS]
            ax.plot(HORIZONS, med, ls, marker=st["marker"],
                    color=st["color"] if model == "shape" else "grey",
                    label=("shape-conditioned" if model == "shape"
                           else "constant velocity"))
            lines.append(f"{kind:5s} {model:5s}: median err @ "
                         + ", ".join(f"+{k}f {m:.1f}mm"
                                     for k, m in zip(HORIZONS, med)))
        ax.set_title(f"{kind} ({st['label']})")
        ax.set_xlabel("prediction horizon (frames)")
        ax.legend()
    axes[0].set_ylabel("median position error (mm)")
    fig.suptitle("Shape-conditioned prediction vs constant-velocity baseline")
    fig.savefig(os.path.join(config.RESULTS_DIR, "prediction_curves.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(config.RESULTS_DIR,
                           "prediction_summary.txt"), "w") as fh:
        fh.write(f"decel ball {dec_ball:.0f} mm/s^2, "
                 f"box {dec_box:.0f} mm/s^2\n{summary}\n")
    print("saved results/prediction_errors.csv, prediction_curves.png, "
          "prediction_summary.txt")


if __name__ == "__main__":
    names = sys.argv[1:] or ["ball_moving", "box_moving",
                             "ball_and_box_moving"]
    main(names)
