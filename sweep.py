"""
sweep.py — THE EXPERIMENT: identification and detection quality vs partition
resolution.

For each cell size h in config.H_SWEEP, the full per-frame pipeline
(background subtraction -> partition at h -> cluster -> classify) is re-run
on sampled frames of the requested recordings, and per-h statistics are
collected.

What this sweep measures
------------------------
* identification rate: fraction of detections classified as the recording's
  known ground-truth object (ball in ball_*, box in box_*)
* ambiguity rate: fraction classified 'unknown' with both hypotheses tested
* sparse rate: fraction where the cluster had too few cells to test at all
  (< MIN_CELL_POINTS). Kept SEPARATE from ambiguity so the geometric limit
  is not confused with an administrative point-count floor.
* wrong rate: fraction classified as the wrong object kind
* detection rate: fraction of trials in which the object was found at all
* correct-per-frame: fraction of trials that were both detected and
  correctly identified
* cells_across: typical number of occupied cells spanning the object

GRID-PHASE AVERAGING (artifact fix)
-----------------------------------
The objects in the *_static recordings do not move, so for a single grid
placement every frame sees the SAME alignment between grid and object.
Whether a 68 mm ball at h = 60 lands in 1 cell (rejected as speckle,
detection = 0) or straddles 4 cells (detected) is decided by that one
arbitrary alignment — producing all-or-nothing, non-monotonic curves.
A real radar array is also arbitrarily aligned to the scene, so the honest
statistic is the EXPECTATION over alignments: each h is now evaluated at
PHASES x PHASES grid origin offsets and every (frame, phase) pair counts as
one trial. Error bars then reflect genuine alignment sensitivity.

Outputs
-------
* results/sweep_results.csv
* results/sweep_summary.txt
* results/sweep_curves.png

Usage
-----
python3 sweep.py ball_static box_static --every 20 [--debug]
"""

from __future__ import annotations

import csv
import glob
import math
import os
import sys
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from discretize import Partition
from fit_static import load_table, object_points
from geometry import classify_cluster, load_masked
from heightfit import classify_height


TRUTH = {
    "ball_static": "ball",
    "ball_moving": "ball",
    "box_static": "box",
    "box_moving": "box",
}

# Number of grid-origin offsets per axis (PHASES x PHASES placements per h).
PHASES = 3

# Minimum occupied cells needed to even ATTEMPT classification in the
# degraded (cell-level) view. 4 = the sphere fit's minimal sample; below
# this no hypothesis is testable and the outcome is 'sparse', not
# 'ambiguous'. (Was 8, which silently forbade classification of the ball
# at h >= ~25 mm and contaminated the measured cliff.)
MIN_CELL_POINTS = 4

DEBUG = False


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    This avoids any dependency on scipy and is more reliable than a naive
    normal approximation when counts are small.
    """
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


METHOD = "fit"   # fit | height | smart  (set via --method)


def degraded_classify(cluster_uvh: np.ndarray, part: Partition) -> Dict[str, object]:
    """Classify from the cell-level view only (simulated radar at
    resolution h): one 3D point per occupied cell, (u_center, v_center,
    mean height).

    The classifier itself is unchanged; only the representation is degraded.
    """
    g = part.grids(cluster_uvh)
    ii, jj = np.nonzero(g["count"] > 0)
    if len(ii) == 0:
        return {"kind": "unknown", "detail": "no occupied cells"}

    uc, vc = part.center_of(ii, jj)
    cell_pts = np.column_stack([uc, vc, g["mean_h"][ii, jj]])
    if METHOD == "height":
        return classify_height(cell_pts, part.h)

    return classify_cluster(
        cell_pts,
        thresh_mm=max(config.SPHERE_THRESH_MM, part.h / 2.0),
        min_points=MIN_CELL_POINTS,
    )


def _largest_cluster(clusters: Sequence[Dict[str, object]]) -> Dict[str, object] | None:
    if not clusters:
        return None
    return clusters[0]


def _phase_partitions(z: Dict[str, np.ndarray], h: float) -> List[Partition]:
    """PHASES x PHASES partitions whose grid origins are offset by
    fractions of a cell. Extending the range start by (frac * h) shifts
    every cell boundary by that amount without excluding any point."""
    u0, u1 = float(z["u_range"][0]), float(z["u_range"][1])
    v0, v1 = float(z["v_range"][0]), float(z["v_range"][1])
    fracs = [k / PHASES for k in range(PHASES)]
    return [Partition((u0 - fu * h, u1), (v0 - fv * h, v1), h=h)
            for fu in fracs for fv in fracs]


def run_one(name: str, files: Sequence[str], bg: np.ndarray, tf,
            z: Dict[str, np.ndarray], h: float) -> Dict[str, object]:
    parts = _phase_partitions(z, h)
    truth = TRUTH.get(name)
    n_trials = 0          # (frame, phase) pairs
    n_detected = 0
    n_correct = 0
    n_unknown = 0         # genuinely ambiguous (both hypotheses tested)
    n_sparse = 0          # too few cells to test any hypothesis
    n_wrong = 0
    cells_seen: List[int] = []
    pred_counts = {"ball": 0, "box": 0, "unknown": 0, "other": 0}

    for f in files:
        pts = object_points(load_masked(f), bg)
        if len(pts) < 10:
            n_trials += len(parts)
            continue
        uvh = tf.to_table(pts)

        for part in parts:
            n_trials += 1
            clusters, _ = part.clusters(uvh)
            cl = _largest_cluster(clusters)
            if cl is None:
                continue

            n_detected += 1
            cells_seen.append(int(cl["n_cells"]))

            # DEGRADED CLASSIFICATION — the point of the experiment.
            # The classifier only sees one point per occupied cell.
            res = degraded_classify(uvh[cl["points_mask"]], part)
            kind = str(res.get("kind", "unknown"))
            detail = str(res.get("detail", ""))

            if DEBUG:
                print(f"[DEBUG] {name} h={h:.1f} cells={cl['n_cells']} "
                      f"kind={kind} sphere={res.get('sphere_rmse')} "
                      f"({res.get('sphere_reason')}) "
                      f"box={res.get('box_rmse')} "
                      f"cov={res.get('coverage')} detail={detail}")

            if kind in pred_counts:
                pred_counts[kind] += 1
            else:
                pred_counts["other"] += 1

            if kind == truth:
                n_correct += 1
            elif kind == "unknown" and detail == "too few points":
                n_sparse += 1
            elif kind == "unknown":
                n_unknown += 1
            else:
                n_wrong += 1

    det_rate = n_detected / max(n_trials, 1)
    id_rate = n_correct / max(n_detected, 1)
    unk_rate = n_unknown / max(n_detected, 1)
    sparse_rate = n_sparse / max(n_detected, 1)
    wrong_rate = n_wrong / max(n_detected, 1)
    correct_per_frame = n_correct / max(n_trials, 1)

    det_lo, det_hi = wilson_ci(n_detected, n_trials)
    id_lo, id_hi = wilson_ci(n_correct, n_detected)
    unk_lo, unk_hi = wilson_ci(n_unknown, n_detected)
    sparse_lo, sparse_hi = wilson_ci(n_sparse, n_detected)
    wrong_lo, wrong_hi = wilson_ci(n_wrong, n_detected)
    cpf_lo, cpf_hi = wilson_ci(n_correct, n_trials)

    return dict(
        recording=name,
        h=float(h),
        m=int(parts[0].m),
        n=int(parts[0].n),
        phases=int(len(parts)),
        trials=int(n_trials),
        detected=int(n_detected),
        correct=int(n_correct),
        unknown=int(n_unknown),
        sparse=int(n_sparse),
        wrong=int(n_wrong),
        det_rate=float(det_rate),
        det_lo=float(det_lo),
        det_hi=float(det_hi),
        id_rate=float(id_rate),
        id_lo=float(id_lo),
        id_hi=float(id_hi),
        unk_rate=float(unk_rate),
        unk_lo=float(unk_lo),
        unk_hi=float(unk_hi),
        sparse_rate=float(sparse_rate),
        sparse_lo=float(sparse_lo),
        sparse_hi=float(sparse_hi),
        wrong_rate=float(wrong_rate),
        wrong_lo=float(wrong_lo),
        wrong_hi=float(wrong_hi),
        correct_per_frame=float(correct_per_frame),
        cpf_lo=float(cpf_lo),
        cpf_hi=float(cpf_hi),
        cells_across=float(np.median(cells_seen)) if cells_seen else 0.0,
        pred_ball=int(pred_counts["ball"]),
        pred_box=int(pred_counts["box"]),
        pred_unknown=int(pred_counts["unknown"]),
        pred_other=int(pred_counts["other"]),
    )


def threshold_h(rows: Sequence[Dict[str, object]], key: str, target: float) -> float | None:
    """Largest h such that the metric meets the target at EVERY h up to and
    including it (monotone-prefix criterion).

    The old definition (largest h with a passing value anywhere) let a
    single lucky grid alignment at a coarse h set the headline number even
    when finer resolutions in between failed.
    """
    rs = sorted(rows, key=lambda r: float(r["h"]))
    best = None
    for r in rs:
        if float(r[key]) >= target:
            best = float(r["h"])
        else:
            break
    return best


def write_summary(rows: Sequence[Dict[str, object]], out_path: str) -> None:
    by_recording: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        by_recording.setdefault(str(r["recording"]), []).append(r)

    lines: List[str] = []
    lines.append("Resolution sweep summary")
    lines.append("=" * 80)
    lines.append(f"(each h evaluated at {PHASES * PHASES} grid phases; "
                 f"rates are expectations over alignment)")
    lines.append("")
    for name in sorted(by_recording):
        rs = sorted(by_recording[name], key=lambda r: float(r["h"]))
        kind = TRUTH.get(name, "?")
        lines.append(f"Recording: {name}  (truth: {kind})")
        lines.append(f"  best detection h at >=0.95: {threshold_h(rs, 'det_rate', 0.95)}")
        lines.append(f"  best identification h at >=0.95: {threshold_h(rs, 'id_rate', 0.95)}")
        lines.append(f"  best identification h at >=0.90: {threshold_h(rs, 'id_rate', 0.90)}")
        lines.append(f"  best identification h at >=0.80: {threshold_h(rs, 'id_rate', 0.80)}")
        lines.append(f"  largest object span in cells (median at finest h): "
                     f"{max(float(r['cells_across']) for r in rs):.2f}")
        lines.append("")

    lines.append("Interpretation guide")
    lines.append("- det_rate: object found at all")
    lines.append("- id_rate: found object classified correctly")
    lines.append("- unk_rate: classifier tested both hypotheses and admits ambiguity")
    lines.append("- sparse_rate: too few occupied cells to test any hypothesis")
    lines.append("- wrong_rate: classifier makes the wrong shape decision")
    lines.append("- correct_per_frame: correct classification per trial")
    lines.append("- thresholds use the monotone-prefix criterion: every finer h must also pass")
    lines.append("")
    lines.append("These thresholds are the first candidates for the paper's 'minimum usable resolution' claim.")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main(names: Sequence[str], every: int) -> None:
    bg = np.load(os.path.join(config.RESULTS_DIR, "background_median.npy"))
    tf, z = load_table()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for name in names:
        if name not in TRUTH:
            print(f"skipping {name}: no single-object ground truth")
            continue
        files = sorted(glob.glob(f"Data/extracted/{name}/depth/*.npy"))[::every]
        print(f"{name}: {len(files)} sampled frames x {PHASES * PHASES} grid phases")
        for h in config.H_SWEEP:
            r = run_one(name, files, bg, tf, z, h)
            rows.append(r)
            print(
                f"  h={h:5.1f} mm  grid {r['m']:3d}x{r['n']:3d}  "
                f"~{r['cells_across']:.0f} cells over object | "
                f"det {r['det_rate']:.2f} [{r['det_lo']:.2f},{r['det_hi']:.2f}]  "
                f"id {r['id_rate']:.2f} [{r['id_lo']:.2f},{r['id_hi']:.2f}]  "
                f"unk {r['unk_rate']:.2f}  sparse {r['sparse_rate']:.2f}  "
                f"wrong {r['wrong_rate']:.2f} [{r['wrong_lo']:.2f},{r['wrong_hi']:.2f}]"
            )

    if not rows:
        print("No sweep rows were generated; check input names and data paths.")
        return

    csv_path = os.path.join(config.RESULTS_DIR, f"sweep_results_{METHOD}.csv")
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"saved {csv_path}")

    summary_path = os.path.join(config.RESULTS_DIR, f"sweep_summary_{METHOD}.txt")
    write_summary(rows, summary_path)
    print(f"saved {summary_path}")

    # ---- curves ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name in sorted({str(r["recording"]) for r in rows}):
        rs = [r for r in rows if r["recording"] == name]
        rs.sort(key=lambda r: float(r["h"]))
        hs = [float(r["h"]) for r in rs]
        kind = TRUTH[name]
        st = config.STYLE[kind]

        axes[0].errorbar(
            hs,
            [float(r["id_rate"]) for r in rs],
            yerr=[
                [max(0.0, float(r["id_rate"]) - float(r["id_lo"])) for r in rs],
                [max(0.0, float(r["id_hi"]) - float(r["id_rate"])) for r in rs],

            ],
            marker=st["marker"],
            color=st["color"],
            linewidth=1.2,
            capsize=2,
            label=f"{name} identification",
        )
        axes[0].plot(
            hs,
            [float(r["unk_rate"]) for r in rs],
            ":",
            marker="$S$",
            color="grey",
            label=f"{name} ambiguous",
        )
        axes[0].plot(
            hs,
            [float(r["sparse_rate"]) for r in rs],
            "--",
            marker=".",
            color="0.6",
            label=f"{name} sparse",
        )

        axes[1].errorbar(
            hs,
            [float(r["det_rate"]) for r in rs],
            yerr=[
                [float(r["det_rate"]) - float(r["det_lo"]) for r in rs],
                [float(r["det_hi"]) - float(r["det_rate"]) for r in rs],
            ],
            marker=st["marker"],
            color=st["color"],
            linewidth=1.2,
            capsize=2,
            label=f"{name} detection",
        )

    for ax, title in zip(
        axes,
        ["identification vs cell size", "detection vs cell size"],
    ):
        ax.axhline(0.95, color="0.85", linewidth=1.0, linestyle="--")
        ax.set_xlabel("cell size h (mm)")
        ax.set_ylabel("rate")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle("Resolution sweep: where does shape knowledge die?")
    fig.savefig(
        os.path.join(config.RESULTS_DIR, f"sweep_curves_{METHOD}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"saved results/sweep_curves_{METHOD}.png")


if __name__ == "__main__":
    if "--debug" in sys.argv:
        DEBUG = True
    if "--method" in sys.argv:
        METHOD = sys.argv[sys.argv.index("--method") + 1]
        assert METHOD in ("fit", "height", ), METHOD
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    names = args or ["ball_static", "box_static"]
    every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 20
    main(names, every)