"""
config.py — Single source of truth for all measured/chosen constants.

Every constant here is either MEASURED from the data, DERIVED from hardware
specs, or an explicit DESIGN DECISION agreed with the professor. Scripts must
import from here and never hardcode their own copies.

Project: geometric (non-ML) identification, tracking, and prediction of
well-formed shapes (spheres, boxes) from depth video, with a mutable m x n
spatial partition simulating a future radar array.
"""

# ---------------------------------------------------------------------------
# Region of interest (pixels, full-frame coordinates).
# Applied as a MASK (outside pixels set to 0), never a physical crop, so the
# camera intrinsics below remain valid unchanged.
# Certified against motion heatmaps of all three moving recordings:
# no object trajectory ever leaves this region except past the right edge,
# where the sensor itself is unreliable (genuine observability limit).
# ---------------------------------------------------------------------------
ROI = dict(y1=0, y2=210, x1=0, x2=290)

# ---------------------------------------------------------------------------
# Camera intrinsics @ 320x240.
# fx, fy are the focal length in pixels; (cx, cy) the principal point.
# CURRENT: calibrated PrimeSense/Kinect values, fx = 570.342 / 2 at half
# resolution (consistent with the 570.3 used by uw_adapter.py), principal
# point at the pixel-center convention ((W-1)/2, (H-1)/2).
# Cross-check: fit_static.py ball diameter under these intrinsics is
# 60.66 +/- 0.64 mm; one ruler/caliper measurement of the ball confirms
# the metric chain (ratio ruler/fit = residual correction, expect ~1.00).
# RETIRED: the earlier nominal fx = 262.5 (lateral dims were ~8.6% larger).
# ---------------------------------------------------------------------------
# FX, FY, CX, CY (retired nominal): 262.5, 262.5, 160.0, 120.0

FX = 285.1711
FY = 285.1711
CX = 159.5
CY = 119.5

# ---------------------------------------------------------------------------
# Timing. Probe of all six .oni files measured 29.97 effective fps with
# ~0.1% dropped frames, so a uniform time step is justified.
# ---------------------------------------------------------------------------
DT = 1.0 / 30.0  # seconds per frame

# Per-pixel depth noise (1 sigma), MEASURED from consecutive static frames
# (frame-to-frame std 1.09 mm / sqrt(2)). Used for principled thresholds.
DEPTH_NOISE_MM = 0.77

# ---------------------------------------------------------------------------
# Partition (professor's mutable m x n grid). H_BASELINE is the default cell
# size for the phase-1 baseline; sweep.py varies h over H_SWEEP.
# Lower bound ~5 mm: source pixels span ~2.1-2.7 mm at working distance, and
# cells must be >= 2-3x the sample spacing to be honestly populated.
# ---------------------------------------------------------------------------
H_BASELINE = 20.0                   # mm
H_SWEEP = [5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0, 90.0, 100.0]

# ---------------------------------------------------------------------------
# Segmentation / classification parameters (all traceable to measurements).
# ---------------------------------------------------------------------------
BG_DIFF_MM = 12.0        # background-subtraction threshold: >> table noise
                         # (0.77 mm), << object height (tens of mm)
PLANE_THRESH_MM = 8.0    # RANSAC plane inlier band (~10 sigma of pixel noise)
SPHERE_THRESH_MM = 4.0   # RANSAC sphere inlier band
MIN_CLUSTER_POINTS = 40  # clusters smaller than this -> 'unknown' (too sparse
                         # for a meaningful shape decision at full resolution)
CLASS_MARGIN = 1.5       # winning model must beat the loser by this RMSE
                         # ratio, else 'unknown' (honest ambiguity)

# ---------------------------------------------------------------------------
# Tracking (Kalman) parameters.
# ---------------------------------------------------------------------------
GATE_MM = 60.0           # association gate: max distance between a track's
                         # prediction and a candidate detection (~1 ball/frame
                         # of very fast motion; generous but excludes swaps)
TRACK_BIRTH_FRAMES = 3   # unmatched detections must persist this many frames
TRACK_DEATH_FRAMES = 8   # missed frames before a track is terminated

# ---------------------------------------------------------------------------
# Plot style convention (professor's directive): every figure in the project
# uses exactly these symbols/colors so identification is readable at a glance.
#   ball  = O blue, box = X red, unknown/ambiguous = S grey
# ---------------------------------------------------------------------------
STYLE = {
    "ball":    dict(marker="o",   color="tab:blue", label="ball (O)"),
    "box":     dict(marker="x",   color="tab:red",  label="box (X)"),
    "unknown": dict(marker="$S$", color="grey",     label="ambiguous (S)"),
}

# Output locations (created on demand by scripts).
RESULTS_DIR = "results"