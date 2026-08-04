# Tabletop geometric tracking + resolution-limit learning pipeline

Purely geometric (phase 1) identification, tracking, and prediction of
well-formed shapes (sphere, box) from PrimeSense depth video, built around
a **mutable m x n spatial partition** simulating a future mmWave radar
array — plus a phase-2 learning experiment testing whether trained models
identify objects **below the geometric resolution limit**.

Design commitments:
- **Phase 1 contains no ML.** Every stage is an explicit geometric
  algorithm; every constant is measured, derived, or a stated assumption
  (see `config.py`).
- **Classification by competing hypotheses** (RANSAC sphere fit vs
  plane-set fit, residual comparison). `unknown` (grey S) is an honest
  first-class outcome.
- **Cells are not binary**: per-cell value matrices (count, mean/max
  height, height variance) are computed and archived for every frame.
- **Phase 2 models are deliberately explainable** (plain-numpy logistic
  regression) so *what* they learned is readable from the weights.
- Plot convention everywhere: **ball = blue O, box = X red,
  ambiguous = S grey**.

## Requirements
Python 3.9+, `pip3 install numpy scipy matplotlib pillow`

## Files (17)

### Libraries (imported, never run directly)
| file | role |
|---|---|
| config.py | all constants with provenance: ROI, intrinsics, DT, measured noise, partition sizes, tracking params, plot style |

| geometry.py | back-projection, RANSAC plane/sphere, plane-set (box) fit, competing-hypotheses classifier, table frame. Includes shallow-cap rejection and camera-oriented plane normal (both real-data bug fixes) |

| discretize.py | the mutable m x n partition: per-cell value matrices + grid-based clustering (the partition IS the segmentation) |

| tracking.py | Kalman filter (measured R), Mahalanobis-gated association, track lifecycle with class votes |

| merge_split.py | shape-aware splitting of merged clusters: when two tracked objects touch, their points divide by distance to each track's predicted BODY (not center), preserving both identities through contact |

### Phase 1 — runnable, in order
| step | script | outputs in results/ |
|---|---|---|
| 1 | selftest.py | must print ALL SELF-TESTS PASSED (6 tests incl. two-object crossing). Rerun after ANY library change |

| 2 | build_reference.py | background_median.npy, table_plane.npz |

| 3 | fit_static.py | ball/box_static_fits.csv, static_summary.txt — intrinsics check vs ruler + Kalman measurement noise |

| 4 | track_moving.py <rec> [--h MM] | *_tracks.csv, *_cells.npz, *_trajectory.png, *_grid.png, two-panel *_tracking.gif (camera view + live partition view with gridlines) |

| 5 | predict.py | prediction_errors.csv, prediction_curves.png — shape-conditioned physics vs constant velocity |

| 6 | sweep.py <recs> --every N | sweep_results.csv, sweep_curves.png — identification/detection vs h, classifying from CELL-LEVEL data only |

| 7 | plot_sweep_mm.py | sweep_curves_mm_6panel.png — six-panel figure matching the earlier pixel-pipeline layout |

### Phase 2 — runnable, in order (after phase 1)
| step | script | purpose |
|---|---|---|
| 8 | make_dataset.py | datasets at h = 10/20/30/40 mm; geometry-produced labels; TIME split (default TRAIN_FRAC = 0.80) |
| 9 | train_model.py | 8 explainable models (4 h x {pattern-only, pattern+size}) |
| 10 | eval_model.py | held-out-test accuracies, top features per model, learning_vs_geometry.png |
| 11 | uw_adapter.py <path> | converts UW RGB-D Object Dataset frames (depthcrop/maskcrop/loc, Kinect intrinsics 570.3) into our patch format, using camera-relief as the height proxy |
| 12 | eval_uw.py | cross-dataset generalization exam of the trained models (no retraining) |

## Headline results (this dataset)
- Intrinsics updated to the calibrated PrimeSense values fx = fy = 285.1711
  (= 570.342 / 2 at 320x240), cx = 159.5, cy = 119.5; the old nominal 262.5
  is retired. Lateral dimensions therefore shrank by x0.9205 vs earlier
  reports; heights are depth-dominated and nearly unchanged.
- Ball diameter 60.66 +/- 0.64 mm (1,117 fits); box 133.1 x 57.3 x 35.3 mm
  (1,027 fits); center scatter 0.075-0.372 mm (regenerated under the new
  intrinsics). Ruler cross-check of the ball diameter (~60-61 mm expected)
  confirms the whole metric chain in one measurement.
- Tracking 92-99.9% coverage, zero identity swaps; merge splitting holds
  both identities through physical contact (verified on the two-object
  recording: all 'active tracks: 1' dips became continuous 2s).
- Measured decelerations: ball 23 mm/s^2 (rolling) vs box 154 mm/s^2
  (sliding). Shape-conditioned prediction beats constant velocity by 18%
  for the box at +10 frames; correctly coincides for the ball.
  (Computed under fx = 262.5; lateral velocities/accelerations scale by
  ~x0.92 under the new intrinsics — rerun track_moving.py + predict.py
  to refresh. Relative conclusions are scale-invariant and stand.)
- **Geometric identification cliff: h <= 25 mm (~2.4 cells across the
  61 mm ball); dead at 30 mm. Detection unaffected out to 60 mm** —
  (cliff values predate the intrinsics update; the object is ~8% smaller
  in mm, so rerun sweep.py — the cliff may shift one bin down) —
  identification and tracking have different resolution limits.
  Reproduced independently by the earlier pixel-cell pipeline
  (cliff at ~15 px ~ 31-38 mm).
- **Phase 2: trained models hold 0.989-0.999 test accuracy at h = 30-40 mm
  where geometry scores 0.02-0.20.** Attribution: dominated by height
  statistics (a single height threshold achieves 0.957-0.989); the learned
  combination reduces residual errors up to 40x.
- **Cross-dataset exam (UW RGB-D): transfer collapses to single-class
  prediction (ball 1.00 / box 0.00)** — within-domain identification was
  carried by scene-specific height, not transferable shape structure.
  This motivates: height-matched objects in the next recording session,
  height-normalized features, and multi-instance training (UW data usable
  as training material: 7 ball + 5 box instances).

## Known limitations / open items
- **Intrinsics updated, one cheap confirmation open**: config now uses the
  calibrated PrimeSense fx = 285.1711 (570.342 / 2). Static fits are
  regenerated; downstream mm results (tracking velocities, prediction
  errors, sweep, phase-2 datasets) still need a rerun under the new
  intrinsics. A single ruler/caliper measurement of the ball (expect
  ~60-61 mm) confirms the metric chain end to end.
- Confident-error band near h = 30 mm (static box -> 'ball'): no-operate
  zone; hardening candidate (stricter cap criterion or multi-frame
  stability).
- Merge splitting needs both tracks confirmed before contact; N-object
  crowds would need JPDA/MHT (named future work).
- Phase-2 scope: one table, two specific objects. Time-split rules out
  frame leakage but not object-level memorization — as the UW exam
  demonstrated. Generalization requires object variety.
- No RGB in these recordings (enable both streams next session).

## Data provenance
Six .oni recordings (PrimeSense, 320x240 @ 30 fps, depth only), 9,657
frames, 0.1% drops, extracted via the Docker/OpenNI2 tooling in
extraction_tools/. Sensor noise 0.77 mm (measured); ROI rows 0-210 /
cols 0-290 as a mask (certified vs motion heatmaps). UW RGB-D Object
Dataset used for the generalization exam — cite Lai, Bo, Ren & Fox,
ICRA 2011.



Run this 

python3 build_reference.py
python3 fit_static.py
python3 track_moving.py ball_moving
python3 track_moving.py box_moving