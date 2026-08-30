# FINAL RESULTS SUMMARY
Generated from already-validated result files. Every number below was PARSED from a source CSV at generation time -- if a value shows 'N/A', that field could not be located in the copied source file(s); it was not filled in from any other source.

## Research objective
Determine how spatial coarsening of real depth-derived 3D observations affects (a) explicit geometric shape identification, (b) a learned ball-vs-box classifier, and (c) multi-frame trajectory prediction -- as a proof of concept toward future radar/mmWave sensing. The spatial cell size `h` used throughout is a depth-camera-derived simulated quantity; it has **not yet been validated against any specific physical radar's resolution characteristics**.

## Geometry results
Fine 20-30mm transition (real ball_static/box_static data, unmodified production geometry pipeline):

| h (mm) | Ball ID | Box ID | Balanced ID |
|---:|---:|---:|---:|
| 20.00 | 100.00% | 94.02% | 97.01% |
| 22.50 | 77.78% | 100.00% | 88.89% |
| 25.00 | 36.90% | 82.91% | 59.91% |
| 27.50 | 21.63% | 16.67% | 19.15% |
| 30.00 | 3.77% | 0.00% | 1.88% |

Largest single-step balanced-accuracy decline (computed from the table above): **25.0→27.5mm** (40.76 percentage points).

## Learned classification results
**Independent generalization evidence:** the frozen learned classifier (trained only on `ball_static`/`ball_moving`/`box_static`/`box_moving`) was evaluated on the SEPARATE `ball_and_box_moving` recording, which was excluded from training entirely. This is not a within-recording test.

| h | Unseen-recording accuracy |
|---:|---:|
| 10.00 | 98.42% |
| 20.00 | 98.11% |
| 30.00 | 98.14% |
| 40.00 | 97.49% |

## Geometry-vs-learned complementarity
**Two SEPARATE experiments below -- do not merge them:**

### (A) Clean matched unseen-frame diagnostic comparison (SMALL sample)
Same real frames, same h, given to both methods; restricted to frames NOT seen during learned-model training. **Small per-h sample size -- indicative, not high-powered statistical evidence.**

| h | Clean unseen N | Geometry accuracy | Learned accuracy |
|---:|---:|---:|---:|
| 10.00 | 32 | 93.75% | 100.00% |
| 20.00 | 32 | 100.00% | 100.00% |
| 30.00 | 32 | 0.00% | 100.00% |
| 40.00 | 32 | 0.00% | 100.00% |

### (B) Independent unseen-recording result (STRONGER generalization evidence)
See the Learned classification results table above (`ball_and_box_moving`, a genuinely separate recording, much larger sample). This is the primary generalization evidence for the learned classifier -- the clean matched comparison in (A) is a complementary diagnostic about what each REPRESENTATION retains at each h, not a replacement for it.

## Trajectory prediction results
Final LOCKED model (k=10, horizon=10; selected using VALIDATION before TEST was ever opened), evaluated on the track-independent TEST set:

- N = 1854

| Metric | Learned k=10 | Constant velocity | Improvement vs CV |
|---|---:|---:|---:|
| Mean | 18.15 mm | 22.14 mm | +18.0% |
| Median | 13.69 mm | 15.86 mm | +13.7% |
| RMSE | 24.24 mm | 33.51 mm | +27.7% |
| P90 | 36.44 mm | 43.73 mm | +16.7% |
| P95 | 47.34 mm | 60.10 mm | +21.2% |

Per-track (ball vs box):

| Object | Learned median | CV median | Improvement |
|---|---:|---:|---:|
| ball | 23.06 mm | 25.21 mm | 8.55% |
| box | 10.35 mm | 13.01 mm | 20.45% |

## Computational profile (k=10 trajectory predictor)
**Covers the trajectory-prediction component only, NOT the complete depth-camera processing pipeline (backprojection, RANSAC fitting, discretization, Kalman tracking are all upstream and unmeasured here).**

| Quantity | Value |
|---|---:|
| Learned parameters | 20 |
| Raw learned parameter memory (W+b) | 160 bytes |
| Total inference-state memory (incl. normalization) | 232 bytes |
| k=10 history-buffer memory | 400 bytes |
| Scalar operations per prediction (profiler's own convention) | 88 |
| Model matvec MACs | 18 |
| Measured median total trajectory-prediction latency | 27.30 microseconds |
| Meets nominal 30-FPS predictor-only requirement | True |

## Main findings
**A. Explicit geometry:**
- Strong (balanced ID ~97%) around h=20mm under the current method.
- Progressively degrades beyond h=20mm.
- The largest single-step balanced-accuracy decline occurs at **25.0→27.5mm**, computed directly from the transition table above.
- Geometric fit validity/spatial support also deteriorates over the same range (see `transition_fit_validity.csv`).

**B. Learned classification:**
- Remains approximately 97.5-98.4% on the separate unseen `ball_and_box_moving` recording across the tested h range (parsed directly from the table above).
- The clean matched unseen-frame evidence (small sample) shows the learned classifier can retain discrimination on coarse observations where the explicit geometry rule fails.

**C. Trajectory:**
- k=10 was selected using VALIDATION performance, before TEST was ever opened.
- Final held-out-track TEST median error ≈ 13.69 mm.
- The learned predictor beats constant velocity across mean/median/RMSE/P90/P95 on the held-out TEST set (see comparison table above).

**D. Computational profile:**
- Extremely small predictor/model (see table above).
- Only the predictor computation has been profiled -- NOT the full depth-to-prediction pipeline.

## Limitations
- `h` is a depth-camera-derived simulated spatial cell size, **not a validated physical radar-resolution specification**.
- Geometry and the independent unseen-recording learned evaluation are not always evaluated on identical populations -- see the matched-comparison section for the sample-matched evidence, and do not conflate it with the independent generalization evidence.
- The clean matched comparison uses a small per-h sample (32 trials per h) -- indicative, not high-powered statistical evidence.
- Ball trajectory prediction is harder than box in the final held-out test (see per-track table above).
- Final trajectory testing is track-independent; the BALL test track is NOT fully recording-independent (other `ball_moving` tracks were in TRAIN/VALIDATION), while the BOX test track is both track- AND recording-independent.
- The measured latency covers the trajectory-prediction component alone, not the full depth-to-prediction pipeline.
- **These results do not prove that a cheaper radar will work.**

## Next step toward radar
These results identify a **potential sensing-resolution versus algorithmic-capability trade space**, not a validated radar specification. The motivated next step is mapping these depth-derived spatial requirements to realistic radar characteristics, including:
- range resolution
- angular resolution
- cross-range resolution
- aperture
- bandwidth
- antenna/virtual-channel requirements
- operating distance
- computational/hardware cost

**Do not interpret h=20mm as directly meaning "20mm radar resolution," and do not interpret the h=40mm results as proof that a cheaper radar is sufficient** -- both require dedicated future validation against real radar hardware.
