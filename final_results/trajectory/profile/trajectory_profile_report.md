# TRAJECTORY MODEL PROFILE REPORT (characterization only)

**No edge-device suitability is claimed anywhere in this report.** All latency/throughput numbers are specific to the machine this script was run on.


## ENVIRONMENT
Environment recorded above and in profiling_environment.txt. **All latency numbers below are specific to THIS machine and are NOT hardware-independent or edge-device benchmarks.**

## PARAMETER SIZE + MEMORY (Sections 3, 4, 19, 22, 23)
**Does parameter count change with k? NO -- confirmed identical ({20} learned params) at every k.** k changes temporal feature-CONSTRUCTION and history-STORAGE requirements (see history_buffer_memory.csv, which scales linearly with k), NOT the regression model's own dimensionality (fixed at 9 inputs -> 2 outputs regardless of k).

## ANALYTICAL OPERATION COUNTS (Sections 6, 7, 8, 9)
**Feature-construction workload is O(k)** (grows with k: 12 ops at k=1 -> 30 ops at k=10). **Model-inference workload is O(9x2)=O(18), FIXED regardless of k** (56 ops, identical at every k, confirmed above). Total workload k=10 vs k=1 ratio: 1.26x -- growth is small in absolute terms since k only affects two small averaging loops, not the model itself.

## MEASURED LATENCY (Sections 10, 11, 12) -- warm-up=200, reps=5000, perf_counter_ns
**Does total latency increase with k?** Median deltas vs k=1: ['0.00us', '-5.70us', '-5.80us', '-5.90us']. These differences are SMALL RELATIVE TO the measurement std (~13.2us) -- likely dominated by Python/NumPy call overhead, not by k itself. Not claiming a meaningful trend.

## THROUGHPUT + 30-FPS REAL-TIME REQUIREMENT (Sections 14, 15, 16)
**This evaluates the TRAJECTORY PREDICTOR ALONE.** It does NOT establish that the complete depth-camera pipeline (backprojection, RANSAC fitting, discretization, Kalman tracking -- all upstream and unmeasured here) can sustain 30 FPS. Do not extrapolate this result to the full system.

## ACCURACY-COST TRADEOFF (Section 18) -- using ALREADY-VALIDATED accuracy, not recomputed
Accuracy-cost table populated from already-validated accuracy figures (not recomputed here) joined with this script's own parameter/memory/latency measurements. No automatic winner is declared -- see the raw table for k=3 (accuracy) vs k=10 (RMSE/tail) vs cost tradeoffs.


## Upstream costs explicitly OUT OF SCOPE (Section 25)

depth loading -> point-cloud backprojection -> RANSAC plane/sphere/box fitting -> grid discretization -> Kalman filter tracking. All of this happens in track_moving.py, entirely BEFORE any row reaches window_features(). None of it is measured by this script.
