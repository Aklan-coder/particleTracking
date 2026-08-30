# GEOMETRY RESOLUTION VALIDATION REPORT

## OFFICIAL IDENTIFICATION NUMBERS -- reproduced via sweep.run_one() verbatim
Official identification numbers reproduced via the EXACT sweep.run_one() function on real data (phase-averaged, matches the production experiment). Any discrepancy vs previously saved results is reported above, not hidden.

## DETAILED PER-MATCHED-FRAME ANALYSIS (single grid phase, zero offset)
## GEOMETRY FEATURE SEPARATION: ball vs box, per h
## THRESHOLD AUDIT (Section 12) -- no thresholds changed
**One threshold is genuinely and deliberately h-dependent**: the RANSAC inlier band scales as max(4mm, h/2) in the degraded (cell-level) path. This is stated in sweep.py's own docstring as an intentional accommodation for growing quantization residuals at coarse h, not a bug. It is flagged here per your audit instruction, not silently accepted. `MIN_CELL_POINTS=4` (local to sweep.py) is a FIXED cell-count floor, distinct from `config.MIN_CLUSTER_POINTS=40` used by the full-resolution (non-degraded) pipeline -- confirmed these are two different constants for two different code paths, not the same threshold reused inconsistently.

## THRESHOLD SENSITIVITY DIAGNOSTIC (Section 13) -- diagnostic only, NOT tuning
This reports how many real samples sit close to the existing CLASS_MARGIN decision boundary at each h -- a HIGH fraction means the classification at that h is inherently brittle regardless of the exact threshold chosen. No new threshold was searched for or reported as improving accuracy.

## GEOMETRY vs LEARNED CLASSIFIER (using ALREADY-VALIDATED learned results)
**CAUTION, stated explicitly:** geometry numbers here come from `ball_static`/`box_static` (single-object recordings). The learned classifier's saved accuracy may come from a different evaluation set (check `results/phase2_results.csv`'s own methodology). **This comparison is CONTEXTUAL, not necessarily sample-matched or apples-to-apples** -- do not present it as a controlled comparison without verifying the underlying test populations match.

## IDENTIFICATION vs TRACKING -- explicit distinction
**This script investigates GEOMETRY-BASED SHAPE IDENTIFICATION under spatial coarsening only.** It does NOT measure temporal tracking robustness vs h -- that is a separate question, not addressed here, and failure of shape identification at a given h does not by itself imply tracking (position estimation across frames) also fails at that h.
