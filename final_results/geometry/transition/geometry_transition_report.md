# GEOMETRY TRANSITION REPORT (20-30mm, real data, unmodified production pipeline)

## OFFICIAL PHASE-AVERAGED ACCURACY (sweep.run_one(), verbatim, 9 phases per h)
## INSTRUMENTED PER-TRIAL DETAIL (same phases/frames/threshold as sweep.run_one())
## IDENTIFICATION ACCURACY vs OCCUPIED-CELL COUNT (data-derived bins)
Bins are DATA-DERIVED (deciles of the observed occupied-cell distribution across all matched trials), not a predetermined minimum-cell threshold chosen in advance. **Cell count alone is not claimed as causal** -- see transition_fit_validity.csv (sphere/box fit success rate) alongside this table before concluding cell count is the sole driver; fit validity and cell arrangement matter too, per your instruction.

## TRANSITION TABLE + STEP-TO-STEP CHANGES
**Largest single-step balanced-accuracy drop occurs at 25.0->27.5mm (+0.4076).** See transition_by_object.csv and transition_fit_validity.csv for which geometric quantity (cell count, sphere/box fit validity, decision margin) changes most sharply at that same step -- reported as observed co-occurring changes, not asserted as the single cause.
