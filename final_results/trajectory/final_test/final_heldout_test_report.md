# FINAL HELD-OUT TRAJECTORY TEST REPORT

**This TEST result is a one-time evaluation of the model selected using VALIDATION. No model or hyperparameter changes were made using TEST performance.** If future model changes are made after examining these results, this TEST set should no longer be described as untouched final test data for those changed models.


## LIMITATION (Section 21)

This is **TRACK-INDEPENDENT** testing. It is NOT completely recording-independent for every object:
- **BOX test (box_moving track 1): track-independent AND recording-independent** -- box_moving never appeared in TRAIN or VALIDATION at all.
- **BALL test (ball_moving track 4): track-independent but NOT recording-independent** -- other ball_moving tracks appeared in TRAIN/VALIDATION.


# FINAL HELD-OUT TRAJECTORY TEST
## PRIMARY TEST METRICS
## IMPROVEMENT VS BASELINES (all metrics, not cherry-picked)
## PER-TRACK TEST PERFORMANCE
## OBJECT-BALANCED SUMMARY (average of ball-track and box-track medians, NOT pooled)
## TRAIN -> VALIDATION -> TEST GENERALIZATION GAP
## LARGE-ERROR TAIL (top 5%)
## ERROR VS MOTION (post-hoc TEST analysis -- NOT used to modify the model)
**This motion-error analysis is post-hoc and descriptive only. It was NOT and must NOT be used to modify the model, per instruction #14.**
