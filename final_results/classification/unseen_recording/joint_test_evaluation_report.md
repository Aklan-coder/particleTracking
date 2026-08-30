# JOINT UNSEEN CLASSIFICATION TEST -- EVALUATION REPORT

**Original test**: held-out frames from recordings represented during training (time-split within ball_static/ball_moving/box_static/box_moving).

**Joint unseen test**: completely separate ball_and_box_moving recording, never used during training. These are NOT equivalent experimental conditions.


## Pattern check (evidence-based, thresholds stated explicitly)

- **Pattern A supported**: unseen accuracy stays within 0.03 of original at every h (max drop observed: 0.0216).


(Any pattern not listed above was checked and NOT supported by the stated threshold on this data -- see the raw numbers in joint_test_metrics.csv and original_vs_joint_comparison.csv to judge for yourself; these thresholds are reporting aids, not a substitute for your own interpretation.)


## THIS TEST SET IS NOW FROZEN

Per your instruction: do not tune the classifier using these results and then re-report performance on this same recording as independent test performance. Any future model change would require a NEW independent recording for an unbiased final test.
