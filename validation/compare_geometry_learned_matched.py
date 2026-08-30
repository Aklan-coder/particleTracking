"""
compare_geometry_learned_matched.py — strictly matched, same-observation
comparison of geometry-based identification vs the frozen learned
classifier. Evaluation only: no retraining, no threshold changes, no
synthetic data, no hybrid-rule design. Modifies no existing file.

=== FIRST GATE (Section 2) -- RESOLVED BEFORE WRITING ANY EVALUATION CODE ===

A. Geometry's representation at each h: one 3D point per occupied cell
   (centroid u,v + mean height), built by sweep.degraded_classify()'s
   internal call to part.grids() -> classify_cluster().

B. Learned classifier's representation at each h: a 13x13x3 object-
   centered PATCH (occupancy, mean height, height variance), built by
   make_dataset.patch_for() -> train_model.features() -> logistic
   regression.

C. YES, both are constructible from the exact same raw ball_static /
   box_static frames -- confirmed by inspecting both code paths, which
   share the identical upstream chain (load_masked -> object_points ->
   tf.to_table -> Partition.clusters()) before diverging into their
   respective representations.

D/GRID PHASE (Section 6) -- CRITICAL FINDING, VERIFIED BY DIRECT CODE
INSPECTION, not assumed: `make_dataset.py` builds its Partition as
`Partition(z["u_range"], z["v_range"], h=h)` -- the RAW, UNSHIFTED
range. This is EXACTLY the zero-offset case of sweep.py's own
`_phase_partitions()` (fu=0, fv=0). The learned classifier has therefore
NEVER seen any grid alignment other than zero-offset.
CONSEQUENCE: this script uses ONLY zero-offset trials for BOTH methods.
This means geometry's accuracy numbers HERE will differ from sweep.py's
own headline numbers (which average over 9 phases) -- stated explicitly,
not conflated. This is Section 6 option (A): a genuinely common phase
condition constructed from each method's own existing production code,
not a compromise or new implementation.

VERDICT: A GENUINELY MATCHED COMPARISON IS POSSIBLE, restricted to
zero grid phase. Proceeding.

=== TRAINING-OVERLAP GATE (Section 2D, 18) ===

make_dataset.py's own dataset build uses TRAIN_FRAC=0.70, split BY TIME
per recording, on its OWN file listing (default every=1, i.e. ALL
frames). This script's matched trials are sampled at a stride (default
20, matching sweep.py's own default) from that SAME full file listing.
For each matched frame, this script computes its position in the FULL
(unstrided) file list and compares it against make_dataset.py's own
cut = int(0.70 * n_full_files) to determine TRAIN/TEST overlap exactly
-- not guessed, computed from the identical formula.

REUSED DIRECTLY, NOT REIMPLEMENTED: sweep.degraded_classify(), sweep.TRUTH,
sweep.MIN_CELL_POINTS, sweep._phase_partitions() (only its phase-0 entry
is used), make_dataset.patch_for(), train_model.features(),
eval_model.predict(), fit_static.object_points()/load_table(),
geometry.load_masked(), discretize.Partition, geometry.ransac_sphere()/
planes_fit() (direct calls, same threshold formula as degraded_classify,
for numeric residuals only).

Usage:
  python compare_geometry_learned_matched.py
  python compare_geometry_learned_matched.py --results-dir results --out validation_results/geometry_vs_learned_matched --every 20
  python compare_geometry_learned_matched.py --help
"""
import argparse
import csv
import glob
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = _THIS_DIR
for _ in range(4):
    if os.path.exists(os.path.join(_PROJECT_ROOT, "config.py")):
        break
    _PROJECT_ROOT = os.path.dirname(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

import config
from discretize import Partition
from fit_static import load_table, object_points
from geometry import load_masked, ransac_sphere, planes_fit
import sweep as sweep_mod
from make_dataset import patch_for, TRAIN_FRAC
import train_model as tm
from eval_model import predict as frozen_predict

H_TEST = [10.0, 20.0, 30.0, 40.0]
RECORDINGS = ["ball_static", "box_static"]


def proba_from_model(model, F):
    """Same formula as eval_model.predict(), threshold removed for a
    continuous decision score -- not a new model."""
    Z = (F - model["mu"]) / model["sd"]
    return 1.0/(1.0+np.exp(-np.clip(Z@model["w"]+float(model["b"]), -500, 500)))

def save_csv(path, rows, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w",newline="") as fh:
        w=csv.writer(fh); w.writerow(header); w.writerows(rows)
    return path

REPORT = []
def note(md): REPORT.append(md)
def heading(md):
    print("\n"+"="*78); print(md); print("="*78); REPORT.append(md)


def train_cut_for_recording(rec):
    """Reproduces make_dataset.py's OWN cut computation exactly (its own
    full, unstrided file listing, TRAIN_FRAC=0.70) -- for the training-
    overlap audit."""
    full_files = sorted(glob.glob(os.path.join(_PROJECT_ROOT, "Data", "extracted", rec, "depth", "*.npy")))
    cut = int(TRAIN_FRAC * len(full_files))
    return len(full_files), cut


def main():
    p = argparse.ArgumentParser(
        description="Strictly matched, same-observation comparison of "
                    "geometry vs the frozen learned classifier. Evaluation "
                    "only.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--out", type=str, default="validation_results/geometry_vs_learned_matched")
    p.add_argument("--every", type=int, default=20)
    args = p.parse_args()
    def _resolve(path): return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)
    args.results_dir = _resolve(args.results_dir); args.out = _resolve(args.out)
    plots_dir = os.path.join(args.out, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.isdir(os.path.join(_PROJECT_ROOT, "Data", "extracted")):
        print("NOTE: Data/extracted/ not found in this environment. This "
             "script requires real raw depth frames. Nothing computed.")
        return

    heading("# MATCHED GEOMETRY vs LEARNED CLASSIFIER")
    print("COMPARABILITY:")
    print("  Same raw frames: YES (identical file list per recording, shared by both methods)")
    print("  Same h: YES (both operate on the same Partition object per h)")
    print("  Same phase: YES, but RESTRICTED TO ZERO-OFFSET ONLY -- make_dataset.py "
         "never trained on any other phase (verified by code inspection above)")

    bg = np.load(os.path.join(args.results_dir, "background_median.npy"))
    tf, z = load_table()

    all_rows = []
    overlap_summary = {}

    for h in H_TEST:
        model_path = os.path.join(args.results_dir, f"model_h{int(h)}_full.npz")
        if not os.path.exists(model_path):
            print(f"\nh={h}: STOP -- {model_path} not found. Cannot proceed for this h.")
            continue
        model = np.load(model_path, allow_pickle=True)
        if any(k not in model for k in ["w","b","mu","sd"]):
            print(f"\nh={h}: STOP -- model missing required keys."); continue

        part = Partition(z["u_range"], z["v_range"], h=h)   # SAME zero-offset construction as make_dataset.py
        print(f"\nh={h} mm")

        for rec in RECORDINGS:
            n_full, cut = train_cut_for_recording(rec)
            files = sorted(glob.glob(os.path.join(_PROJECT_ROOT, "Data", "extracted", rec, "depth", "*.npy")))[::args.every]
            truth = sweep_mod.TRUTH[rec]
            thresh_mm = max(config.SPHERE_THRESH_MM, h/2.0)

            for i, f in enumerate(files):
                full_pos = i * args.every
                seen_in_train = full_pos < cut

                pts = object_points(load_masked(f), bg)
                if len(pts) < 10:
                    all_rows.append([rec, i, full_pos, truth, h, "unavailable","", "", "", "", "",
                                    "unavailable","", "", seen_in_train])
                    continue
                uvh = tf.to_table(pts)
                clusters, _ = part.clusters(uvh)
                if not clusters:
                    all_rows.append([rec, i, full_pos, truth, h, "unavailable","", "", "", "", "",
                                    "unavailable","", "", seen_in_train])
                    continue
                cl = clusters[0]
                cluster_uvh = uvh[cl["points_mask"]]
                cpts = pts[cl["points_mask"]]

                # ---- GEOMETRY (reused verbatim) ----
                geo_res = sweep_mod.degraded_classify(cluster_uvh, part)
                geo_kind = geo_res.get("kind","unknown")
                geo_correct = (geo_kind == truth)
                g = part.grids(cluster_uvh)
                ii, jj = np.nonzero(g["count"]>0)
                n_occ = len(ii)
                rmse_s = rmse_b = np.inf
                if n_occ > 0:
                    uc, vc = part.center_of(ii, jj)
                    cell_pts = np.column_stack([uc, vc, g["mean_h"][ii,jj]])
                    try: _,_,_,rmse_s = ransac_sphere(cell_pts, thresh_mm=thresh_mm)
                    except (ValueError, np.linalg.LinAlgError): pass
                    try: _,rmse_b,_ = planes_fit(cell_pts, thresh_mm=thresh_mm)
                    except (ValueError, np.linalg.LinAlgError): pass

                # ---- LEARNED (reused verbatim: same patch_for, same features, same frozen model) ----
                patch, ncells = patch_for(cluster_uvh, part)
                if patch is None:
                    learned_kind, learned_correct, proba = "unavailable", False, ""
                else:
                    F, _ = tm.features(patch[None], np.array([ncells]), with_size=True)
                    p_box = float(proba_from_model(model, F)[0])
                    learned_kind = "box" if p_box > 0.5 else "ball"
                    learned_correct = (learned_kind == truth)
                    proba = p_box

                all_rows.append([rec, i, full_pos, truth, h, geo_kind, geo_correct,
                                f"{rmse_s:.3f}" if np.isfinite(rmse_s) else "inf",
                                f"{rmse_b:.3f}" if np.isfinite(rmse_b) else "inf",
                                n_occ, "", learned_kind, learned_correct, proba, seen_in_train])

        h_rows = [r for r in all_rows if r[4]==h]
        n = len(h_rows)
        geo_acc = sum(1 for r in h_rows if r[6]==True)/max(n,1)
        learned_acc = sum(1 for r in h_rows if r[12]==True)/max(n,1)
        n_train_overlap = sum(1 for r in h_rows if r[14])
        print(f"  N={n}  geometry_acc={geo_acc:.4f}  learned_acc={learned_acc:.4f}  "
             f"trials_seen_in_learned_TRAIN={n_train_overlap}/{n} "
             f"({100*n_train_overlap/max(n,1):.1f}%)")
        overlap_summary[h] = (n, n_train_overlap)

    save_csv(os.path.join(args.out, "matched_trial_predictions.csv"), all_rows,
            ["recording","sample_index","full_list_position","truth","h_mm",
             "geometry_pred","geometry_correct","sphere_rmse","box_rmse","occupied_cells",
             "geometry_detail","learned_pred","learned_correct","learned_proba_box","seen_in_learned_train"])

    # ---- primary matched comparison table ----
    heading("## PRIMARY MATCHED COMPARISON")
    summary_rows, obj_rows, unavail_rows, recovery_rows, reverse_recovery_rows, agree_rows, mcnemar_rows = [],[],[],[],[],[],[]
    for h in H_TEST:
        h_rows = [r for r in all_rows if r[4]==h]
        if not h_rows: continue
        n = len(h_rows)
        geo_ok = [r[6]==True for r in h_rows]
        learn_ok = [r[12]==True for r in h_rows]
        both = sum(1 for g,l in zip(geo_ok,learn_ok) if g and l)
        geo_only = sum(1 for g,l in zip(geo_ok,learn_ok) if g and not l)
        learn_only = sum(1 for g,l in zip(geo_ok,learn_ok) if not g and l)
        both_wrong = sum(1 for g,l in zip(geo_ok,learn_ok) if not g and not l)
        geo_acc = sum(geo_ok)/n; learn_acc = sum(learn_ok)/n
        print(f"\nh={h}: N={n} geometry_acc={geo_acc:.4f} learned_acc={learn_acc:.4f} "
             f"both_correct={both} geo_only={geo_only} learned_only={learn_only} both_wrong={both_wrong}")
        summary_rows.append([h, n, geo_acc, learn_acc, both, geo_only, learn_only, both_wrong])

        # McNemar (paired discordant test) -- caveat about correlated trials stated
        b, c = geo_only, learn_only
        chi2 = ((abs(b-c)-1)**2)/(b+c) if (b+c)>0 else float("nan")
        print(f"  McNemar discordant pairs: geometry-only={b} learned-only={c} chi2(approx)={chi2:.3f} "
             f"[CAUTION: trials include adjacent frames from the same recording, "
             f"not fully independent -- treat as descriptive, not a rigorous p-value]")
        mcnemar_rows.append([h, b, c, chi2])

        # by object
        for obj in ["ball","box"]:
            osel = [r for r in h_rows if r[3]==obj]
            if not osel: continue
            og = [r[6]==True for r in osel]; ol = [r[12]==True for r in osel]
            ob_ = sum(1 for g,l in zip(og,ol) if g and l)
            og_only = sum(1 for g,l in zip(og,ol) if g and not l)
            ol_only = sum(1 for g,l in zip(og,ol) if not g and l)
            obw = sum(1 for g,l in zip(og,ol) if not g and not l)
            print(f"  {obj}: N={len(osel)} geo_acc={sum(og)/len(osel):.3f} learn_acc={sum(ol)/len(osel):.3f} "
                 f"both={ob_} geo_only={og_only} learned_only={ol_only} both_wrong={obw}")
            obj_rows.append([h, obj, len(osel), sum(og)/len(osel), sum(ol)/len(osel), ob_, og_only, ol_only, obw])

        # geometry unavailable
        unavail = [r for r in h_rows if r[5]=="unavailable"]
        valid = [r for r in h_rows if r[5]!="unavailable"]
        n_unavail = len(unavail)
        learned_on_unavail = [r for r in unavail if r[11]!="unavailable"]
        learned_correct_on_unavail = sum(1 for r in learned_on_unavail if r[12]==True)
        recovery = learned_correct_on_unavail/max(len(learned_on_unavail),1)
        print(f"  geometry unavailable: {n_unavail}/{n} ({100*n_unavail/n:.1f}%)  "
             f"learned accuracy on those same trials: {recovery:.4f} "
             f"(n_learned_available={len(learned_on_unavail)})")
        unavail_rows.append([h, n_unavail, n, 100*n_unavail/n, len(learned_on_unavail), learned_correct_on_unavail, recovery])

        # geometry failure (wrong OR unavailable) -> learned recovery
        geo_fail = [r for r in h_rows if r[6]!=True]
        learned_correct_on_fail = sum(1 for r in geo_fail if r[12]==True)
        recov_rate = learned_correct_on_fail/max(len(geo_fail),1)
        recovery_rows.append([h, len(geo_fail), learned_correct_on_fail, recov_rate])
        print(f"  among {len(geo_fail)} geometry FAILURES (wrong or unavailable): "
             f"learned correct on {learned_correct_on_fail} ({recov_rate:.4f} recovery rate)")

        # reverse: learned failure -> geometry recovery
        learn_fail = [r for r in h_rows if r[12]!=True]
        geo_correct_on_learn_fail = sum(1 for r in learn_fail if r[6]==True)
        rev_rate = geo_correct_on_learn_fail/max(len(learn_fail),1)
        reverse_recovery_rows.append([h, len(learn_fail), geo_correct_on_learn_fail, rev_rate])
        print(f"  among {len(learn_fail)} learned FAILURES: geometry correct on "
             f"{geo_correct_on_learn_fail} ({rev_rate:.4f} reverse-recovery rate)")

        # agreement
        both_ball = sum(1 for r in h_rows if r[5]=="ball" and r[11]=="ball")
        both_box = sum(1 for r in h_rows if r[5]=="box" and r[11]=="box")
        agree = sum(1 for r in h_rows if r[5]==r[11])
        print(f"  agreement rate: {agree/n:.4f}  both_ball={both_ball} both_box={both_box}")
        agree_rows.append([h, n, agree/n, both_ball, both_box])

    save_csv(os.path.join(args.out, "matched_comparison_summary.csv"), summary_rows,
            ["h_mm","n","geometry_accuracy","learned_accuracy","both_correct",
             "geometry_only","learned_only","both_wrong"])
    save_csv(os.path.join(args.out, "matched_comparison_by_object.csv"), obj_rows,
            ["h_mm","object","n","geometry_accuracy","learned_accuracy","both_correct",
             "geometry_only","learned_only","both_wrong"])
    save_csv(os.path.join(args.out, "geometry_unavailable_analysis.csv"), unavail_rows,
            ["h_mm","n_unavailable","n_total","pct_unavailable","n_learned_available_on_those",
             "n_learned_correct_on_those","learned_recovery_rate"])
    save_csv(os.path.join(args.out, "geometry_failure_learned_recovery.csv"), recovery_rows,
            ["h_mm","n_geometry_failures","n_learned_correct","learned_recovery_rate"])
    save_csv(os.path.join(args.out, "learned_failure_geometry_recovery.csv"), reverse_recovery_rows,
            ["h_mm","n_learned_failures","n_geometry_correct","geometry_reverse_recovery_rate"])
    save_csv(os.path.join(args.out, "agreement_analysis.csv"), agree_rows,
            ["h_mm","n","agreement_rate","both_predict_ball","both_predict_box"])

    # ---- training overlap audit ----
    heading("## TRAINING-OVERLAP AUDIT (Section 18) -- critical interpretation gate")
    overlap_rows = []
    for h, (n, n_overlap) in overlap_summary.items():
        pct = 100*n_overlap/max(n,1)
        overlap_rows.append([h, n, n_overlap, n-n_overlap, pct])
        print(f"  h={h}: {n_overlap}/{n} matched samples ({pct:.1f}%) were seen "
             f"during learned-model TRAIN.")
    save_csv(os.path.join(args.out, "training_overlap_audit.csv"), overlap_rows,
            ["h_mm","n_matched","n_seen_in_train","n_unseen","pct_seen_in_train"])
    any_high_overlap = any(r[4] > 20 for r in overlap_rows)
    note(f"**{'MOST/MANY matched samples were used in learned-model TRAIN.' if any_high_overlap else 'Most matched samples were NOT used in learned-model TRAIN.'}** "
        f"**THE MATCHED COMPARISON IS THEREFORE {'DIAGNOSTIC, NOT AN UNBIASED GENERALIZATION TEST' if any_high_overlap else 'A REASONABLE PROXY FOR GENERALIZATION, THOUGH NOT AS STRONG AS THE INDEPENDENT UNSEEN-RECORDING TEST'} for the learned classifier**, per your instruction #18. This does not invalidate the comparison's value for understanding what information each REPRESENTATION retains at each h, but it must not be presented as unbiased learned-model accuracy.")

    # ---- clean unseen subset ----
    clean_rows = [r for r in all_rows if r[14]==False and r[5]!="unavailable"]
    save_csv(os.path.join(args.out, "clean_unseen_subset.csv"), clean_rows,
            ["recording","sample_index","full_list_position","truth","h_mm",
             "geometry_pred","geometry_correct","sphere_rmse","box_rmse","occupied_cells",
             "geometry_detail","learned_pred","learned_correct","learned_proba_box","seen_in_learned_train"])
    print(f"\nClean unseen-frame subset (not in learned TRAIN): {len(clean_rows)} trials saved "
         f"to clean_unseen_subset.csv")

    # ---- independent learned generalization, reported separately, not recomputed ----
    heading("## INDEPENDENT LEARNED-CLASSIFIER GENERALIZATION (already-validated, reported separately, NOT recomputed)")
    joint_path = os.path.join(_PROJECT_ROOT, "validation_results", "joint_test", "evaluation", "joint_test_metrics.csv")
    if os.path.exists(joint_path):
        print(f"  (see {joint_path} for the independent ball_and_box_moving unseen-recording result)")
        note(f"Independent learned-classifier generalization evidence exists at "
            f"`{joint_path}` (ball_and_box_moving, a genuinely separate recording "
            f"never touched by this matched-comparison script). **Do not mix "
            f"that number with the matched-comparison numbers above** -- they "
            f"answer different questions.")
    else:
        print("  NOTE: independent evaluation file not found -- not fabricated here.")

    # ---- plots ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    hs = [r[0] for r in summary_rows]
    fig, ax = plt.subplots(figsize=(7,4.5))
    ax.plot(hs, [r[2] for r in summary_rows], "o-", label="geometry")
    ax.plot(hs, [r[3] for r in summary_rows], "s-", label="learned")
    ax.set_xlabel("h (mm)"); ax.set_ylabel("accuracy (matched trials)"); ax.legend(); ax.set_ylim(-0.05,1.05)
    ax.set_title("Matched-trial accuracy: geometry vs learned")
    fig.savefig(os.path.join(plots_dir, "accuracy_by_h.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7,4.5))
    ax.plot(hs, [r[3] for r in recovery_rows], "o-")
    ax.set_xlabel("h (mm)"); ax.set_ylabel("learned recovery rate")
    ax.set_title("Learned accuracy among geometry failures, vs h")
    fig.savefig(os.path.join(plots_dir, "recovery_by_h.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,5))
    width=0.2; x=np.arange(len(hs))
    ax.bar(x-1.5*width, [r[4] for r in summary_rows], width, label="both correct")
    ax.bar(x-0.5*width, [r[5] for r in summary_rows], width, label="geometry only")
    ax.bar(x+0.5*width, [r[6] for r in summary_rows], width, label="learned only")
    ax.bar(x+1.5*width, [r[7] for r in summary_rows], width, label="both wrong")
    ax.set_xticks(x); ax.set_xticklabels([str(h) for h in hs])
    ax.set_xlabel("h (mm)"); ax.set_ylabel("count"); ax.legend()
    ax.set_title("Paired outcome composition by h")
    fig.savefig(os.path.join(plots_dir, "paired_outcomes_by_h.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- report ----
    report_path = os.path.join(args.out, "matched_geometry_vs_learned_report.md")
    lines = ["# MATCHED GEOMETRY vs LEARNED CLASSIFIER REPORT\n"]
    lines.append("**Interpretation rule, preserved exactly as instructed:** the "
                "most this evidence can support, if the numbers show it, is: "
                "*\"The learned classifier retains discriminative capability "
                "under spatial representations where explicit geometric fitting "
                "becomes unreliable.\"* This does NOT establish an edge-radar "
                "hardware/cost conclusion -- h is a depth-derived simulated "
                "spatial cell size here, not a validated physical radar "
                "resolution specification.\n\n")
    lines.extend(x if x.startswith("#") else x+"\n" for x in REPORT)
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"\n\nFull report: {report_path}")


if __name__ == "__main__":
    main()