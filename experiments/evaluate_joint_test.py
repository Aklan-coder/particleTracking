"""
evaluate_joint_test.py — evaluate the EXISTING, FROZEN ball-vs-box
classifier on the completely unseen ball_and_box_moving recording.
Trains nothing. Fits nothing. Modifies no existing file.

INSPECTION FINDINGS (from train_model.py, make_dataset.py, eval_model.py):
  - Classifier: plain-numpy logistic regression, one PER cell size h
    (h=10,20,30,40), saved at results/model_h{h}_full.npz.
  - Each saved model file already contains w, b, mu, sd (TRAINING-derived
    normalization statistics, saved at training time) -- nothing needs to
    be reconstructed from X_train; the frozen scaler is already on disk.
  - (3,13,13) patch -> feature vector: train_model.features(), imported
    directly here, not reimplemented.
  - Decision rule (probability -> class): eval_model.predict(), imported
    directly here for the official boolean decision, guaranteeing the
    EXACT same decision rule used by your existing evaluation. A separate
    tiny helper below (proba_from_model) computes the identical formula
    WITHOUT the 0.5 threshold, only because ROC-AUC/Brier/log-loss need
    continuous scores -- this is not a new model, it is the same formula
    with the last step omitted.
  - Class encoding, confirmed from make_dataset.py's RECORDINGS dict and
    make_joint_test_dataset.py's labeling: class 0 = ball, class 1 = box.

NO RETRAINING GUARANTEE: this script never calls train_logreg(),
train_linreg(), or any .fit()-like function. It only ever calls
np.load() on already-saved model/dataset files and train_model.features()
(a pure feature-extraction function, not a training function) /
eval_model.predict() (a pure inference function). grep this file yourself
for "train_logreg" or "train_linreg" -- neither appears.

Usage:
  python evaluate_joint_test.py
  python evaluate_joint_test.py --results-dir results --out results/joint_test/evaluation
  python evaluate_joint_test.py --help
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
from train_model import features, H_TRAIN
from eval_model import predict as frozen_predict   # exact existing decision rule


# ===========================================================================
# Small helpers (metrics only -- no model logic reimplemented)
# ===========================================================================
def proba_from_model(model, F):
    """IDENTICAL formula to eval_model.predict(), with the 0.5 threshold
    removed -- needed only because ROC-AUC/Brier/log-loss require a
    continuous score, not a boolean. Not a new model."""
    Z = (F - model["mu"]) / model["sd"]
    return 1.0 / (1.0 + np.exp(-np.clip(Z @ model["w"] + float(model["b"]), -500, 500)))


def confusion(y_true, y_pred):
    tn = int(((y_pred==0)&(y_true==0)).sum())
    fp = int(((y_pred==1)&(y_true==0)).sum())
    fn = int(((y_pred==0)&(y_true==1)).sum())
    tp = int(((y_pred==1)&(y_true==1)).sum())
    return tn, fp, fn, tp


def prf(tp, fp, fn):
    p = tp/(tp+fp) if (tp+fp) else float("nan")
    r = tp/(tp+fn) if (tp+fn) else float("nan")
    f1 = 2*p*r/(p+r) if (p+r) and not np.isnan(p) and not np.isnan(r) and (p+r)>0 else float("nan")
    return p, r, f1


def roc_auc(y_true, proba):
    n_pos, n_neg = int(y_true.sum()), int(len(y_true)-y_true.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(proba)) + 1
    return float((ranks[y_true==1].sum() - n_pos*(n_pos+1)/2) / (n_pos*n_neg))


def brier(y_true, proba):
    return float(np.mean((proba - y_true)**2))


def logloss(y_true, proba, eps=1e-9):
    p = np.clip(proba, eps, 1-eps)
    return float(-np.mean(y_true*np.log(p) + (1-y_true)*np.log(1-p)))


def full_metrics(y_true, y_pred, proba=None):
    tn, fp, fn, tp = confusion(y_true, y_pred)
    tp_box, fp_box, fn_box = tp, fp, fn
    p_box, r_box, f1_box = prf(tp_box, fp_box, fn_box)
    tp_ball, fp_ball, fn_ball = tn, fn, fp   # ball-as-positive counts
    p_ball, r_ball, f1_ball = prf(tp_ball, fp_ball, fn_ball)

    acc = float((y_pred==y_true).mean()) if len(y_true) else float("nan")
    bal_acc = float((r_ball + r_box)/2) if not (np.isnan(r_ball) or np.isnan(r_box)) else float("nan")
    macro_p = float(np.nanmean([p_ball, p_box]))
    macro_r = float(np.nanmean([r_ball, r_box]))
    macro_f1 = float(np.nanmean([f1_ball, f1_box]))

    out = dict(n=len(y_true), n_ball=int((y_true==0).sum()), n_box=int((y_true==1).sum()),
              n_correct=int((y_pred==y_true).sum()), n_incorrect=int((y_pred!=y_true).sum()),
              accuracy=acc, balanced_accuracy=bal_acc,
              tn=tn, fp=fp, fn=fn, tp=tp,
              ball_precision=p_ball, ball_recall=r_ball, ball_f1=f1_ball,
              box_precision=p_box, box_recall=r_box, box_f1=f1_box,
              macro_precision=macro_p, macro_recall=macro_r, macro_f1=macro_f1)
    if proba is not None and len(set(y_true.tolist())) > 1:
        out["roc_auc"] = roc_auc(y_true, proba)
        out["brier"] = brier(y_true, proba)
        out["logloss"] = logloss(y_true, proba)
    else:
        out["roc_auc"] = out["brier"] = out["logloss"] = float("nan")
    return out


def save_csv(path, rows, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


# ===========================================================================
def main():
    p = argparse.ArgumentParser(
        description="Evaluate the EXISTING FROZEN ball-vs-box classifier on "
                    "the unseen ball_and_box_moving recording. Trains nothing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--joint-dir", type=str, default="results/joint_test")
    p.add_argument("--out", type=str, default="results/joint_test/evaluation")
    args = p.parse_args()

    def _resolve(path):
        return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)
    args.results_dir = _resolve(args.results_dir)
    args.joint_dir = _resolve(args.joint_dir)
    args.out = _resolve(args.out)
    plots_dir = os.path.join(args.out, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    all_metrics_rows, comparison_rows, all_pred_rows, all_fail_rows = [], [], [], []
    dup_rows, status_rows = [], []
    h_summary = {}

    print("="*78)
    print("JOINT UNSEEN CLASSIFICATION TEST")
    print("="*78)

    for h in H_TRAIN:
        model_path = os.path.join(args.results_dir, f"model_h{h}_full.npz")
        orig_path = os.path.join(args.results_dir, f"dataset_h{h}.npz")
        joint_path = os.path.join(args.joint_dir, f"joint_test_dataset_h{int(h)}.npz")

        if not os.path.exists(model_path):
            print(f"\nh={h}mm: STOP -- {model_path} not found. Cannot evaluate "
                 f"without the frozen model. Not skipping silently.")
            continue
        model = np.load(model_path, allow_pickle=True)
        required = ["w", "b", "mu", "sd"]
        missing = [k for k in required if k not in model]
        if missing:
            print(f"\nh={h}mm: STOP -- model file is missing required keys "
                 f"{missing}. Cannot guarantee frozen, no-retrain evaluation. "
                 f"NOT retraining to fill the gap.")
            continue

        if not os.path.exists(joint_path):
            print(f"\nh={h}mm: STOP -- {joint_path} not found. Run "
                 f"make_joint_test_dataset.py first.")
            continue
        joint = np.load(joint_path, allow_pickle=True)

        # ---- SECTION 4: compatibility verification ----
        Xj = joint["X_test"]
        if len(Xj) == 0:
            print(f"\nh={h}mm: STOP -- joint test dataset has 0 samples. "
                 f"Nothing to evaluate.")
            continue
        yj = joint["y_test"]
        if os.path.exists(orig_path):
            orig = np.load(orig_path, allow_pickle=True)
            orig_shape = orig["X_test"].shape[1:]
            if tuple(Xj.shape[1:]) != tuple(orig_shape):
                print(f"\nh={h}mm: STOP -- INCOMPATIBLE shapes. original="
                     f"{orig_shape}  joint={Xj.shape[1:]}. Not reshaping. "
                     f"Fix the dataset generator instead.")
                continue
        nan_ct, inf_ct = int(np.isnan(Xj).sum()), int(np.isinf(Xj).sum())
        if nan_ct or inf_ct:
            print(f"\nh={h}mm: STOP -- joint test data contains NaN={nan_ct} "
                 f"Inf={inf_ct}. Not silently repairing. Fix upstream.")
            continue
        if Xj.dtype != np.float32:
            print(f"  NOTE: joint X dtype is {Xj.dtype}, not float32 as "
                 f"expected -- proceeding, numpy will upcast for the "
                 f"arithmetic below, but flagging per your instruction.")

        # ---- SECTION 5: duplicate physical-object check ----
        frame_id = joint["frame_id"]
        true_label_str = joint["true_label"]
        n_joint = len(frame_id)
        seen = {}
        dup_frames = []
        for i in range(n_joint):
            key = (int(frame_id[i]), str(true_label_str[i]))
            seen.setdefault(key, []).append(i)
        for key, idxs in seen.items():
            if len(idxs) > 1:
                dup_frames.append(key[0])
                dup_rows.append([h, key[0], key[1], len(idxs),
                                str(joint["track_id"][idxs].tolist()),
                                str(joint["status"][idxs].tolist())])
        n_dup = len(dup_frames)
        if n_dup:
            print(f"\nh={h}mm: DUPLICATE PHYSICAL-OBJECT CHECK: {n_dup} "
                 f"(frame, true_object) pairs appear MORE THAN ONCE. "
                 f"Affected frames: {sorted(set(dup_frames))[:20]}"
                 f"{' ...' if len(set(dup_frames))>20 else ''}. "
                 f"See duplicate_check.csv for full detail including which "
                 f"track IDs and extraction statuses produced each duplicate "
                 f"(most likely explanation to check yourself: the track "
                 f"1->track 4 handoff window, where both IDs may be briefly "
                 f"confirmed in the same frame). NOT auto-corrected -- both "
                 f"copies are still counted below unless you decide "
                 f"otherwise; this is reported, not silently fixed.")
        else:
            print(f"\nh={h}mm: duplicate physical-object check: 0 duplicates found.")

        # ---- predictions (frozen model, no fitting) ----
        if "n_cells" not in joint:
            print(f"\nh={h}mm: STOP -- joint_test_dataset_h{int(h)}.npz has no "
                 f"'n_cells' field, needed for the 'occupied_cells' feature "
                 f"column. Not substituting a placeholder value.")
            continue
        F, _ = features(Xj, joint["n_cells"], with_size=True)
        y_pred = frozen_predict(model, F).astype(int)     # exact existing decision rule
        proba = proba_from_model(model, F)                # same formula, continuous

        m = full_metrics(yj, y_pred, proba)
        all_metrics_rows.append([h] + list(m.values()))

        # per-sample predictions
        track_id = joint["track_id"]; status = joint["status"]
        for i in range(n_joint):
            all_pred_rows.append([h, int(frame_id[i]), int(track_id[i]),
                                 str(true_label_str[i]), int(yj[i]),
                                 "ball" if y_pred[i]==0 else "box", int(y_pred[i]),
                                 int(y_pred[i]==yj[i]), f"{proba[i]:.6f}",
                                 str(status[i])])
            if y_pred[i] != yj[i]:
                all_fail_rows.append([h, int(frame_id[i]), int(track_id[i]),
                                     str(true_label_str[i]), "ball" if y_pred[i]==0 else "box",
                                     f"{proba[i]:.6f}", str(status[i])])

        # ---- SECTION 8: by object ----
        ball_mask, box_mask = yj==0, yj==1
        print(f"  N={m['n']}  Ball={m['n_ball']}  Box={m['n_box']}")
        print(f"  Accuracy: {m['accuracy']:.4f}   Balanced accuracy: {m['balanced_accuracy']:.4f}")
        print(f"  Ball  precision={m['ball_precision']:.4f} recall={m['ball_recall']:.4f} f1={m['ball_f1']:.4f}")
        print(f"  Box   precision={m['box_precision']:.4f} recall={m['box_recall']:.4f} f1={m['box_f1']:.4f}")
        print(f"  Macro F1: {m['macro_f1']:.4f}   ROC-AUC: {m['roc_auc']:.4f}")

        # ---- SECTION 9: by extraction condition ----
        clean_mask = status == "single_track_clean"
        merged_mask = status == "merged_successfully_split"
        clean_acc = float((y_pred[clean_mask]==yj[clean_mask]).mean()) if clean_mask.sum() else float("nan")
        merged_acc = float((y_pred[merged_mask]==yj[merged_mask]).mean()) if merged_mask.sum() else float("nan")
        print(f"  Clean accuracy (n={int(clean_mask.sum())}): {clean_acc:.4f}")
        print(f"  Merged/split accuracy (n={int(merged_mask.sum())}): {merged_acc:.4f}")
        status_rows.append([h, "single_track_clean", int(clean_mask.sum()), clean_acc])
        status_rows.append([h, "merged_successfully_split", int(merged_mask.sum()), merged_acc])

        # ---- SECTION 11: original within-recording comparison (reproduced, not retrained) ----
        orig_acc = orig_macro_f1 = float("nan")
        if os.path.exists(orig_path):
            Fo, _ = features(orig["X_test"], orig["n_test"], with_size=True)
            y_pred_o = frozen_predict(model, Fo).astype(int)
            proba_o = proba_from_model(model, Fo)
            mo = full_metrics(orig["y_test"], y_pred_o, proba_o)
            orig_acc, orig_macro_f1 = mo["accuracy"], mo["macro_f1"]
        diff = m["accuracy"] - orig_acc if not np.isnan(orig_acc) else float("nan")
        print(f"  Original within-recording accuracy: {orig_acc:.4f}")
        print(f"  Unseen joint-recording accuracy:     {m['accuracy']:.4f}")
        print(f"  Difference:                          {diff:+.4f}")
        comparison_rows.append([h, orig_acc, m["accuracy"], diff, orig_macro_f1, m["macro_f1"]])

        h_summary[h] = dict(m=m, orig_acc=orig_acc, clean_acc=clean_acc, merged_acc=merged_acc)

    # ---- write all CSVs ----
    metrics_header = ["h_mm"] + list(full_metrics(np.array([0,1]), np.array([0,1]), np.array([0.1,0.9])).keys())
    save_csv(os.path.join(args.out, "joint_test_metrics.csv"), all_metrics_rows, metrics_header)
    save_csv(os.path.join(args.out, "original_vs_joint_comparison.csv"), comparison_rows,
            ["h_mm","original_within_recording_accuracy","unseen_joint_accuracy",
             "difference","original_macro_f1","unseen_macro_f1"])
    save_csv(os.path.join(args.out, "joint_test_predictions.csv"), all_pred_rows,
            ["h_mm","frame_id","track_id","true_object","true_label_numeric",
             "predicted_object","predicted_label_numeric","correct","probability_box","status"])
    save_csv(os.path.join(args.out, "joint_test_failures.csv"), all_fail_rows,
            ["h_mm","frame_id","track_id","true_object","predicted_object","probability_box","status"])
    save_csv(os.path.join(args.out, "duplicate_check.csv"), dup_rows,
            ["h_mm","frame_id","true_object","n_occurrences","track_ids","statuses"])
    save_csv(os.path.join(args.out, "performance_by_status.csv"), status_rows,
            ["h_mm","status","n","accuracy"])

    # ---- plots ----
    if h_summary:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        hs = sorted(h_summary.keys())
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(hs, [h_summary[h]["orig_acc"] for h in hs], "o-", label="original within-recording")
        ax.plot(hs, [h_summary[h]["m"]["accuracy"] for h in hs], "s-", label="unseen joint recording")
        ax.axhline(0.5, color="grey", ls=":", label="chance (0.5)")
        ax.set_xlabel("cell size h (mm)"); ax.set_ylabel("accuracy"); ax.set_ylim(-0.02,1.05)
        ax.set_title("Accuracy vs h: original vs unseen joint recording")
        ax.legend(); ax.grid(alpha=0.3)
        fig.savefig(os.path.join(plots_dir, "accuracy_vs_h.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(hs, [h_summary[h]["m"]["macro_f1"] for h in hs], "o-", color="tab:purple")
        ax.set_xlabel("cell size h (mm)"); ax.set_ylabel("macro F1")
        ax.set_title("Unseen joint recording: macro F1 vs h")
        ax.grid(alpha=0.3)
        fig.savefig(os.path.join(plots_dir, "macro_f1_vs_h.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(hs, [h_summary[h]["clean_acc"] for h in hs], "o-", label="clean/separated")
        ax.plot(hs, [h_summary[h]["merged_acc"] for h in hs], "s-", label="merged, successfully split")
        ax.set_xlabel("cell size h (mm)"); ax.set_ylabel("accuracy"); ax.set_ylim(-0.02,1.05)
        ax.set_title("Unseen joint recording: clean vs merged/split accuracy")
        ax.legend(); ax.grid(alpha=0.3)
        fig.savefig(os.path.join(plots_dir, "clean_vs_merged_accuracy.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        for h in hs:
            m = h_summary[h]["m"]
            fig, ax = plt.subplots(figsize=(5,4.5))
            cm = np.array([[m["tn"], m["fp"]],[m["fn"], m["tp"]]])
            ax.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=14)
            ax.set_xticks([0,1]); ax.set_xticklabels(["Pred Ball","Pred Box"])
            ax.set_yticks([0,1]); ax.set_yticklabels(["Actual Ball","Actual Box"])
            ax.set_title(f"Confusion matrix, h={h}mm (unseen joint recording)")
            fig.savefig(os.path.join(plots_dir, f"confusion_matrix_h{int(h)}.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

    # ---- SECTION 17: interpretation, evidence-based, not auto-declared ----
    lines = ["# JOINT UNSEEN CLASSIFICATION TEST -- EVALUATION REPORT\n"]
    lines.append("**Original test**: held-out frames from recordings represented "
                "during training (time-split within ball_static/ball_moving/"
                "box_static/box_moving).\n")
    lines.append("**Joint unseen test**: completely separate ball_and_box_moving "
                "recording, never used during training. These are NOT "
                "equivalent experimental conditions.\n\n")
    if h_summary:
        accs = [h_summary[h]["m"]["accuracy"] for h in sorted(h_summary)]
        origs = [h_summary[h]["orig_acc"] for h in sorted(h_summary)]
        bal_accs = [h_summary[h]["m"]["balanced_accuracy"] for h in sorted(h_summary)]
        ball_f1s = [h_summary[h]["m"]["ball_f1"] for h in sorted(h_summary)]
        box_f1s = [h_summary[h]["m"]["box_f1"] for h in sorted(h_summary)]
        clean_accs = [h_summary[h]["clean_acc"] for h in sorted(h_summary)]
        merged_accs = [h_summary[h]["merged_acc"] for h in sorted(h_summary)]

        diffs = [o-a for o,a in zip(origs,accs) if not (np.isnan(o) or np.isnan(a))]
        lines.append("## Pattern check (evidence-based, thresholds stated explicitly)\n")
        if diffs and max(diffs) < 0.03:
            lines.append(f"- **Pattern A supported**: unseen accuracy stays within 0.03 "
                        f"of original at every h (max drop observed: {max(diffs):.4f}).\n")
        elif diffs and max(diffs) < 0.20 and min(bal_accs) > 0.65:
            lines.append(f"- **Pattern B supported**: accuracy decreases (up to "
                        f"{max(diffs):.4f}) but balanced accuracy stays above 0.65 "
                        f"at every h -- substantially above chance.\n")
        if bal_accs and min(bal_accs) < 0.60:
            lines.append(f"- **Pattern C supported at some h**: balanced accuracy "
                        f"drops to {min(bal_accs):.4f} at h={sorted(h_summary)[bal_accs.index(min(bal_accs))]}mm "
                        f"-- approaching chance-level (0.5).\n")
        f1_gaps = [abs(a-b) for a,b in zip(ball_f1s, box_f1s) if not (np.isnan(a) or np.isnan(b))]
        if f1_gaps and max(f1_gaps) > 0.15:
            lines.append(f"- **Pattern D supported**: ball vs box F1 differs by up to "
                        f"{max(f1_gaps):.4f} at some h -- performance is NOT symmetric "
                        f"between classes.\n")
        clean_v_merged = [c-m_ for c,m_ in zip(clean_accs, merged_accs)
                          if not (np.isnan(c) or np.isnan(m_))]
        if clean_v_merged and max(clean_v_merged) > 0.10:
            lines.append(f"- **Pattern E supported**: clean-sample accuracy exceeds "
                        f"merged/split accuracy by up to {max(clean_v_merged):.4f} at "
                        f"some h -- merging/splitting measurably hurts classification.\n")
        if len(accs) >= 2 and (accs[0]-accs[-1]) > 0.10:
            lines.append(f"- **Pattern F supported**: unseen accuracy drops by "
                        f"{accs[0]-accs[-1]:.4f} from h={sorted(h_summary)[0]}mm to "
                        f"h={sorted(h_summary)[-1]}mm.\n")
        lines.append("\n(Any pattern not listed above was checked and NOT supported "
                    "by the stated threshold on this data -- see the raw numbers in "
                    "joint_test_metrics.csv and original_vs_joint_comparison.csv to "
                    "judge for yourself; these thresholds are reporting aids, not a "
                    "substitute for your own interpretation.)\n")

    lines.append("\n## THIS TEST SET IS NOW FROZEN\n")
    lines.append("Per your instruction: do not tune the classifier using these "
                "results and then re-report performance on this same recording "
                "as independent test performance. Any future model change would "
                "require a NEW independent recording for an unbiased final test.\n")

    report_path = os.path.join(args.out, "joint_test_evaluation_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines))

    print("\n" + "="*78)
    print(f"Full report: {report_path}")
    print("="*78)


if __name__ == "__main__":
    main()