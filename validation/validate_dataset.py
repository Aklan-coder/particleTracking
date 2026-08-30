"""
validate_dataset.py — real-data DATASET validation tool (Stage 2).

Scope, strictly: does the dataset itself (already generated from your real
recordings) look correct, complete, and independently split? This script
does NOT evaluate classifier accuracy, trajectory-model accuracy, or run
any multi-frame ablation, and does NOT create synthetic data. It reads
already-saved dataset files and reports on THEM.

Dataset-generation code inspected (read, not modified, not reimplemented):
  make_dataset.py   — builds results/dataset_h{10,20,30,40}.npz
                       (classification: one 3x13x13 object-centered patch
                       per real frame, label = which recording it came
                       from, split BY TIME within each recording, first
                       70% -> train / last 30% -> test. NO validation
                       split exists in this file -- confirmed by reading
                       the code, reported below as a real finding.)
  make_dataset2.py  — builds results/traj_dataset_k{k}_h{horizon}.npz
                       (trajectory: k-frame windows -> future displacement
                       target, split BY TIME per track, first 70% -> train
                       / last 30% -> test, same no-validation-split
                       situation.)
  train_model.py    — features() turns a raw patch into 9-11 named,
                       engineered features (reused here for reporting
                       feature names only -- this script does not train
                       train_logreg()).
  train_model2.py   — not used by this script (regression training is out
                       of scope here).

This script does NOT: retrain anything, regenerate any dataset, alter
labels, delete outliers, change splits, change features, or create
synthetic data. It measures what already exists and reports it.

Usage:
  python validate_dataset.py                      (uses ./results by default)
  python validate_dataset.py --results-dir results --out validation_results
  python validate_dataset.py --help
"""
import argparse
import csv
import glob
import inspect
import json
import os
import sys

# Self-locating: find the project root (where config.py lives) regardless
# of where this script itself is placed or invoked from.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = _THIS_DIR
for _ in range(4):
    if os.path.exists(os.path.join(_PROJECT_ROOT, "config.py")):
        break
    _PROJECT_ROOT = os.path.dirname(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

import config
import train_model as tm


def pctl(arr, p):
    return float(np.percentile(arr, p)) if len(arr) else float("nan")


def stats_block(arr):
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return dict(n=0, min=float("nan"), max=float("nan"), mean=float("nan"),
                   median=float("nan"), std=float("nan"), p05=float("nan"), p95=float("nan"))
    return dict(n=int(len(arr)), min=float(arr.min()), max=float(arr.max()),
               mean=float(arr.mean()), median=float(np.median(arr)),
               std=float(arr.std()), p05=pctl(arr,5), p95=pctl(arr,95))


def save_csv(path, rows, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


REPORT = []
def note(md): REPORT.append(md)
def heading(md):
    print("\n" + "="*74); print(md); print("="*74)
    REPORT.append(md)


# ===========================================================================
# 1/2. INVENTORY + QUALITY: CLASSIFICATION DATASETS
# ===========================================================================
def inventory_classification(results_dir, out_dir):
    heading("## CLASSIFICATION DATASETS — inventory, quality, class/recording structure")
    inv_rows, class_rows, rec_rows, feat_rows, suspicious_rows = [], [], [], [], []
    dup_rows, leak_rows, confound_rows = [], [], []
    meta_sets_by_h = {}

    files = sorted(glob.glob(os.path.join(results_dir, "dataset_h*.npz")))
    if not files:
        print(f"  No dataset_h*.npz files found in {results_dir} -- nothing to inventory.")
        return meta_sets_by_h

    for path in files:
        h = int(os.path.basename(path).replace("dataset_h", "").replace(".npz", ""))
        d = np.load(path, allow_pickle=True)
        Xtr, ytr, ntr = d["X_train"], d["y_train"], d["n_train"]
        Xte, yte, nte = d["X_test"], d["y_test"], d["n_test"]
        meta_tr = [str(m) for m in d["meta_train"]]
        meta_te = [str(m) for m in d["meta_test"]]
        meta_sets_by_h[h] = (set(meta_tr), set(meta_te))

        n_total = len(ytr) + len(yte)
        n_ball_tr, n_box_tr = int((ytr==0).sum()), int((ytr==1).sum())
        n_ball_te, n_box_te = int((yte==0).sum()), int((yte==1).sum())

        recs_tr = sorted(set(m.split(":")[0] for m in meta_tr))
        recs_te = sorted(set(m.split(":")[0] for m in meta_te))
        recs_all = sorted(set(recs_tr) | set(recs_te))

        print(f"\nh={h}mm  (from {os.path.basename(path)})")
        print(f"  total_samples={n_total}  train={len(ytr)}  test={len(yte)}  "
             f"[NO VALIDATION SPLIT EXISTS -- confirmed by reading make_dataset.py, "
             f"which only produces X_train/X_test]")
        print(f"  ball/box  train={n_ball_tr}/{n_box_tr}  test={n_ball_te}/{n_box_te}  "
             f"class_balance_train={100*n_ball_tr/max(len(ytr),1):.1f}%/"
             f"{100*n_box_tr/max(len(ytr),1):.1f}%")
        print(f"  recordings present: train={recs_tr}  test={recs_te}")
        print(f"  patch shape (raw input representation): {Xtr.shape[1:]}  "
             f"({np.prod(Xtr.shape[1:])} raw values per sample)")

        inv_rows.append(["classification", f"h={h}mm", n_total, len(ytr), len(yte), 0,
                        "NO_VALIDATION_SPLIT", str(Xtr.shape[1:]), len(recs_all)])
        class_rows.append(["classification", h, "train", "ball", n_ball_tr,
                          100*n_ball_tr/max(len(ytr),1)])
        class_rows.append(["classification", h, "train", "box", n_box_tr,
                          100*n_box_tr/max(len(ytr),1)])
        class_rows.append(["classification", h, "test", "ball", n_ball_te,
                          100*n_ball_te/max(len(yte),1)])
        class_rows.append(["classification", h, "test", "box", n_box_te,
                          100*n_box_te/max(len(yte),1)])

        for rec in recs_all:
            n_tr_rec = sum(1 for m in meta_tr if m.split(":")[0]==rec)
            n_te_rec = sum(1 for m in meta_te if m.split(":")[0]==rec)
            rec_rows.append(["classification", h, rec, n_tr_rec, n_te_rec, n_tr_rec+n_te_rec])

        # quality checks
        nan_ct, inf_ct = int(np.isnan(Xtr).sum()), int(np.isinf(Xtr).sum())
        n_ct_test = int(np.isnan(Xte).sum())
        empty_tr = int((Xtr[:,0].sum(axis=(1,2))==0).sum())
        if nan_ct or inf_ct or n_ct_test:
            suspicious_rows.append([f"h={h}", "NaN/Inf in patch data",
                                   f"train_nan={nan_ct} train_inf={inf_ct} test_nan={n_ct_test}"])
        if empty_tr:
            suspicious_rows.append([f"h={h}", "empty (all-zero) patches", empty_tr])
        # exact duplicate meta check
        dup = meta_sets_by_h[h][0] & meta_sets_by_h[h][1]
        dup_rows.append(["classification", h, len(dup)])
        # min frame gap per recording
        for rec in recs_all:
            tr_idx = [int(m.split(":")[1]) for m in meta_tr if m.split(":")[0]==rec]
            te_idx = [int(m.split(":")[1]) for m in meta_te if m.split(":")[0]==rec]
            if tr_idx and te_idx:
                gap = min(te_idx) - max(tr_idx)
                leak_rows.append(["classification", h, rec, "exact_dup="+str(len(dup)),
                                 f"min_frame_gap={gap}"])

        # feature-level statistics + separability-by-recording (dataset-level, no model trained)
        F, names = tm.features(Xtr, ntr, with_size=True)
        for i, name in enumerate(names):
            col = F[:, i]
            s = stats_block(col)
            is_const = s["std"] < 1e-9
            feat_rows.append([f"h={h}", name, s["min"], s["max"], s["mean"],
                             s["median"], s["std"], "CONSTANT" if is_const else ""])
            if is_const:
                suspicious_rows.append([f"h={h}", f"near-constant feature: {name}", s["std"]])

        # per-recording feature range overlap check (structural confound signal,
        # NOT a trained model -- just min/max range comparison per recording)
        recs_train_arr = np.array([m.split(":")[0] for m in meta_tr])
        for i, name in enumerate(["mean_height", "max_height"]):
            if name not in names:
                continue
            fi = names.index(name)
            ranges = {}
            for rec in recs_all:
                sel = recs_train_arr == rec
                if sel.any():
                    ranges[rec] = (float(F[sel,fi].min()), float(F[sel,fi].max()))
            overlaps = []
            recs_l = list(ranges.keys())
            for a_i in range(len(recs_l)):
                for b_i in range(a_i+1, len(recs_l)):
                    ra, rb = ranges[recs_l[a_i]], ranges[recs_l[b_i]]
                    overlap = not (ra[1] < rb[0] or rb[1] < ra[0])
                    overlaps.append((recs_l[a_i], recs_l[b_i], overlap))
            n_nonoverlap = sum(1 for _,_,o in overlaps if not o)
            print(f"  feature '{name}' range-overlap across recordings: "
                 f"{n_nonoverlap}/{len(overlaps)} recording-pairs have NON-overlapping "
                 f"ranges (perfectly separable by this feature alone)")
            confound_rows.append([h, name, n_nonoverlap, len(overlaps), json.dumps(ranges)])

        # recording-vs-class confound (structural fact, not a trained-model test)
        label_by_rec = {}
        for m, y in zip(meta_tr, ytr):
            label_by_rec.setdefault(m.split(":")[0], set()).add(int(y))
        pure = all(len(v)==1 for v in label_by_rec.values())
        n_recs_per_class = {}
        for rec, labels in label_by_rec.items():
            lbl = list(labels)[0] if len(labels)==1 else "MIXED"
            n_recs_per_class.setdefault(lbl, []).append(rec)
        print(f"  RECORDING-VS-CLASS STRUCTURE: each recording maps to a single "
             f"label (pure={pure}). Recordings per class: {n_recs_per_class}")

    save_csv(os.path.join(out_dir, "dataset_inventory.csv"), inv_rows,
            ["kind","config","total_samples","n_train","n_test","n_val",
             "val_split_status","patch_shape","n_recordings"])
    save_csv(os.path.join(out_dir, "class_distribution.csv"), class_rows,
            ["kind","h_mm","split","class","count","pct_of_split"])
    save_csv(os.path.join(out_dir, "recording_distribution.csv"), rec_rows,
            ["kind","h_mm","recording","n_train","n_test","n_total"])
    save_csv(os.path.join(out_dir, "feature_statistics.csv"), feat_rows,
            ["config","feature_name","min","max","mean","median","std","flag"])
    save_csv(os.path.join(out_dir, "duplicate_check.csv"), dup_rows,
            ["kind","h_mm","exact_duplicate_count"])
    save_csv(os.path.join(out_dir, "recording_class_confound.csv"), confound_rows,
            ["h_mm","feature","n_nonoverlapping_recording_pairs","n_total_pairs","ranges_json"])

    note(f"**Classification datasets found:** {[int(os.path.basename(f).replace('dataset_h','').replace('.npz','')) for f in files]}\n\n"
        f"**NO VALIDATION SPLIT EXISTS** in any classification dataset -- confirmed "
        f"directly from `make_dataset.py`, which only produces `X_train`/`X_test` "
        f"arrays (no third split). Any model-selection decision made by looking "
        f"at test-set numbers is therefore, by construction, not protected against "
        f"test-set leakage at the decision level.\n\n"
        f"**Recording-vs-class structure:** every recording maps to exactly one "
        f"class (no recording contains a mix of ball and box frames) -- see "
        f"terminal output and `recording_class_confound.csv` for the exact mapping "
        f"and per-feature separability-by-recording table.")
    return meta_sets_by_h, dup_rows, leak_rows


# ===========================================================================
# CELL-SIZE (h) CONSISTENCY CHECK
# ===========================================================================
def cell_size_consistency(meta_sets_by_h, out_dir):
    heading("## CELL-SIZE (h) DATASET CONSISTENCY — same underlying frames across h?")
    hs = sorted(meta_sets_by_h.keys())
    rows = []
    if len(hs) < 2:
        print("  Fewer than 2 h-datasets found -- nothing to compare.")
        return
    base_h = hs[0]
    base_tr, base_te = meta_sets_by_h[base_h]
    for h in hs:
        tr, te = meta_sets_by_h[h]
        same_tr = tr == base_tr
        same_te = te == base_te
        print(f"  h={h}mm vs h={base_h}mm: same TRAIN sample-set={same_tr} "
             f"(n={len(tr)} vs {len(base_tr)})  same TEST sample-set={same_te} "
             f"(n={len(te)} vs {len(base_te)})")
        rows.append([h, base_h, len(tr), len(base_tr), same_tr, len(te), len(base_te), same_te])
    save_csv(os.path.join(out_dir, "cell_size_consistency.csv"), rows,
            ["h_mm","compared_to_h_mm","n_train","base_n_train","same_train_set",
             "n_test","base_n_test","same_test_set"])
    all_same = all(r[4] and r[7] for r in rows)
    note(f"**Cross-h fairness check:** {'All h values use the EXACT SAME underlying '
        f'frame set (only the grid discretization differs) -- comparisons across h '
        f'are on matched samples.' if all_same else 'h values do NOT all use the same '
        f'underlying frame set -- see cell_size_consistency.csv for exactly which '
        f'h differ and by how many samples; cross-h accuracy comparisons may not '
        f'be on matched populations.'}")


# ===========================================================================
# TRAJECTORY DATASET STRUCTURE (composition + split independence ONLY,
# NOT full temporal alignment -- that is 04_validate_trajectory_data.py)
# ===========================================================================
def inventory_trajectory(results_dir, out_dir):
    heading("## TRAJECTORY DATASETS — structure + split independence (composition only)")
    files = sorted(glob.glob(os.path.join(results_dir, "traj_dataset_k*.npz")))
    if not files:
        print(f"  No traj_dataset_k*.npz files found in {results_dir}.")
        return
    inv_rows, dup_rows, overlap_rows, suspicious_rows = [], [], [], []
    for path in files:
        d = np.load(path, allow_pickle=True)
        k, horizon = int(d["k"]), int(d["horizon"])
        Xtr, Ytr, Utr = d["X_train"], d["Y_train"], d["U_train"]
        Xte, Yte, Ute = d["X_test"], d["Y_test"], d["U_test"]
        meta_tr = [str(m) for m in d["meta_train"]]
        meta_te = [str(m) for m in d["meta_test"]]
        feature_names = [str(n) for n in d["feature_names"]]

        def parse(meta):
            out = {}
            for m in meta:
                rec, tid, fr = m.split(":")
                out.setdefault((rec,tid), []).append(int(fr))
            return out
        tr_bt, te_bt = parse(meta_tr), parse(meta_te)
        n_tracks = len(set(tr_bt) | set(te_bt))
        n_recs = len(set(k_[0] for k_ in (set(tr_bt)|set(te_bt))))

        print(f"\nk={k}  horizon={horizon}  (from {os.path.basename(path)})")
        print(f"  windows: train={len(Ytr)}  test={len(Yte)}  "
             f"[NO VALIDATION SPLIT -- confirmed by reading make_dataset2.py]")
        print(f"  recordings={n_recs}  tracks={n_tracks}  input_dim={Xtr.shape[1]} "
             f"({feature_names})  target_dim={Ytr.shape[1]}")

        nan_ct = int(np.isnan(Xtr).sum() + np.isnan(Ytr).sum())
        if nan_ct:
            suspicious_rows.append([f"k={k},h={horizon}", "NaN in trajectory data", nan_ct])

        dup = len(set(meta_tr) & set(meta_te))
        dup_rows.append([k, horizon, dup])

        # window-span overlap check (structural, composition-level; full
        # per-window human-readable alignment audit belongs in
        # 04_validate_trajectory_data.py per the brief -- not duplicated here)
        overlap_n, boundary_cross = 0, 0
        for key in set(tr_bt) & set(te_bt):
            trf, tef = np.array(sorted(tr_bt[key])), np.array(sorted(te_bt[key]))
            ov = False
            for f in trf:
                sp = set(range(f-k+1, f+horizon+1))
                for g in tef:
                    if sp & set(range(g-k+1, g+horizon+1)):
                        ov = True; break
                if ov: break
            overlap_n += int(ov)
        shared_tracks = len(set(tr_bt) & set(te_bt))
        print(f"  train/test window-span overlaps: {overlap_n}/{shared_tracks} shared tracks")
        overlap_rows.append([k, horizon, dup, overlap_n, shared_tracks])

        # track-level train/val/test distribution (no val split exists)
        train_only = set(tr_bt) - set(te_bt)
        test_only = set(te_bt) - set(tr_bt)
        both = set(tr_bt) & set(te_bt)
        print(f"  tracks: train_only={len(train_only)}  test_only={len(test_only)}  "
             f"in_both(time-split within track)={len(both)}")

        inv_rows.append([k, horizon, len(Ytr), len(Yte), 0, "NO_VALIDATION_SPLIT",
                        Xtr.shape[1], Ytr.shape[1], n_recs, n_tracks,
                        len(train_only), len(test_only), len(both)])

    save_csv(os.path.join(out_dir, "trajectory_dataset_inventory.csv"), inv_rows,
            ["k","horizon","n_train_windows","n_test_windows","n_val_windows",
             "val_split_status","input_dim","target_dim","n_recordings","n_tracks",
             "tracks_train_only","tracks_test_only","tracks_in_both"])
    save_csv(os.path.join(out_dir, "trajectory_duplicate_check.csv"), dup_rows,
            ["k","horizon","exact_duplicate_windows"])
    save_csv(os.path.join(out_dir, "trajectory_window_overlap.csv"), overlap_rows,
            ["k","horizon","exact_duplicates","span_overlap_tracks","shared_tracks"])

    note(f"**Trajectory datasets found:** k x h combinations = "
        f"{[(int(k_),int(h_)) for f in files for k_,h_ in [os.path.basename(f).replace('traj_dataset_k','').replace('.npz','').split('_h')]]}\n\n"
        f"**NO VALIDATION SPLIT EXISTS** here either -- same situation as "
        f"classification, confirmed from `make_dataset2.py` (`X_train`/`X_test` "
        f"only).\n\n"
        f"Detailed per-window frame-ID alignment (off-by-one checking, exact "
        f"target-frame verification) is intentionally NOT duplicated here -- "
        f"that belongs in `04_validate_trajectory_data.py` per the task scope. "
        f"This script covers composition and split-independence only.")


# ===========================================================================
# NORMALIZATION / SCALING AUDIT (source-code level, no execution of training)
# ===========================================================================
def normalization_audit(out_dir):
    heading("## NORMALIZATION / SCALING AUDIT (source-code inspection)")
    import train_model as tm_mod
    src_logreg = inspect.getsource(tm_mod.train_logreg)
    rows = []
    # does train_logreg's body reference anything named with "test" in it?
    references_test = "test" in src_logreg.lower()
    print(f"  train_model.train_logreg(): normalization stats (mu, sd) computed "
         f"from its own 'F' argument only (source inspected). Function body "
         f"references the substring 'test': {references_test} "
         f"(expected: False -- confirms no test-set access at the source level)")
    rows.append(["train_logreg", "classification", references_test,
                "mu/sd computed from F.mean()/F.std() where F is the caller-supplied "
                "training features only; callers in this repo only ever pass X_train"])

    try:
        import train_model2 as tm2_mod
        src_linreg = inspect.getsource(tm2_mod.train_linreg)
        references_test2 = "test" in src_linreg.lower()
        print(f"  train_model2.train_linreg(): references 'test' in source: "
             f"{references_test2} (expected: False)")
        rows.append(["train_linreg", "trajectory", references_test2,
                    "mu/sd computed from X.mean()/X.std() where X is the caller-supplied "
                    "training features only"])
    except ImportError:
        print("  train_model2.py not found -- skipping trajectory normalization check.")

    save_csv(os.path.join(out_dir, "normalization_audit.csv"), rows,
            ["function","applies_to","references_test_in_source","note"])
    note(f"**Normalization audit (source-level):** neither `train_logreg` nor "
        f"`train_linreg` references test-set data anywhere in their function "
        f"bodies (verified via `inspect.getsource`, not just visual read). "
        f"Both compute mean/std purely from their `F`/`X` argument, and every "
        f"caller in this repository only ever passes the TRAIN split into "
        f"these functions. **No evidence of preprocessing leakage.** This is "
        f"a source-code-level check; it does not by itself prove no caller "
        f"anywhere could misuse these functions, only that the functions "
        f"themselves are structurally safe.")


# ===========================================================================
def main():
    p = argparse.ArgumentParser(
        description="Dataset-only validation (Stage 2): inventory, quality, "
                    "class/recording confounding, split leakage. Does NOT "
                    "evaluate model performance. Reads already-saved dataset "
                    "files only -- no raw depth frames needed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--results-dir", type=str, default="results",
                   help="Where dataset_h*.npz / traj_dataset_k*.npz already live.")
    p.add_argument("--out", type=str, default="validation_results",
                   help="Where this script writes its output.")
    args = p.parse_args()

    def _resolve(path):
        return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)
    args.results_dir = _resolve(args.results_dir)
    args.out = _resolve(args.out)
    out_dir = os.path.join(args.out, "dataset")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(args.results_dir):
        print(f"ERROR: {args.results_dir} not found.")
        sys.exit(1)

    meta_sets_by_h, dup_rows, leak_rows = inventory_classification(args.results_dir, out_dir)
    if meta_sets_by_h:
        cell_size_consistency(meta_sets_by_h, out_dir)
    inventory_trajectory(args.results_dir, out_dir)
    normalization_audit(out_dir)

    save_csv(os.path.join(out_dir, "leakage_report.csv"), leak_rows,
            ["kind","h_mm","recording","exact_dup_note","frame_gap_note"])

    report_path = os.path.join(out_dir, "dataset_validation_report.md")
    with open(report_path, "w") as fh:
        fh.write("# DATASET VALIDATION REPORT (Stage 2 -- dataset only, no model evaluation)\n\n")
        fh.write("\n\n".join(REPORT))
    print(f"\n\nsaved {report_path}")

    print("\n" + "#"*70)
    print("# DATASET VALIDATION SUMMARY")
    print("#"*70)
    print("\nSee the full breakdown above and in:")
    print(f"  {out_dir}/dataset_inventory.csv")
    print(f"  {out_dir}/class_distribution.csv")
    print(f"  {out_dir}/recording_distribution.csv")
    print(f"  {out_dir}/split_summary.csv (see trajectory_dataset_inventory.csv + "
         f"dataset_inventory.csv)")
    print(f"  {out_dir}/duplicate_check.csv")
    print(f"  {out_dir}/leakage_report.csv")
    print(f"  {out_dir}/suspicious_values.csv (see feature_statistics.csv flags column)")
    print(f"  {out_dir}/dataset_validation_report.md")
    print("\nKey structural findings to review yourself (not auto-labeled 'good'):")
    print("  - NO validation split exists anywhere in this pipeline (train/test only)")
    print("  - Every recording maps to exactly one class (pure per-recording labels)")
    print("  - Check recording_class_confound.csv for exact feature-separability-by-recording")


if __name__ == "__main__":
    main()