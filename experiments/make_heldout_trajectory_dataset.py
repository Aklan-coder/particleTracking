"""
make_heldout_trajectory_dataset.py — creates a SEPARATE, track-independent
TRAIN/VALIDATION/TEST trajectory dataset, alongside (never overwriting)
the original time-split-within-track proof-of-concept datasets. Does NOT
train anything.

REUSED DIRECTLY (not reimplemented): make_dataset2.load_tracks(),
make_dataset2.window_features(), make_dataset2.FEATURE_NAMES,
make_dataset2.build_examples(). Critically, build_examples() ALREADY
takes a plain list of track-dicts and returns all windows from them --
it does not itself perform any time-split (that happens in
make_dataset2.main(), outside build_examples(), by pre-slicing each
track's array before calling it). This means assigning WHOLE tracks to
train/val/test and calling build_examples() separately on each group is
a direct, correct reuse of the exact production windowing/target logic,
with zero reimplementation.

KNOWN OBJECT IDENTITY (manual, NOT classify_cluster(), per your own
established mapping): every ball_moving track = ball; box_moving track =
box; ball_and_box_moving track 2 = ball, tracks 1 and 4 = box. This
script reads track IDs from the real CSVs at runtime and applies this
mapping -- it does not hardcode track COUNTS (those vary run to run;
verified during development that this script's own sandbox copy of the
data differs from live data, so nothing about track sizes is assumed).

Usage:
  # Step 1 -- see the proposed split, generates NOTHING:
  python make_heldout_trajectory_dataset.py --plan

  # Step 2 -- review/edit the proposed split (written to
  # heldout_track_split_PROPOSED.json), then copy/rename it to
  # heldout_track_split.json once you approve it, and generate:
  python make_heldout_trajectory_dataset.py --split heldout_track_split.json

  python make_heldout_trajectory_dataset.py --help
"""
import argparse
import csv
import json
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
import make_dataset2 as md2

RECORDINGS = ["ball_moving", "box_moving", "ball_and_box_moving"]
K_VALUES = [1, 3, 5, 10]
HORIZON = 10

KNOWN_IDENTITY = {
    ("ball_moving", None): "ball", ("box_moving", None): "box",
    ("ball_and_box_moving", "2"): "ball",
    ("ball_and_box_moving", "1"): "box", ("ball_and_box_moving", "4"): "box",
}
def object_identity(rec, tid):
    tid = str(tid)
    if (rec, tid) in KNOWN_IDENTITY: return KNOWN_IDENTITY[(rec, tid)]
    if (rec, None) in KNOWN_IDENTITY: return KNOWN_IDENTITY[(rec, None)]
    return "unknown"


def load_all_real_tracks(results_dir):
    """Reads the REAL CSVs at runtime -- nothing about track IDs/counts
    is hardcoded anywhere in this script."""
    all_tracks = []
    for rec in RECORDINGS:
        path = os.path.join(results_dir, f"{rec}_tracks.csv")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found -- skipping {rec}.")
            continue
        tracks = md2.load_tracks(rec)
        for t in tracks:
            t["object"] = object_identity(rec, t["id"])
            all_tracks.append(t)
    return all_tracks


def windows_estimate(t, k, horizon):
    """Cheap estimate of usable windows for this track at this k/horizon,
    without running the full production windowing (used only for the
    --plan report's approximate counts)."""
    n = len(t["arr"])
    return max(0, n - k - horizon + 1)


# ===========================================================================
# --plan mode: propose a defensible split, generate nothing
# ===========================================================================
def propose_split(all_tracks):
    """One reasonable, explainable heuristic -- NOT the only possible
    split. Prefers: (1) a TEST track per object type from a DIFFERENT
    recording than that object's majority recording, when such a
    recording exists (maximizes recording-independence where possible);
    (2) a VALIDATION track per object type, distinct from TEST; (3)
    everything else to TRAIN. This is a STARTING SUGGESTION for you to
    review/edit via the JSON override -- not a forced final answer."""
    by_object = {"ball": [], "box": []}
    for t in all_tracks:
        if t["object"] in by_object:
            by_object[t["object"]].append(t)

    plan = {"train": [], "validation": [], "test": []}
    for obj, tracks in by_object.items():
        recs_present = sorted(set(t["recording"] for t in tracks))
        by_rec_count = {r: sum(len(t["arr"]) for t in tracks if t["recording"]==r) for r in recs_present}
        majority_rec = max(by_rec_count, key=by_rec_count.get) if by_rec_count else None

        # TEST: prefer a track from a NON-majority recording (recording-
        # independent test for this object), else the smallest track
        minority_tracks = [t for t in tracks if t["recording"] != majority_rec]
        if minority_tracks:
            test_t = min(minority_tracks, key=lambda t: len(t["arr"]))
        else:
            test_t = min(tracks, key=lambda t: len(t["arr"])) if tracks else None

        remaining = [t for t in tracks if t is not test_t]
        # VALIDATION: smallest remaining track (keeps train as large as possible)
        val_t = min(remaining, key=lambda t: len(t["arr"])) if remaining else None
        remaining2 = [t for t in remaining if t is not val_t]

        if test_t: plan["test"].append(test_t)
        if val_t: plan["validation"].append(val_t)
        plan["train"].extend(remaining2)

    return plan


def print_plan(plan, all_tracks, out_dir):
    print("\n" + "="*78)
    print("# HELD-OUT TRAJECTORY SPLIT PLAN")
    print("="*78)
    split_recs = {}
    for split_name in ["train", "validation", "test"]:
        tracks = plan[split_name]
        print(f"\n{split_name.upper()}")
        for t in sorted(tracks, key=lambda t: (t["recording"], t["id"])):
            print(f"  {t['recording']} track {t['id']} ({t['object']}): {len(t['arr'])} observations")
        n_ball = sum(1 for t in tracks if t["object"]=="ball")
        n_box = sum(1 for t in tracks if t["object"]=="box")
        total_obs = sum(len(t["arr"]) for t in tracks)
        print(f"  Total tracks: {len(tracks)}  Ball tracks: {n_ball}  Box tracks: {n_box}  "
             f"Observations: {total_obs}")
        for k in K_VALUES:
            est = sum(windows_estimate(t, k, HORIZON) for t in tracks)
            print(f"    approx windows at k={k}, horizon={HORIZON}: {est}")
        split_recs[split_name] = set(t["recording"] for t in tracks)

    print("\nTRACK OVERLAP CHECK:")
    ids = {name: set((t["recording"],t["id"]) for t in plan[name]) for name in plan}
    overlap_tv = ids["train"] & ids["validation"]
    overlap_tt = ids["train"] & ids["test"]
    overlap_vt = ids["validation"] & ids["test"]
    print(f"  train ∩ validation: {overlap_tv if overlap_tv else 'empty (OK)'}")
    print(f"  train ∩ test: {overlap_tt if overlap_tt else 'empty (OK)'}")
    print(f"  validation ∩ test: {overlap_vt if overlap_vt else 'empty (OK)'}")

    print("\nRECORDING OVERLAP CHECK:")
    for a in ["train","validation","test"]:
        for b in ["train","validation","test"]:
            if a<b:
                shared = split_recs[a] & split_recs[b]
                if shared:
                    print(f"  {a} and {b} SHARE recording(s) {shared} -- "
                         f"TRACK-INDEPENDENT BUT NOT RECORDING-INDEPENDENT for those recordings.")
    print("\nOBJECT COVERAGE:")
    for name in ["train","validation","test"]:
        objs = set(t["object"] for t in plan[name])
        print(f"  {name}: {sorted(objs)} "
             f"{'(both object types present)' if objs=={'ball','box'} else '(MISSING an object type -- see instruction E/F caveats)'}")

    # write PROPOSED json for review -- NOT auto-approved
    out_json = {name: [[t["recording"], t["id"]] for t in plan[name]] for name in plan}
    proposed_path = os.path.join(out_dir, "..", "..", "heldout_track_split_PROPOSED.json")
    proposed_path = os.path.normpath(proposed_path)
    with open(proposed_path, "w") as fh:
        json.dump(out_json, fh, indent=2)
    print(f"\nProposed split written to: {proposed_path}")
    print("This is a SUGGESTION ONLY -- review it, edit if needed, then copy/rename")
    print("it to heldout_track_split.json and re-run with --split to generate.")
    print("\nSTOPPING -- no datasets generated in --plan mode.")


# ===========================================================================
# Validate a user-supplied (or approved) split JSON
# ===========================================================================
def load_and_validate_split(split_path, all_tracks):
    with open(split_path) as fh:
        raw = json.load(fh)
    by_key = {(t["recording"], str(t["id"])): t for t in all_tracks}

    plan = {"train": [], "validation": [], "test": []}
    seen = set()
    for split_name in ["train", "validation", "test"]:
        for rec, tid in raw.get(split_name, []):
            key = (rec, str(tid))
            if key not in by_key:
                print(f"STOP: {split_path} references track {key} which does "
                     f"not exist in the real track CSVs. Not proceeding.")
                sys.exit(1)
            if key in seen:
                print(f"STOP: track {key} appears in more than one split in "
                     f"{split_path}. Not proceeding.")
                sys.exit(1)
            seen.add(key)
            plan[split_name].append(by_key[key])

    for split_name in ["train", "validation", "test"]:
        if not plan[split_name]:
            print(f"STOP: split '{split_name}' is empty in {split_path}. "
                 f"train/validation/test must all be non-empty. Not proceeding.")
            sys.exit(1)

    total_train_obs = sum(len(t["arr"]) for t in plan["train"])
    if total_train_obs < 200:
        print(f"STOP: TRAIN split has only {total_train_obs} total observations "
             f"-- almost certainly insufficient to fit anything meaningful. "
             f"Not proceeding without an explicit override.")
        sys.exit(1)

    print(f"Split file {split_path} validated: "
         f"train={len(plan['train'])} tracks, validation={len(plan['validation'])} "
         f"tracks, test={len(plan['test'])} tracks. No overlaps. All non-empty.")
    return plan


# ===========================================================================
# Generation (reuses build_examples() directly, per track group)
# ===========================================================================
def generate(plan, out_dir, k_values, horizon):
    os.makedirs(out_dir, exist_ok=True)
    summary_rows = []

    # Section 11: assert zero cross-split track overlap (redundant safety
    # check, already enforced in load_and_validate_split, re-checked here
    # right before generation)
    ids = {name: set((t["recording"],t["id"]) for t in plan[name]) for name in plan}
    assert not (ids["train"] & ids["validation"]), "TRAIN/VALIDATION overlap -- STOP"
    assert not (ids["train"] & ids["test"]), "TRAIN/TEST overlap -- STOP"
    assert not (ids["validation"] & ids["test"]), "VALIDATION/TEST overlap -- STOP"
    print("Verified: zero track overlap across train/validation/test.")

    for k in k_values:
        Xtr, Ytr, Utr, meta_tr, _ = md2.build_examples(plan["train"], k, horizon)
        Xva, Yva, Uva, meta_va, _ = md2.build_examples(plan["validation"], k, horizon)
        Xte, Yte, Ute, meta_te, _ = md2.build_examples(plan["test"], k, horizon)

        # Section 12: verify zero duplicate windows / shared target frames
        # across splits (should be structurally impossible given track
        # separation, verified here rather than merely assumed)
        overlap_tv = set(meta_tr) & set(meta_va)
        overlap_tt = set(meta_tr) & set(meta_te)
        overlap_vt = set(meta_va) & set(meta_te)
        if overlap_tv or overlap_tt or overlap_vt:
            print(f"STOP k={k}: window overlap found across splits "
                 f"(train/val={len(overlap_tv)}, train/test={len(overlap_tt)}, "
                 f"val/test={len(overlap_vt)}) -- this should be structurally "
                 f"impossible with track separation; STOPPING, not saving.")
            sys.exit(1)

        out_path = os.path.join(out_dir, f"heldout_traj_dataset_k{k}_h{horizon}.npz")
        np.savez_compressed(
            out_path,
            F_train=np.array(Xtr, dtype=np.float32), Y_train=np.array(Ytr, dtype=np.float32),
            U_train=np.array(Utr, dtype=np.float32), meta_train=np.array(meta_tr),
            F_val=np.array(Xva, dtype=np.float32), Y_val=np.array(Yva, dtype=np.float32),
            U_val=np.array(Uva, dtype=np.float32), meta_val=np.array(meta_va),
            F_test=np.array(Xte, dtype=np.float32), Y_test=np.array(Yte, dtype=np.float32),
            U_test=np.array(Ute, dtype=np.float32), meta_test=np.array(meta_te),
            feature_names=np.array(md2.FEATURE_NAMES), k=k, horizon=horizon)
        print(f"\nk={k}")
        print(f"  Train windows: {len(Ytr)}   Validation windows: {len(Yva)}   Test windows: {len(Yte)}")
        print(f"  Window overlap across splits: 0 (verified)")
        print(f"  saved {out_path}")

        for split_name, meta, Y in [("train",meta_tr,Ytr), ("validation",meta_va,Yva), ("test",meta_te,Yte)]:
            recs = [m.split(":")[0] for m in meta]
            tids = [m.split(":")[1] for m in meta]
            objs = [object_identity(r,t) for r,t in zip(recs,tids)]
            n_ball = objs.count("ball"); n_box = objs.count("box")
            summary_rows.append([k, split_name, len(Y), n_ball, n_box,
                                ",".join(sorted(set(recs)))])

    save_path = os.path.join(out_dir, "heldout_dataset_summary.csv")
    with open(save_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["k","split","n_windows","n_ball","n_box","recordings"])
        w.writerows(summary_rows)
    print(f"\nsaved {save_path}")

    report_path = os.path.join(out_dir, "heldout_dataset_generation_report.md")
    with open(report_path, "w") as fh:
        fh.write("# HELD-OUT TRAJECTORY DATASET GENERATION REPORT\n\n")
        fh.write("**ORIGINAL POC:** time split WITHIN every track (earlier "
                "portion -> train, later portion -> test). Preserved, "
                "unmodified, at results/traj_dataset_k*_h10.npz.\n\n")
        fh.write("**NEW HELD-OUT EXPERIMENT (this file):** COMPLETE tracks "
                "assigned to train/validation/test -- a track in TEST "
                "contributes zero observations to TRAIN or VALIDATION, and "
                "vice versa. This tests generalization to UNSEEN TRACKS.\n\n")
        fh.write("**This does NOT necessarily test generalization to unseen "
                "RECORDINGS** -- see the recording-overlap check printed "
                "during --plan and re-derivable from the split JSON; state "
                "this explicitly rather than assuming full recording "
                "independence.\n\n")
        fh.write("No model was trained by this script. Normalization (mu/sd) "
                "was NOT computed here -- raw features are stored; the "
                "training script must compute mu/sd from TRAIN ONLY.\n\n")
        fh.write("| k | split | n_windows | n_ball | n_box | recordings |\n")
        fh.write("|---|---|---|---|---|---|\n")
        for r in summary_rows:
            fh.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |\n")
    print(f"saved {report_path}")


# ===========================================================================
def main():
    p = argparse.ArgumentParser(
        description="Create a track-independent TRAIN/VALIDATION/TEST "
                    "trajectory dataset, separate from the original POC "
                    "datasets. Trains nothing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--plan", action="store_true",
                   help="Propose a split and STOP -- generates nothing.")
    p.add_argument("--split", type=str, default=None,
                   help="Path to an approved split JSON (e.g. heldout_track_split.json). "
                        "Required for generation mode.")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--out", type=str, default="results/heldout_trajectory")
    p.add_argument("--k", type=str, default="1,3,5,10")
    p.add_argument("--horizon", type=int, default=HORIZON)
    args = p.parse_args()

    def _resolve(path):
        return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)
    args.results_dir = _resolve(args.results_dir)
    args.out = _resolve(args.out)
    if args.split:
        args.split = _resolve(args.split)

    all_tracks = load_all_real_tracks(args.results_dir)
    if not all_tracks:
        print("No tracks found. Cannot proceed."); sys.exit(1)
    print(f"Loaded {len(all_tracks)} real tracks from {args.results_dir} "
         f"(read live -- nothing hardcoded).")

    if args.plan:
        plan = propose_split(all_tracks)
        os.makedirs(args.out, exist_ok=True)
        print_plan(plan, all_tracks, args.out)
        return

    if not args.split:
        print("ERROR: generation mode requires --split <path to approved JSON>. "
             "Run with --plan first to get a starting proposal.")
        sys.exit(1)
    if not os.path.exists(args.split):
        print(f"ERROR: {args.split} not found.")
        sys.exit(1)

    plan = load_and_validate_split(args.split, all_tracks)
    k_values = [int(x) for x in args.k.split(",")]
    print("\n" + "="*78)
    print("# HELD-OUT TRAJECTORY DATASET GENERATED")
    print("="*78)
    generate(plan, args.out, k_values, args.horizon)
    print(f"\nOutput: {args.out}/")


if __name__ == "__main__":
    main()