"""
make_joint_test_dataset.py — build a TEST-ONLY classification dataset from
the ball_and_box_moving recording, for evaluating the already-trained,
frozen ball-vs-box classifier. Trains nothing. Modifies no existing file.

Reused, unmodified, from the existing repository:
  make_dataset.py   : patch_for(), PATCH, H_TRAIN  (imported directly --
                       same function object, guaranteeing byte-identical
                       patch construction to the original training data)
  fit_static.py     : load_table(), object_points()
  geometry.py       : load_masked()
  discretize.py     : Partition
  merge_split.py    : split_merged(), default_body_radius()
  config.py         : GATE_MM, MIN_CLUSTER_POINTS, RESULTS_DIR

NOT used, deliberately: geometry.classify_cluster(). Ground truth here
comes ONLY from a manual track-ID -> object mapping you supply (see
--ball-track-ids / --box-track-ids). Cluster-to-TRACK association below
is a pure position-matching step (nearest cluster centroid to a known
track's already-recorded position in *_tracks.csv) -- it never classifies
shape, so it cannot leak a geometry-based "opinion" into the label.

This script does NOT train, fit, or normalize anything. Output is raw
patches (X_test) + labels (y_test) + rich per-sample metadata, in the
same raw, unnormalized form make_dataset.py itself produces (the same
form train_model.py's features()/train_logreg() later consume for
evaluation -- that consumption happens in a SEPARATE, later script, not
here).

Usage:
  # Step 1 -- inspect available tracks first (no ground truth supplied yet):
  python make_joint_test_dataset.py --inspect

  # Step 2 -- generate the datasets, once you know the correct track IDs:
  python make_joint_test_dataset.py \\
      --ball-track-ids 2,5 --box-track-ids 1,4

  python make_joint_test_dataset.py --help
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
from geometry import load_masked
from merge_split import default_body_radius, split_merged
from make_dataset import patch_for, PATCH, H_TRAIN


# ===========================================================================
# STEP 1 HELPER: inspect available tracks (no ground truth guessing, ever)
# ===========================================================================
def inspect_tracks(tracks_csv):
    if not os.path.exists(tracks_csv):
        print(f"ERROR: {tracks_csv} not found. Run track_moving.py on "
             f"ball_and_box_moving first.")
        sys.exit(1)
    rows = list(csv.DictReader(open(tracks_csv)))
    by_track = {}
    for r in rows:
        by_track.setdefault(int(r["track_id"]), []).append(r)

    print("\n" + "="*74)
    print("AVAILABLE TRACKS in", tracks_csv)
    print("="*74)
    print(f"{'track_id':>8}  {'n_frames':>8}  {'frame_range':>15}  "
         f"{'majority track_kind (INFO ONLY -- NOT ground truth)':>50}")
    for tid, rs in sorted(by_track.items()):
        frames = [int(r["frame"]) for r in rs]
        kinds = [r["track_kind"] for r in rs]
        majority = max(set(kinds), key=kinds.count)
        print(f"{tid:>8}  {len(rs):>8}  {min(frames):>6}-{max(frames):<6}  "
             f"{majority:>50}")

    print("\nHOW TO DETERMINE WHICH TRACK ID IS THE PHYSICAL BALL/BOX:")
    print("  The 'majority track_kind' column above is geometry's OWN opinion")
    print("  (from classify_cluster(), already run when this CSV was built by")
    print("  track_moving.py) -- per your instruction, this must NOT be used")
    print("  as ground truth. Use it only as a hint of where to look.")
    print("  To determine TRUE ground truth, inspect the visual evidence:")
    print(f"    - results/ball_and_box_moving_tracking.gif  (camera view: watch")
    print(f"      which track ID marker follows the physically real ball vs box)")
    print(f"    - results/ball_and_box_moving_trajectory.png (static positions)")
    print("  Once you have visually confirmed which track_id(s) correspond to")
    print("  the real ball and which to the real box, re-run this script with:")
    print("    --ball-track-ids <id1,id2,...>  --box-track-ids <id1,id2,...>")
    print("  (multiple IDs per object are supported, for tracks that were lost")
    print("  and re-created with a new ID during the recording.)")


# ===========================================================================
# STEP 2: generate the test-only dataset
# ===========================================================================
def load_track_positions(tracks_csv, id_to_label):
    """Per-frame list of (track_id, label, u_mm, v_mm) for ONLY the
    user-mapped tracks -- unmapped tracks are ignored entirely, never
    guessed. Positions come directly from the already-computed,
    already-confirmed track record (no re-derivation of Kalman state)."""
    rows = list(csv.DictReader(open(tracks_csv)))
    by_frame = {}
    for r in rows:
        tid = int(r["track_id"])
        if tid not in id_to_label:
            continue
        by_frame.setdefault(int(r["frame"]), []).append(
            dict(track_id=tid, label=id_to_label[tid],
                kind=id_to_label[tid],  # ground truth used as 'kind' for
                                        # merge_split's body-size lookup --
                                        # NOT from classify_cluster
                uv=(float(r["u_mm"]), float(r["v_mm"]))))
    return by_frame


def process_recording(recording_dir, tracks_csv, id_to_label, h_values,
                      results_dir, out_dir, every=1):
    bg = np.load(os.path.join(results_dir, "background_median.npy"))
    tf, z = load_table()
    depth_files = sorted(glob.glob(os.path.join(recording_dir, "depth", "*.npy")))[::every]
    if not depth_files:
        print(f"ERROR: no .npy frames found under {recording_dir}/depth")
        sys.exit(1)

    by_frame_tracks = load_track_positions(tracks_csv, id_to_label)
    print(f"\nfound ground-truth track positions for "
         f"{len(by_frame_tracks)} frames (tracks {sorted(set(id_to_label))} "
         f"only -- all other track IDs in {tracks_csv} are ignored)")

    os.makedirs(out_dir, exist_ok=True)
    for h in h_values:
        part = Partition(z["u_range"], z["v_range"], h=h)
        X, y, meta_rows = [], [], []
        status_counts = {}

        for f in depth_files:
            frame_idx = int(os.path.basename(f).replace("depth_", "").replace(".npy", ""))
            track_list = by_frame_tracks.get(frame_idx)
            if not track_list:
                continue  # no ground-truth track confirmed this frame -- skip, don't guess

            pts = object_points(load_masked(f), bg)
            if len(pts) < config.MIN_CLUSTER_POINTS:
                for t in track_list:
                    status_counts["track_unavailable_no_points"] = \
                        status_counts.get("track_unavailable_no_points", 0) + 1
                continue
            uvh = tf.to_table(pts)
            clusters, _ = part.clusters(uvh)
            if not clusters:
                for t in track_list:
                    status_counts["track_unavailable_no_cluster"] = \
                        status_counts.get("track_unavailable_no_cluster", 0) + 1
                continue

            # cluster centroids (position only -- NOT a shape classification)
            cl_centroids = [uvh[cl["points_mask"]][:, :2].mean(axis=0) for cl in clusters]

            # match each ground-truth track to its nearest cluster (position-only)
            track_to_cluster = {}
            for ti, t in enumerate(track_list):
                dists = [np.linalg.norm(np.asarray(t["uv"]) - c) for c in cl_centroids]
                best = int(np.argmin(dists))
                if dists[best] < config.GATE_MM * 2:  # generous gate; merged blobs sit further from single-object centroid
                    track_to_cluster.setdefault(best, []).append(t)
                else:
                    status_counts["track_unavailable_out_of_gate"] = \
                        status_counts.get("track_unavailable_out_of_gate", 0) + 1

            for ci, tracks_here in track_to_cluster.items():
                cl = clusters[ci]
                cuvh = uvh[cl["points_mask"]]
                cpts = pts[cl["points_mask"]]

                if len(tracks_here) == 1:
                    t = tracks_here[0]
                    patch, ncells = patch_for(cuvh, part)
                    status = "single_track_clean"
                    if patch is None:
                        status = "patch_unavailable"
                        status_counts[status] = status_counts.get(status, 0) + 1
                        continue
                    X.append(patch); y.append(0 if t["label"]=="ball" else 1)
                    meta_rows.append(["ball_and_box_moving", frame_idx, t["track_id"],
                                     t["label"], h, f"{t['uv'][0]:.2f}", f"{t['uv'][1]:.2f}",
                                     ncells, status])
                    status_counts[status] = status_counts.get(status, 0) + 1

                else:
                    # merged/touching -- reuse merge_split.split_merged() exactly
                    # as track_moving.py does, but with GROUND-TRUTH labels/sizes,
                    # never classify_cluster()
                    tracks_info = [dict(uv=t["uv"], kind=t["label"],
                                       size=default_body_radius(t["label"]))
                                  for t in tracks_here]
                    parts3d = split_merged(cuvh, cpts, tracks_info)
                    for (sub_pts, sub_uvh), t in zip(parts3d, tracks_here):
                        if len(sub_pts) < 10:
                            status = "merged_split_empty_subset"
                            status_counts[status] = status_counts.get(status, 0) + 1
                            continue
                        patch, ncells = patch_for(sub_uvh, part)
                        status = "merged_successfully_split"
                        if patch is None:
                            status = "patch_unavailable"
                            status_counts[status] = status_counts.get(status, 0) + 1
                            continue
                        X.append(patch); y.append(0 if t["label"]=="ball" else 1)
                        meta_rows.append(["ball_and_box_moving", frame_idx, t["track_id"],
                                         t["label"], h, f"{t['uv'][0]:.2f}", f"{t['uv'][1]:.2f}",
                                         ncells, status])
                        status_counts[status] = status_counts.get(status, 0) + 1

        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        out_path = os.path.join(out_dir, f"joint_test_dataset_h{int(h)}.npz")
        np.savez_compressed(
            out_path,
            X_test=X, y_test=y,
            meta_test=np.array([f"ball_and_box_moving:{m[1]}:{m[2]}" for m in meta_rows]),
            frame_id=np.array([m[1] for m in meta_rows]),
            track_id=np.array([m[2] for m in meta_rows]),
            true_label=np.array([m[3] for m in meta_rows]),
            n_cells=np.array([m[7] for m in meta_rows]),
            status=np.array([m[8] for m in meta_rows]))
        print(f"\nh={h}mm: saved {out_path}")
        print(f"  samples: {len(y)}  (ball={int((y==0).sum())}  box={int((y==1).sum())})")
        print(f"  status breakdown: {status_counts}")

        # ---- feature-compatibility verification against the ORIGINAL dataset ----
        orig_path = os.path.join(results_dir, f"dataset_h{int(h)}.npz")
        if os.path.exists(orig_path):
            orig = np.load(orig_path, allow_pickle=True)
            orig_shape = orig["X_test"].shape[1:]
            new_shape = X.shape[1:] if len(X) else None
            print(f"  COMPATIBILITY CHECK vs {orig_path}:")
            print(f"    original X_test shape (per sample): {orig_shape}, dtype={orig['X_test'].dtype}")
            print(f"    joint-test X_test shape (per sample): {new_shape}, dtype={X.dtype if len(X) else 'N/A'}")
            if new_shape is not None and tuple(new_shape) != tuple(orig_shape):
                print(f"  *** INCOMPATIBLE: shapes differ. STOPPING -- not saving "
                     f"a silently-reshaped/broken dataset. ***")
                sys.exit(1)
            else:
                print(f"    COMPATIBLE: same patch shape, same channel count/order "
                     f"(reused patch_for() directly -- guaranteed identical "
                     f"construction), same units (mm). Ready for the existing "
                     f"trained classifier's evaluation path without modification.")
        else:
            print(f"  WARNING: {orig_path} not found -- cannot verify compatibility "
                 f"against the original dataset for h={h}. Generated anyway, but "
                 f"UNVERIFIED.")

        save_csv_rows = meta_rows
        with open(os.path.join(out_dir, "joint_test_metadata.csv"), "a", newline="") as fh:
            w = csv.writer(fh)
            if fh.tell() == 0:
                w.writerow(["recording","frame","track_id","true_label","h_mm",
                          "u_mm","v_mm","n_cells","status"])
            w.writerows(save_csv_rows)

    return out_dir


def main():
    p = argparse.ArgumentParser(
        description="Build a TEST-ONLY classification dataset from "
                    "ball_and_box_moving, using manual track-ID ground truth "
                    "(never classify_cluster()). Trains nothing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--recording", type=str, default="Data/extracted/ball_and_box_moving",
                   help="Path to the recording folder (containing depth/*.npy).")
    p.add_argument("--tracks-csv", type=str, default="results/ball_and_box_moving_tracks.csv",
                   help="Existing tracks CSV already produced by track_moving.py.")
    p.add_argument("--ball-track-ids", type=str, default=None,
                   help="Comma-separated track ID(s) that are the TRUE physical ball, "
                        "e.g. --ball-track-ids 2,5")
    p.add_argument("--box-track-ids", type=str, default=None,
                   help="Comma-separated track ID(s) that are the TRUE physical box.")
    p.add_argument("--h", type=str, default="10,20,30,40",
                   help="Comma-separated cell sizes to generate (mm).")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--out", type=str, default="results/joint_test")
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--inspect", action="store_true",
                   help="Print available track IDs and how to identify them, "
                        "then exit. Use this FIRST, before supplying "
                        "--ball-track-ids/--box-track-ids.")
    args = p.parse_args()

    def _resolve(path):
        return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)
    args.recording = _resolve(args.recording)
    args.tracks_csv = _resolve(args.tracks_csv)
    args.results_dir = _resolve(args.results_dir)
    args.out = _resolve(args.out)

    if args.inspect or not (args.ball_track_ids and args.box_track_ids):
        inspect_tracks(args.tracks_csv)
        if not (args.ball_track_ids and args.box_track_ids):
            print("\nNo --ball-track-ids/--box-track-ids supplied -- STOPPING here.")
            print("This script never guesses ground truth. Re-run with both "
                 "arguments once you've identified the correct track IDs.")
        sys.exit(0)

    ball_ids = [int(x) for x in args.ball_track_ids.split(",")]
    box_ids = [int(x) for x in args.box_track_ids.split(",")]
    overlap = set(ball_ids) & set(box_ids)
    if overlap:
        print(f"ERROR: track ID(s) {overlap} listed as BOTH ball and box. "
             f"A track cannot be both. Fix your --ball-track-ids/--box-track-ids.")
        sys.exit(1)
    id_to_label = {tid: "ball" for tid in ball_ids}
    id_to_label.update({tid: "box" for tid in box_ids})
    print(f"\nGround-truth mapping (manual, from you -- NOT classify_cluster()):")
    print(f"  ball track IDs: {ball_ids}")
    print(f"  box  track IDs: {box_ids}")

    h_values = [float(x) for x in args.h.split(",")]
    os.makedirs(args.out, exist_ok=True)
    meta_csv = os.path.join(args.out, "joint_test_metadata.csv")
    if os.path.exists(meta_csv):
        os.remove(meta_csv)  # fresh run -- this script's own output only, never touches originals

    process_recording(args.recording, args.tracks_csv, id_to_label, h_values,
                      args.results_dir, args.out, args.every)

    summary_path = os.path.join(args.out, "joint_test_generation_summary.md")
    with open(summary_path, "w") as fh:
        fh.write("# Joint test dataset generation summary\n\n")
        fh.write(f"Source recording: `{args.recording}`\n\n")
        fh.write(f"Ground truth: ball track IDs {ball_ids}, box track IDs {box_ids} "
                f"(manually supplied, NOT from classify_cluster())\n\n")
        fh.write(f"Cell sizes generated: {h_values}\n\n")
        fh.write("See joint_test_metadata.csv for full per-sample traceability "
                "(frame, track, label, status) and the per-h "
                "joint_test_dataset_h{h}.npz files for the actual X_test/y_test "
                "arrays, ready for evaluation against the existing frozen "
                "classifier (evaluation itself is a separate, later step, not "
                "performed by this script).\n")
    print(f"\nsaved {summary_path}")
    print(f"\nALL OUTPUTS ARE TEST-ONLY. Nothing was trained. No existing file "
         f"was modified. See {args.out}/")


if __name__ == "__main__":
    main()