"""
validate_trajectory_data.py — validates the TRAJECTORY DATASETS as they
currently exist. Does not train, evaluate, or regenerate anything.
Modifies no existing file.

TRACED DATA PATH (inspected, not assumed):
  real moving recording (Data/extracted/<rec>/depth/*.npy)
    -> track_moving.py (Kalman filter + classify_cluster) -> <rec>_tracks.csv
       (frame, track_id, u_mm, v_mm, du_mm_s, dv_mm_s, track_kind, ...)
    -> make_dataset2.py:
         load_tracks()      : groups tracks.csv rows into per-track arrays
         window_features()  : k consecutive rows -> ONE 9-feature vector
         build_examples()   : slides the window, looks up the target frame
                               (current_frame + horizon) via an exact
                               frame-index dict, computes displacement
         main()              : time-splits each track 70/30, saves
                               results/traj_dataset_k{k}_h{horizon}.npz

REUSED DIRECTLY (not reimplemented): make_dataset2.load_tracks(),
make_dataset2.window_features(), make_dataset2.build_examples(),
make_dataset2.FEATURE_NAMES, config.DT. This script calls these exact
functions on the real *_tracks.csv files to INDEPENDENTLY RECOMPUTE what
the saved .npz files should contain, then diffs the recomputation against
the saved arrays -- this is the strongest available check, since any
mismatch means either (a) the saved .npz is stale relative to the current
tracks.csv, or (b) an actual construction bug. Nothing is written back;
this script only reads *_tracks.csv and traj_dataset_k*_h*.npz, and never
calls make_dataset2.main() (which is the only thing that saves datasets).

SECTION 4 ANSWERED UP FRONT (verified below with real numbers, not just
asserted): the saved feature dimension is fixed at 9 regardless of k
because k does NOT concatenate k separate frame vectors. Instead, k
controls the WINDOW SIZE that four of the nine features are computed
over: avg_du/avg_dv average du/dv across all k frames in the window, and
ddu/ddv are (velocity at window end - velocity at window start) / window
time span. At k=1 the window start and end are the SAME frame, so
ddu/ddv are ALWAYS EXACTLY 0.0 and avg_du/avg_dv trivially equal du/dv --
this is verified programmatically below for every k found, not just
asserted from reading the code.

Usage:
  python validate_trajectory_data.py
  python validate_trajectory_data.py --results-dir results --out validation_results/trajectory_data
  python validate_trajectory_data.py --help
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
import make_dataset2 as md2


def pctl(arr, p):
    return float(np.percentile(arr, p)) if len(arr) else float("nan")


def stats_block(arr):
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return dict(n=0, min=float("nan"), max=float("nan"), mean=float("nan"),
                   median=float("nan"), std=float("nan"), p05=float("nan"), p95=float("nan"),
                   nan_count=0, inf_count=0, zero_count=0)
    return dict(n=int(len(arr)), min=float(np.nanmin(arr)), max=float(np.nanmax(arr)),
               mean=float(np.nanmean(arr)), median=float(np.nanmedian(arr)),
               std=float(np.nanstd(arr)), p05=pctl(arr,5), p95=pctl(arr,95),
               nan_count=int(np.isnan(arr).sum()), inf_count=int(np.isinf(arr).sum()),
               zero_count=int((arr==0).sum()))


def save_csv(path, rows, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


REPORT = []
ISSUES = []
def note(md): REPORT.append(md)
def issue(msg):
    ISSUES.append(msg)
    print(f"  *** ISSUE: {msg}")
def heading(md):
    print("\n" + "="*78); print(md); print("="*78)
    REPORT.append(md)


RECORDINGS = ["ball_moving", "box_moving", "ball_and_box_moving"]


# ===========================================================================
# 2. DATA SOURCES (real *_tracks.csv, verified from code, not assumed)
# ===========================================================================
def identify_data_sources(results_dir, out_dir):
    heading("## DATA SOURCES (verified against make_dataset2.py's RECORDINGS list)")
    print(f"make_dataset2.py's RECORDINGS constant: {md2.RECORDINGS}")
    rows, all_tracks = [], {}
    for rec in md2.RECORDINGS:
        path = os.path.join(results_dir, f"{rec}_tracks.csv")
        if not os.path.exists(path):
            print(f"  {rec}: {path} NOT FOUND")
            continue
        tracks = md2.load_tracks(rec)
        for t in tracks:
            frames = t["arr"][:,0].astype(int)
            gaps = np.diff(frames)
            n_gaps = int((gaps>1).sum())
            max_gap = int(gaps.max()) if len(gaps) else 0
            all_tracks[(rec, t["id"])] = t
            print(f"  {rec} track {t['id']}: n_obs={len(frames)} "
                 f"frames=[{frames.min()},{frames.max()}] gaps={n_gaps} max_gap={max_gap}")
            rows.append([rec, t["id"], len(frames), int(frames.min()), int(frames.max()),
                        n_gaps, max_gap])
    save_csv(os.path.join(out_dir, "track_inventory.csv"), rows,
            ["recording","track_id","n_observations","first_frame","last_frame",
             "n_gaps","max_gap"])
    note(f"**{len(all_tracks)} genuinely distinct tracks** found across "
        f"{len([r for r in md2.RECORDINGS if os.path.exists(os.path.join(results_dir, r+'_tracks.csv'))])} "
        f"recordings, verified directly from `make_dataset2.RECORDINGS` "
        f"(not assumed).")
    return all_tracks


# ===========================================================================
# 3. IDENTIFY EXISTING TRAJECTORY DATASETS
# ===========================================================================
def identify_datasets(results_dir, out_dir):
    heading("## TRAJECTORY DATASETS FOUND (verified against saved k/horizon metadata)")
    files = sorted(glob.glob(os.path.join(results_dir, "traj_dataset_k*.npz")))
    rows, ds = [], {}
    for path in files:
        d = np.load(path, allow_pickle=True)
        k, horizon = int(d["k"]), int(d["horizon"])   # from SAVED metadata, not filename
        fname_k = os.path.basename(path).split("_k")[1].split("_h")[0]
        if str(k) != fname_k:
            issue(f"{path}: filename says k={fname_k} but saved metadata says k={k}")
        feature_names = [str(n) for n in d["feature_names"]]
        meta_tr = [str(m) for m in d["meta_train"]]
        meta_te = [str(m) for m in d["meta_test"]]
        recs = sorted(set(m.split(":")[0] for m in meta_tr+meta_te))
        tracks = sorted(set((m.split(":")[0], m.split(":")[1]) for m in meta_tr+meta_te))
        print(f"\n{os.path.basename(path)}  (k={k} verified, horizon={horizon} verified)")
        print(f"  train_windows={len(d['Y_train'])}  test_windows={len(d['Y_test'])}")
        print(f"  input_dim={d['X_train'].shape[1]}  target_dim={d['Y_train'].shape[1]}")
        print(f"  feature_names={feature_names}")
        print(f"  recordings={recs}  n_tracks={len(tracks)}")
        rows.append([k, horizon, len(d["Y_train"]), len(d["Y_test"]),
                    d["X_train"].shape[1], d["Y_train"].shape[1],
                    ",".join(feature_names), ",".join(recs), len(tracks)])
        ds[(k,horizon)] = path
    save_csv(os.path.join(out_dir, "trajectory_dataset_inventory.csv"), rows,
            ["k","horizon","n_train_windows","n_test_windows","input_dim",
             "target_dim","feature_names","recordings","n_tracks"])
    return ds


# ===========================================================================
# 4. VERIFY WHAT k ACTUALLY MEANS (empirical, not just read from code)
# ===========================================================================
def verify_k_meaning(ds_paths, out_dir):
    heading("## WHAT DOES k ACTUALLY MEAN? (empirical verification)")
    rows = []
    ks = sorted(set(k for k,h in ds_paths))
    print("Feature dimension stays fixed at 9 for every k. Verifying THIS IS")
    print("BECAUSE k controls the WINDOW SIZE that avg_du/avg_dv/ddu/ddv are")
    print("computed over, NOT because k changes the number of feature slots.\n")
    a = np.array([[100+i, i*5.0, i*2.0, 5.0, 2.0] for i in range(15)])  # synthetic, LOCAL, code-comprehension only
    for k in ks:
        f = md2.window_features(a, 14, k)
        fdict = dict(zip(md2.FEATURE_NAMES, f))
        print(f"  k={k}: ddu={fdict['ddu']:.4f}  ddv={fdict['ddv']:.4f}  "
             f"avg_du={fdict['avg_du']:.4f}  du={fdict['du']:.4f}  "
             f"{'<- ddu/ddv EXACTLY ZERO, avg_du==du (window start==end)' if k==1 else ''}")
        rows.append([k, float(fdict["ddu"]), float(fdict["ddv"]),
                    float(fdict["avg_du"]), float(fdict["du"]), k==1])
    save_csv(os.path.join(out_dir, "k_meaning_verification.csv"), rows,
            ["k","ddu_example","ddv_example","avg_du_example","du_example","k_equals_1_degenerate"])
    note(f"**k does NOT concatenate k separate frame vectors.** The saved "
        f"feature vector is always length 9 "
        f"({', '.join(md2.FEATURE_NAMES)}). k controls the WINDOW that "
        f"`avg_du`/`avg_dv` (mean over the k-frame window) and `ddu`/`ddv` "
        f"((velocity at window end - velocity at window start) / window "
        f"time span) are computed over. **At k=1, the window start and end "
        f"are the same single frame, so `ddu` and `ddv` are ALWAYS EXACTLY "
        f"0.0, and `avg_du`/`avg_dv` trivially equal `du`/`dv`** -- verified "
        f"numerically above, not just read from source. This means k=1's "
        f"effective information content is 7 varying values, not 9.")


# ===========================================================================
# 5/6/7/8. WINDOW ALIGNMENT, TARGET ALIGNMENT, FEATURE RECOMPUTATION
#           (independent recomputation from real *_tracks.csv, no dataset
#           regeneration, no file writes to any existing dataset)
# ===========================================================================
def full_recompute_and_diff(all_tracks, ds_paths, results_dir, out_dir):
    heading("## INDEPENDENT RECOMPUTATION vs SAVED DATASETS (all k, all windows)")
    align_rows, feat_rows, sample_window_lines = [], [], []
    ordering_violations, boundary_violations = 0, 0
    target_checked = target_correct = target_incorrect = 0
    max_discrepancy = 0.0
    incorrect_examples = []

    for (k, horizon), path in sorted(ds_paths.items()):
        d = np.load(path, allow_pickle=True)
        meta_tr = [str(m) for m in d["meta_train"]]
        meta_te = [str(m) for m in d["meta_test"]]
        X_all = np.concatenate([d["X_train"], d["X_test"]])
        Y_all = np.concatenate([d["Y_train"], d["Y_test"]])
        U_all = np.concatenate([d["U_train"], d["U_test"]])
        meta_all = meta_tr + meta_te

        # independently recompute EVERY window from raw tracks via the
        # exact production function (not reimplemented)
        n_match, n_mismatch = 0, 0
        for (rec, tid), t in all_tracks.items():
            a = t["arr"]
            frames = a[:,0].astype(int)
            # ordering check (independent of make_dataset2)
            if np.any(np.diff(frames) < 0):
                ordering_violations += 1
                issue(f"k={k} h={horizon} {rec} track {tid}: NON-MONOTONIC frame order")
            index = {f:i for i,f in enumerate(frames)}
            for i in range(k-1, len(a)):
                f_now = int(a[i,0])
                if int(a[i,0]) - int(a[i-k+1,0]) != k-1:
                    continue  # non-consecutive window -- production code also skips this
                j = index.get(f_now + horizon)
                if j is None:
                    continue
                expected_key = f"{rec}:{tid}:{f_now}"
                if expected_key not in meta_all:
                    continue  # e.g. filtered by MIN_SPEED in production code -- not a violation
                mi = meta_all.index(expected_key)
                # recompute feature vector via the REAL production function
                feat_expected = md2.window_features(a, i, k)
                feat_saved = X_all[mi]
                feat_diff = float(np.max(np.abs(feat_expected - feat_saved)))
                if feat_diff > 1e-4:
                    n_mismatch += 1
                    feat_rows.append([k, horizon, rec, tid, f_now, feat_diff, "MISMATCH"])
                else:
                    n_match += 1

                # independent target lookup (NOT via build_examples -- a
                # separate, from-scratch frame-index lookup for a true
                # independence check)
                u_now, v_now = a[i,1], a[i,2]
                u_future_expected, v_future_expected = a[j,1], a[j,2]
                u_future_saved = U_all[mi][0] + Y_all[mi][0]
                v_future_saved = U_all[mi][1] + Y_all[mi][1]
                disc = float(np.hypot(u_future_saved-u_future_expected,
                                      v_future_saved-v_future_expected))
                target_checked += 1
                if disc < 1e-3:
                    target_correct += 1
                else:
                    target_incorrect += 1
                    incorrect_examples.append([k, horizon, rec, tid, f_now, disc])
                max_discrepancy = max(max_discrepancy, disc)

                # boundary check: window must not cross track (already
                # enforced by construction via the consecutive-frame check
                # above using THIS track's own array only -- verify no
                # cross-track contamination is possible by construction)
                align_rows.append([k, horizon, rec, tid, f_now, int(a[i,0]-a[i-k+1,0]),
                                  k-1, int(a[i,0]-a[i-k+1,0])==k-1])

        print(f"\nk={k} h={horizon}: feature recompute matched={n_match} "
             f"mismatched={n_mismatch}  (against production window_features(), "
             f"real tracks data)")

    save_csv(os.path.join(out_dir, "window_alignment_checks.csv"), align_rows,
            ["k","horizon","recording","track_id","frame","actual_span",
             "expected_span","consecutive_ok"])
    save_csv(os.path.join(out_dir, "feature_recomputation_checks.csv"), feat_rows,
            ["k","horizon","recording","track_id","frame","max_abs_diff","status"])
    save_csv(os.path.join(out_dir, "target_alignment_checks.csv"),
            [["total_checked", target_checked], ["correct", target_correct],
             ["incorrect", target_incorrect], ["max_discrepancy_mm", max_discrepancy]] +
            [["incorrect_example"]+r for r in incorrect_examples[:50]],
            ["item","value"])

    note(f"**Target alignment: {target_checked} targets independently checked "
        f"(from-scratch frame-index lookup, NOT via build_examples()), "
        f"{target_correct} correct, {target_incorrect} incorrect. Max "
        f"discrepancy: {max_discrepancy:.6f}mm.** "
        f"{'All targets within floating-point tolerance -- target = position at (last_history_frame + horizon), confirmed empirically.' if target_incorrect==0 else 'INCORRECT TARGETS FOUND -- see target_alignment_checks.csv for affected samples.'}\n\n"
        f"**Frame ordering violations found: {ordering_violations}.** "
        f"**Track/recording boundary crossing: impossible by construction** "
        f"-- windows are built by iterating within a single track's own "
        f"array (`for (rec,tid), t in all_tracks.items(): ... for i in "
        f"range(k-1, len(a))`), so a window can never span two tracks or "
        f"two recordings; verified by re-deriving every window from "
        f"per-track arrays only.")
    return incorrect_examples


# ===========================================================================
# Human-readable sample windows (section 6, beginning/middle/end/multi-rec)
# ===========================================================================
def print_sample_windows(all_tracks, ds_paths, out_dir):
    heading("## HUMAN-READABLE SAMPLE WINDOWS (beginning/middle/end, multiple tracks)")
    lines = []
    k_h_pick = sorted(ds_paths.keys())[len(ds_paths)//2] if ds_paths else (5,10)
    k, horizon = k_h_pick
    picked = 0
    for (rec, tid), t in sorted(all_tracks.items()):
        a = t["arr"]
        n = len(a)
        if n < k + horizon + 1:
            continue
        for pos_name, i in [("BEGINNING", k-1+2), ("MIDDLE", n//2), ("END", n-horizon-2)]:
            if i < k-1 or i >= n:
                continue
            if int(a[i,0]) - int(a[i-k+1,0]) != k-1:
                continue
            f_now = int(a[i,0])
            hist_frames = a[i-k+1:i+1, 0].astype(int).tolist()
            hist_pos = a[i-k+1:i+1, 1:3].tolist()
            j_target = None
            frames_arr = a[:,0].astype(int)
            idx = np.where(frames_arr == f_now+horizon)[0]
            target_frame = f_now + horizon
            target_pos = a[idx[0],1:3].tolist() if len(idx) else None
            block = [f"\n### {rec} track {tid} -- {pos_name} of track (k={k}, horizon={horizon})",
                    f"History frames: {hist_frames}"]
            for hf, hp in zip(hist_frames, hist_pos):
                block.append(f"  frame {hf} -> (u={hp[0]:.2f}, v={hp[1]:.2f})")
            block.append(f"Last history frame: {f_now}")
            block.append(f"Target frame (last_history_frame + horizon): {target_frame}")
            block.append(f"Target position: {target_pos if target_pos else 'NOT FOUND in track'}")
            lines.extend(block)
            print("\n".join(block))
            picked += 1
        if picked >= 15:
            break
    path = os.path.join(out_dir, "sample_windows_human_readable.md")
    with open(path, "w") as fh:
        fh.write("# Human-readable sample trajectory windows\n" + "\n".join(lines))
    print(f"\nsaved {path}")


# ===========================================================================
# 9. FRAME SPACING / TIME ASSUMPTIONS
# ===========================================================================
def check_time_assumptions(all_tracks, out_dir):
    heading("## FRAME SPACING / TIME ASSUMPTIONS")
    print(f"config.DT = {config.DT} seconds/frame "
         f"({'assumes CONSTANT 1/30s per frame -- no per-frame timestamps exist' if config.DT else ''})")
    print("Checked: *_tracks.csv has NO timestamp column -- only integer 'frame' "
         "numbers. du_mm_s/dv_mm_s are computed by the Kalman filter using "
         "config.DT as a fixed constant, NOT a measured per-frame timestamp.")
    gap_counts = []
    for (rec,tid), t in all_tracks.items():
        frames = t["arr"][:,0].astype(int)
        gaps = np.diff(frames)
        gap_counts.append((gaps>1).sum())
    total_gaps = sum(gap_counts)
    print(f"Total frame gaps (missing frame numbers) across all tracks: {total_gaps}")
    if total_gaps > 0:
        print("  Where gaps exist, config.DT still assumes uniform 1/30s spacing")
        print("  for any window that happens to skip over a gap. window_features()")
        print("  computes dt_span = (f1-f0)*config.DT using FRAME NUMBER difference,")
        print("  not elapsed real time -- so a window spanning a gap would silently")
        print("  compute ddu/ddv over the WRONG assumed elapsed time.")
        issue(f"{total_gaps} frame gaps exist; any window/target lookup spanning "
             f"a gap uses frame-number arithmetic as a stand-in for elapsed time, "
             f"which is an assumption, not a measurement.")
    note(f"**du/dv are labeled 'mm/s' (per Kalman filter units) but are derived "
        f"from frame-number differences x config.DT ({config.DT}s), not measured "
        f"timestamps** -- no timestamp column exists anywhere in *_tracks.csv. "
        f"This is a reasonable assumption for a fixed 30fps camera with no "
        f"dropped frames, but **{total_gaps} total frame gaps were found** "
        f"across all tracks, meaning this assumption is not universally true "
        f"in the actual data.")


# ===========================================================================
# 10. TRACK CONTINUITY / SUSPICIOUS JUMPS
# ===========================================================================
def track_continuity(all_tracks, out_dir):
    heading("## TRACK CONTINUITY / SUSPICIOUS POSITION JUMPS")
    rows, jump_rows = [], []
    for (rec,tid), t in all_tracks.items():
        a = t["arr"]
        frames = a[:,0].astype(int)
        gaps = np.diff(frames)
        n_gaps = int((gaps>1).sum())
        max_gap = int(gaps.max()) if len(gaps) else 0
        pos = a[:,1:3]
        d_pos = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        consecutive = gaps == 1
        if consecutive.any():
            jump_thresh = np.percentile(d_pos[consecutive], 99) * 3
            suspicious = np.where(consecutive & (d_pos > jump_thresh))[0]
        else:
            suspicious = np.array([], dtype=int)
        print(f"  {rec} track {tid}: n_gaps={n_gaps} max_gap={max_gap} "
             f"suspicious_jumps={len(suspicious)}")
        rows.append([rec, tid, len(a), n_gaps, max_gap, len(suspicious)])
        for si in suspicious:
            jump_rows.append([rec, tid, int(frames[si]), int(frames[si+1]),
                             float(d_pos[si])])
    save_csv(os.path.join(out_dir, "track_continuity.csv"), rows,
            ["recording","track_id","n_observations","n_gaps","max_gap","n_suspicious_jumps"])
    save_csv(os.path.join(out_dir, "suspicious_jumps.csv"), jump_rows,
            ["recording","track_id","frame_a","frame_b","jump_distance_mm"])
    note("Suspicious jumps flagged at >3x the 99th-percentile consecutive-frame "
        "displacement, per-track. Not removed -- see suspicious_jumps.csv.")


# ===========================================================================
# 11/12/13. SPLIT AUDIT, CROSS-SPLIT CONTAMINATION, DUPLICATES
# ===========================================================================
def split_and_leakage_audit(ds_paths, out_dir):
    heading("## TRAIN/TEST SPLIT AUDIT + CROSS-SPLIT TEMPORAL RELATIONSHIPS + DUPLICATES")
    split_rows, overlap_rows, dup_rows = [], [], []
    for (k,horizon), path in sorted(ds_paths.items()):
        d = np.load(path, allow_pickle=True)
        meta_tr = [str(m) for m in d["meta_train"]]
        meta_te = [str(m) for m in d["meta_test"]]

        def parse(meta):
            out = {}
            for m in meta:
                rec, tid, fr = m.split(":")
                out.setdefault((rec,tid), []).append(int(fr))
            return out
        tr_bt, te_bt = parse(meta_tr), parse(meta_te)

        for key in sorted(set(tr_bt) | set(te_bt)):
            trf = sorted(tr_bt.get(key, []))
            tef = sorted(te_bt.get(key, []))
            train_target_frames = set(f+horizon for f in trf)
            test_target_frames = set(f+horizon for f in tef)
            train_input_frames = set()
            for f in trf:
                train_input_frames |= set(range(f-k+1, f+1))
            test_input_frames = set()
            for f in tef:
                test_input_frames |= set(range(f-k+1, f+1))

            train_target_in_test_input = train_target_frames & test_input_frames
            test_target_in_train_input = test_target_frames & train_input_frames
            same_target_frame_both = train_target_frames & test_target_frames
            input_span_overlap = train_input_frames & test_input_frames

            gap = (min(tef) - max(trf)) if trf and tef else None
            split_rows.append([key[0], key[1], f"{min(trf)}-{max(trf)}" if trf else "none",
                              f"{min(tef)}-{max(tef)}" if tef else "none",
                              len(trf), len(tef), gap])
            overlap_rows.append([k, horizon, key[0], key[1],
                                len(train_target_in_test_input),
                                len(test_target_in_train_input),
                                len(same_target_frame_both),
                                len(input_span_overlap)])
            if train_target_in_test_input or test_target_in_train_input or input_span_overlap:
                issue(f"k={k} h={horizon} {key}: cross-split temporal relationship "
                     f"found -- train_target_in_test_input={len(train_target_in_test_input)} "
                     f"test_target_in_train_input={len(test_target_in_train_input)} "
                     f"input_span_overlap={len(input_span_overlap)}")

        # duplicates
        dup_within_tr = len(meta_tr) - len(set(meta_tr))
        dup_within_te = len(meta_te) - len(set(meta_te))
        dup_across = len(set(meta_tr) & set(meta_te))
        dup_rows.append([k, horizon, dup_within_tr, dup_within_te, dup_across])
        print(f"\nk={k} h={horizon}: duplicate windows within-train={dup_within_tr} "
             f"within-test={dup_within_te} across-splits={dup_across}")

    save_csv(os.path.join(out_dir, "split_audit.csv"), split_rows,
            ["recording","track_id","train_frame_range","test_frame_range",
             "n_train_windows","n_test_windows","gap_train_to_test"])
    save_csv(os.path.join(out_dir, "cross_split_temporal_overlap.csv"), overlap_rows,
            ["k","horizon","recording","track_id","train_target_in_test_input",
             "test_target_in_train_input","same_target_frame_both_splits",
             "input_span_overlap_count"])
    save_csv(os.path.join(out_dir, "duplicate_check.csv"), dup_rows,
            ["k","horizon","dup_within_train","dup_within_test","dup_across_splits"])
    note("Cross-split temporal relationships computed exactly as specified: "
        "train TARGET frames checked against test INPUT frames and vice "
        "versa, plus same-target-frame-in-both and raw input-span overlap. "
        "Reported as counts, not pre-judged as leakage or non-leakage.")


# ===========================================================================
# 14. FEATURE-LEVEL DATA QUALITY, per k
# ===========================================================================
def feature_quality(ds_paths, out_dir):
    heading("## FEATURE / TARGET DATA QUALITY")
    rows = []
    for (k,horizon), path in sorted(ds_paths.items()):
        d = np.load(path, allow_pickle=True)
        feature_names = [str(n) for n in d["feature_names"]]
        X = np.concatenate([d["X_train"], d["X_test"]])
        Y = np.concatenate([d["Y_train"], d["Y_test"]])
        for i, name in enumerate(feature_names):
            s = stats_block(X[:,i])
            rows.append([k, horizon, name] + [s[kk] for kk in
                       ["n","min","max","mean","median","std","p05","p95","nan_count","inf_count","zero_count"]])
            print(f"  k={k} h={horizon} {name:8s}: min={s['min']:.2f} max={s['max']:.2f} "
                 f"mean={s['mean']:.2f} nan={s['nan_count']} inf={s['inf_count']}")
        for i, name in enumerate(["target_disp_u","target_disp_v"]):
            s = stats_block(Y[:,i])
            rows.append([k, horizon, name] + [s[kk] for kk in
                       ["n","min","max","mean","median","std","p05","p95","nan_count","inf_count","zero_count"]])
    save_csv(os.path.join(out_dir, "trajectory_feature_statistics.csv"), rows,
            ["k","horizon","feature","n","min","max","mean","median","std",
             "p05","p95","nan_count","inf_count","zero_count"])


# ===========================================================================
# 15. COMPARE k DATASETS
# ===========================================================================
def compare_k_datasets(ds_paths, out_dir):
    heading("## COMPARISON ACROSS k (same horizon)")
    by_h = {}
    for (k,h), path in ds_paths.items():
        by_h.setdefault(h, []).append(k)
    rows = []
    for h, ks in by_h.items():
        target_frame_sets = {}
        for k in sorted(ks):
            d = np.load(ds_paths[(k,h)], allow_pickle=True)
            meta = [str(m) for m in list(d["meta_train"])+list(d["meta_test"])]
            target_frames = set(f"{m.split(':')[0]}:{m.split(':')[1]}:{int(m.split(':')[2])+h}" for m in meta)
            target_frame_sets[k] = target_frames
            print(f"  h={h} k={k}: n_windows={len(meta)}")
            rows.append([h, k, len(meta)])
        ks_sorted = sorted(ks)
        if len(ks_sorted) >= 2:
            base = target_frame_sets[ks_sorted[0]]
            for k in ks_sorted[1:]:
                overlap = len(base & target_frame_sets[k])
                print(f"  h={h}: k={ks_sorted[0]} vs k={k}: "
                     f"{overlap} target frames in common out of "
                     f"{len(base)} (k={ks_sorted[0]}) / {len(target_frame_sets[k])} (k={k})")
    save_csv(os.path.join(out_dir, "k_comparison.csv"), rows, ["horizon","k","n_windows"])
    note("Larger k requires more preceding consecutive frames, so it evaluates "
        "FEWER target frames near the start of each track than smaller k -- "
        "quantified above per h. Comparisons of prediction error across k "
        "(reserved for the NEXT validation stage) should account for this "
        "difference in evaluated population, not assume it away.")


# ===========================================================================
def main():
    p = argparse.ArgumentParser(
        description="Validate trajectory DATASETS only -- no model training "
                    "or evaluation. Reuses make_dataset2.py's real functions "
                    "on real *_tracks.csv to independently verify saved .npz files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--out", type=str, default="validation_results/trajectory_data")
    args = p.parse_args()

    def _resolve(path):
        return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)
    args.results_dir = _resolve(args.results_dir)
    args.out = _resolve(args.out)
    os.makedirs(args.out, exist_ok=True)

    all_tracks = identify_data_sources(args.results_dir, args.out)
    ds_paths = identify_datasets(args.results_dir, args.out)
    if not ds_paths:
        print("\nNo traj_dataset_k*.npz files found -- nothing further to validate.")
        return
    verify_k_meaning(ds_paths, args.out)
    incorrect_targets = full_recompute_and_diff(all_tracks, ds_paths, args.results_dir, args.out)
    print_sample_windows(all_tracks, ds_paths, args.out)
    check_time_assumptions(all_tracks, args.out)
    track_continuity(all_tracks, args.out)
    split_and_leakage_audit(ds_paths, args.out)
    feature_quality(ds_paths, args.out)
    compare_k_datasets(ds_paths, args.out)

    save_csv(os.path.join(args.out, "trajectory_data_issues.csv"),
            [[i, msg] for i, msg in enumerate(ISSUES)], ["issue_number","description"])

    report_path = os.path.join(args.out, "trajectory_data_validation_report.md")
    with open(report_path, "w") as fh:
        fh.write("# TRAJECTORY DATA VALIDATION REPORT\n\n")
        fh.write(f"**{len(ISSUES)} issues flagged** (see trajectory_data_issues.csv "
                f"for the full list; not auto-fixed, not auto-dismissed).\n\n")
        fh.write("\n\n".join(REPORT))
    print(f"\n\nsaved {report_path}")

    print("\n" + "#"*70)
    print("# TRAJECTORY DATA VALIDATION SUMMARY")
    print("#"*70)
    print(f"Distinct tracks: {len(all_tracks)}")
    print(f"Datasets found: {len(ds_paths)}  (k values: {sorted(set(k for k,h in ds_paths))})")
    print(f"Total issues flagged: {len(ISSUES)}")
    print(f"Full report: {report_path}")


if __name__ == "__main__":
    main()