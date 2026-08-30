"""
visualize_tracks.py — save annotated camera-view images for specific frames
of ball_and_box_moving, labeling every tracked position with its plain
numerical track ID only. Does NOT classify anything (no ball/box, no
color-coding by class) -- purely for YOU to visually identify which track
ID corresponds to which physical object. Modifies no existing file.

Reused, unmodified: geometry.load_masked, fit_static.load_table,
config.FX/FY/CX/CY. Pixel-projection formula is the SAME one
track_moving.py's own GIF-overlay already uses (uv_to_pixel), so marker
positions land exactly where that existing, trusted code would draw them.

Usage:
  python visualize_tracks.py --frames 100,1060,1500
  python visualize_tracks.py --frames 100,1060,1500 \\
      --recording "Data/extracted/ball_and_box_moving" \\
      --tracks-csv "results/ball_and_box_moving_tracks.csv"
  python visualize_tracks.py --help
"""
import argparse
import csv
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
from geometry import load_masked
from fit_static import load_table


def uv_to_pixel(tf, uv, assumed_height_mm=30.0):
    """Same formula track_moving.py's make_gif() already uses to place
    markers on the camera view -- reused verbatim, not reinvented.
    assumed_height_mm is a fixed approximation for visualization only
    (matches the original code's own convention)."""
    cam = tf.to_camera(np.array([[uv[0], uv[1], assumed_height_mm]]))[0]
    u_px = cam[0] * config.FX / cam[2] + config.CX
    v_px = cam[1] * config.FY / cam[2] + config.CY
    return u_px, v_px


def main():
    p = argparse.ArgumentParser(
        description="Save annotated camera-view images with PLAIN NUMERIC "
                    "track-ID labels only -- no classification, no ball/box "
                    "color-coding. For visual ground-truth identification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--frames", type=str, default="100,1060,1500",
                   help="Comma-separated frame numbers to render.")
    p.add_argument("--recording", type=str, default="Data/extracted/ball_and_box_moving",
                   help="Path to the recording folder (containing depth/*.npy).")
    p.add_argument("--tracks-csv", type=str, default="results/ball_and_box_moving_tracks.csv")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--out", type=str, default="results/track_id_check",
                   help="Where to save the annotated images.")
    args = p.parse_args()

    def _resolve(path):
        return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)
    args.recording = _resolve(args.recording)
    args.tracks_csv = _resolve(args.tracks_csv)
    args.results_dir = _resolve(args.results_dir)
    args.out = _resolve(args.out)

    if not os.path.exists(args.tracks_csv):
        print(f"ERROR: {args.tracks_csv} not found.")
        sys.exit(1)
    depth_dir = os.path.join(args.recording, "depth")
    if not os.path.isdir(depth_dir):
        print(f"ERROR: {depth_dir} not found.")
        sys.exit(1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tf, z = load_table()
    rows = list(csv.DictReader(open(args.tracks_csv)))
    by_frame = {}
    for r in rows:
        by_frame.setdefault(int(r["frame"]), []).append(r)

    frames_wanted = [int(x) for x in args.frames.split(",")]
    os.makedirs(args.out, exist_ok=True)

    for frame_idx in frames_wanted:
        depth_path = os.path.join(depth_dir, f"depth_{frame_idx:06d}.npy")
        if not os.path.exists(depth_path):
            print(f"frame {frame_idx}: {depth_path} not found -- skipping")
            continue
        tracks_here = by_frame.get(frame_idx)
        if not tracks_here:
            print(f"frame {frame_idx}: no confirmed tracks in "
                 f"{args.tracks_csv} at this frame -- image will be saved "
                 f"with no labels (object may not have been confirmed yet, "
                 f"or was lost this frame)")
            tracks_here = []

        depth = load_masked(depth_path)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(np.where(depth > 0, depth, np.nan), cmap="turbo")
        ax.set_title(f"ball_and_box_moving -- frame {frame_idx}\n"
                    f"(plain track IDs only -- NOT classified)")
        ax.set_xlabel("pixel u")
        ax.set_ylabel("pixel v")

        for r in tracks_here:
            tid = int(r["track_id"])
            uv = (float(r["u_mm"]), float(r["v_mm"]))
            px, py = uv_to_pixel(tf, uv)
            ax.scatter([px], [py], s=140, facecolors="none",
                      edgecolors="white", linewidths=2.5, marker="o")
            ax.annotate(f"Track {tid}", (px, py), xytext=(10, 10),
                       textcoords="offset points", color="white",
                       fontsize=13, fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6))

        out_path = os.path.join(args.out, f"frame_{frame_idx:06d}_tracks.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        track_ids_here = [int(r["track_id"]) for r in tracks_here]
        print(f"frame {frame_idx}: saved {out_path}  (tracks present: {track_ids_here})")

    print(f"\nAll images saved under {args.out}/")
    print("Open each image and note, for yourself, which track ID number sits")
    print("on the round object and which sits on the boxy object. This script")
    print("does not make that determination for you.")


if __name__ == "__main__":
    main()