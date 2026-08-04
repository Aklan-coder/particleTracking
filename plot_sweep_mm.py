"""
plot_sweep_mm.py — Six-panel resolution-sweep figure in mm units,
matching the layout of the earlier pixel-based pipeline's figure so the
two implementations can be compared side by side.

Panels (rows = BALL, BOX; columns = footprint, identification, detection):
  1. median cells over object vs cell size h        (static vs moving)
  2. identification rate vs h  (competing-hypotheses classifier,
     cell-level data only)                          (static vs moving)
  3. detection rate vs h                            (static vs moving)

Style matches the pixel figure: static = blue solid circles,
moving = orange dashed squares.

Usage:
  python3 sweep.py ball_static box_static ball_moving box_moving --every 20
  python3 plot_sweep_mm.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


def main():
    path = os.path.join(config.RESULTS_DIR, "sweep_results.csv")
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in ("h", "det_rate", "id_rate", "unk_rate",
                  "wrong_rate", "cells_across"):
            r[k] = float(r[k])

    def series(rec):
        rs = sorted((r for r in rows if r["recording"] == rec),
                    key=lambda r: r["h"])
        return rs

    STYLES = {"static": dict(color="tab:blue", marker="o", ls="-"),
              "moving": dict(color="tab:orange", marker="s", ls="--")}

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for row, obj in enumerate(["ball", "box"]):
        variants = {"static": series(f"{obj}_static"),
                    "moving": series(f"{obj}_moving")}
        panels = [
            ("cells_across", f"{obj.upper()} — cells over object vs h",
             "median cells over object"),
            ("id_rate", f"{obj.upper()} — identification vs h",
             "identification rate"),
            ("det_rate", f"{obj.upper()} — detection vs h",
             "detection rate"),
        ]
        for col, (key, title, ylabel) in enumerate(panels):
            ax = axes[row][col]
            for name, rs in variants.items():
                if not rs:
                    continue
                ax.plot([r["h"] for r in rs], [r[key] for r in rs],
                        label=name, **STYLES[name])
            if key == "cells_across":
                ax.axhline(8, color="red", ls=":", lw=1,
                           label="MIN 8 cells")
            else:
                ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("cell size h (mm)")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.suptitle("Resolution sweep (mm cells) — static (no prior) vs "
                 "moving — companion to the pixel-cell figure",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(config.RESULTS_DIR, "sweep_curves_mm_6panel.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()