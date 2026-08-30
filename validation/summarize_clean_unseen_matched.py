"""
summarize_clean_unseen_matched.py — reads ONLY
validation_results/geometry_vs_learned_matched/clean_unseen_subset.csv.
Does not read raw camera data, models, datasets, or training files. Does
not retrain, rerun geometry, rerun the learned classifier, or modify
clean_unseen_subset.csv. Pure read-and-aggregate, with a hard integrity
check before computing anything.

Usage:
  python summarize_clean_unseen_matched.py
  python summarize_clean_unseen_matched.py --csv validation_results/geometry_vs_learned_matched/clean_unseen_subset.csv
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

DEFAULT_CSV = "validation_results/geometry_vs_learned_matched/clean_unseen_subset.csv"
OUT_CSV = "validation_results/geometry_vs_learned_matched/clean_unseen_summary.csv"


def is_true(s):
    return str(s).strip().lower() in ("true", "1")


def main():
    p = argparse.ArgumentParser(description="Read-only summary of the already-generated clean unseen subset.")
    p.add_argument("--csv", type=str, default=DEFAULT_CSV)
    args = p.parse_args()
    in_path = args.csv if os.path.isabs(args.csv) else os.path.join(_PROJECT_ROOT, args.csv)
    out_path = os.path.join(_PROJECT_ROOT, OUT_CSV)

    if not os.path.exists(in_path):
        print(f"ERROR: {in_path} not found. This script reads ONLY that file "
             f"and does nothing else -- run compare_geometry_learned_matched.py "
             f"first if you haven't.")
        sys.exit(1)

    rows = list(csv.DictReader(open(in_path)))
    if not rows:
        print("The clean unseen subset is empty -- nothing to summarize.")
        return

    # ---- INTEGRITY CHECK, hard stop, before any calculation ----
    if "seen_in_learned_train" in rows[0]:
        violations = [r for r in rows if is_true(r["seen_in_learned_train"])]
        if violations:
            print(f"STOP: {len(violations)} row(s) in {in_path} are marked "
                 f"seen_in_learned_train=True. This file is supposed to be "
                 f"restricted to unseen trials only. NOT computing any "
                 f"summary. Affected rows (first 10 shown):")
            for r in violations[:10]:
                print(f"  {r.get('recording')} sample_index={r.get('sample_index')} "
                     f"h={r.get('h_mm')}")
            sys.exit(1)
        print(f"Integrity check passed: all {len(rows)} rows confirmed "
             f"seen_in_learned_train=False.")
    else:
        print("NOTE: 'seen_in_learned_train' column not found in this CSV -- "
             "cannot verify the training-overlap guarantee from this file "
             "alone. Proceeding, since the file's own name/origin already "
             "implies this restriction, but flagging that this script could "
             "not independently re-verify it here.")

    print("="*78)
    print("CLEAN UNSEEN SUBSET SUMMARY (read-only, nothing rerun)")
    print("="*78)

    hs = sorted(set(float(r["h_mm"]) for r in rows))
    final_table_rows = []
    summary_out_rows = []

    for h in hs:
        h_rows = [r for r in rows if float(r["h_mm"]) == h]
        n = len(h_rows)
        geo_ok = [is_true(r["geometry_correct"]) for r in h_rows]
        learn_ok = [is_true(r["learned_correct"]) for r in h_rows]
        geo_correct_n, geo_incorrect_n = sum(geo_ok), n - sum(geo_ok)
        learn_correct_n, learn_incorrect_n = sum(learn_ok), n - sum(learn_ok)
        both = sum(1 for g, l in zip(geo_ok, learn_ok) if g and l)
        geo_only = sum(1 for g, l in zip(geo_ok, learn_ok) if g and not l)
        learn_only = sum(1 for g, l in zip(geo_ok, learn_ok) if not g and l)
        both_wrong = sum(1 for g, l in zip(geo_ok, learn_ok) if not g and not l)
        geo_acc = geo_correct_n / n
        learn_acc = learn_correct_n / n

        geo_fail_idx = [i for i, g in enumerate(geo_ok) if not g]
        n_geo_fail = len(geo_fail_idx)
        learn_correct_on_fail = sum(1 for i in geo_fail_idx if learn_ok[i])
        recovery = learn_correct_on_fail / n_geo_fail if n_geo_fail else float("nan")

        print(f"\nh={h:.0f} mm")
        print(f"  N clean unseen trials: {n}")
        print(f"  geometry correct/incorrect: {geo_correct_n}/{geo_incorrect_n}  "
             f"(accuracy={geo_acc:.4f})")
        print(f"  learned correct/incorrect:  {learn_correct_n}/{learn_incorrect_n}  "
             f"(accuracy={learn_acc:.4f})")
        print(f"  both correct: {both}   geometry-only correct: {geo_only}   "
             f"learned-only correct: {learn_only}   both wrong: {both_wrong}")
        print(f"  learned recovery rate among geometry failures: "
             f"{recovery:.4f}  (n_geometry_failures={n_geo_fail}, "
             f"n_learned_correct_on_those={learn_correct_on_fail})")

        obj_summaries = {}
        for obj in ["ball", "box"]:
            o_rows = [r for r in h_rows if r["truth"] == obj]
            if not o_rows:
                print(f"  {obj.upper()}: no clean unseen samples at this h")
                obj_summaries[obj] = None
                continue
            og = [is_true(r["geometry_correct"]) for r in o_rows]
            ol = [is_true(r["learned_correct"]) for r in o_rows]
            ob = sum(1 for g, l in zip(og, ol) if g and l)
            og_only = sum(1 for g, l in zip(og, ol) if g and not l)
            ol_only = sum(1 for g, l in zip(og, ol) if not g and l)
            obw = sum(1 for g, l in zip(og, ol) if not g and not l)
            o_geo_acc = sum(og) / len(o_rows)
            o_learn_acc = sum(ol) / len(o_rows)
            o_geo_fail_idx = [i for i, g in enumerate(og) if not g]
            o_recovery = (sum(1 for i in o_geo_fail_idx if ol[i]) / len(o_geo_fail_idx)
                         if o_geo_fail_idx else float("nan"))
            print(f"  {obj.upper()}: N={len(o_rows)} geometry_acc={o_geo_acc:.4f} "
                 f"learned_acc={o_learn_acc:.4f} both={ob} geo_only={og_only} "
                 f"learned_only={ol_only} both_wrong={obw} "
                 f"recovery_rate={o_recovery:.4f}")
            obj_summaries[obj] = dict(n=len(o_rows), geo_acc=o_geo_acc, learn_acc=o_learn_acc,
                                     both=ob, geo_only=og_only, learn_only=ol_only,
                                     both_wrong=obw, recovery=o_recovery)

        final_table_rows.append([h, n, geo_acc, learn_acc, n_geo_fail, learn_correct_on_fail, recovery])
        summary_out_rows.append(["overall", h, n, geo_correct_n, geo_incorrect_n, geo_acc,
                                learn_correct_n, learn_incorrect_n, learn_acc,
                                both, geo_only, learn_only, both_wrong,
                                n_geo_fail, learn_correct_on_fail, recovery])
        for obj, s in obj_summaries.items():
            if s is None:
                summary_out_rows.append([obj, h, 0, "", "", "", "", "", "", "", "", "", "", "", "", ""])
            else:
                summary_out_rows.append([obj, h, s["n"], "", "", s["geo_acc"], "", "", s["learn_acc"],
                                        s["both"], s["geo_only"], s["learn_only"], s["both_wrong"],
                                        "", "", s["recovery"]])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group","h_mm","n","geometry_correct","geometry_incorrect","geometry_accuracy",
                   "learned_correct","learned_incorrect","learned_accuracy",
                   "both_correct","geometry_only_correct","learned_only_correct","both_wrong",
                   "n_geometry_failures","n_learned_correct_on_failures","learned_recovery_rate"])
        w.writerows(summary_out_rows)
    print(f"\nsaved {out_path}")

    print("\n" + "="*78)
    print("FINAL TABLE")
    print("="*78)
    print(f"{'h':>4} | {'N':>5} | {'Geom acc':>9} | {'Learn acc':>9} | "
         f"{'Geom fail':>9} | {'Learn correct on fail':>21} | {'Recovery':>9}")
    for h, n, ga, la, nf, lc, rec in final_table_rows:
        print(f"{h:>4.0f} | {n:>5d} | {ga:>9.4f} | {la:>9.4f} | {nf:>9d} | "
             f"{lc:>21d} | {rec:>9.4f}")


if __name__ == "__main__":
    main()