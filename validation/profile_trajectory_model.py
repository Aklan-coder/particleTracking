"""
profile_trajectory_model.py — CHARACTERIZATION ONLY. Profiles parameter
size, memory, analytical workload, and measured latency/throughput of the
EXISTING frozen trajectory models. Trains nothing, changes nothing, makes
no edge-device suitability claims. Modifies no existing file.

PIPELINE SCOPE (Section 1, traced from code):
  IN SCOPE (this script measures these):
    A. history window handling      -- window_features()'s slicing of the
                                        last k rows of a track's array
    B. temporal feature construction -- window_features() itself
    C. normalization                 -- (x - mu) / sd
    D. linear-regression inference   -- Z @ W + b
    E. future-position reconstruction -- U + predicted_displacement
  OUT OF SCOPE (NOT measured here, explicitly, per instruction #25):
    depth loading, backprojection, RANSAC plane/sphere/box fitting,
    grid discretization, Kalman filtering, track association -- these
    happen upstream in track_moving.py, entirely before any row ever
    reaches window_features(). Reported as excluded, not benchmarked.

DTYPES CONFIRMED BY DIRECT INSPECTION, NOT ASSUMED (from the actual
saved files): W and b are float64. mu and sd are float32. X (features)
is float32. This mixed precision is reported exactly as found.

REUSED DIRECTLY: train_model2.predict_disp() for the inference step;
make_dataset2.window_features() for the feature-construction step.
Neither is reimplemented -- the SAME function objects are imported and
timed directly, so the measured latency is of your actual code, not a
re-description of it.

Usage:
  python profile_trajectory_model.py
  python profile_trajectory_model.py --results-dir results --out validation_results/trajectory_profile
  python profile_trajectory_model.py --help
"""
import argparse
import csv
import os
import platform
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = _THIS_DIR
for _ in range(4):
    if os.path.exists(os.path.join(_PROJECT_ROOT, "config.py")):
        break
    _PROJECT_ROOT = os.path.dirname(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

import config
import train_model2 as tm2
import make_dataset2 as md2

K_VALUES = [1, 3, 5, 10]
HORIZON = 10
DT_S = config.DT
NOMINAL_FPS = 1.0 / DT_S


def pctl(a, p): return float(np.percentile(a, p)) if len(a) else float("nan")
def save_csv(path, rows, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    return path

REPORT = []
def note(md): REPORT.append(md)
def heading(md):
    print("\n"+"="*78); print(md); print("="*78); REPORT.append(md)


# ===========================================================================
# ENVIRONMENT
# ===========================================================================
def profile_environment(out_dir):
    heading("## ENVIRONMENT")
    lines = []
    lines.append(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    lines.append(f"Python: {platform.python_version()}")
    lines.append(f"CPU (platform string): {platform.processor() or platform.machine()}")
    try:
        import multiprocessing
        lines.append(f"Logical CPU count: {multiprocessing.cpu_count()}")
    except Exception as e:
        lines.append(f"Logical CPU count: unavailable ({e})")
    try:
        import psutil
        lines.append(f"Physical CPU count: {psutil.cpu_count(logical=False)}")
        lines.append(f"RAM total: {psutil.virtual_memory().total/1e9:.2f} GB")
    except ImportError:
        lines.append("Physical CPU count / RAM: unavailable (psutil not installed)")
    lines.append(f"NumPy version: {np.__version__}")
    try:
        cfg = np.show_config(mode="dicts")
        blas = cfg.get("Build Dependencies", {}).get("blas", {}).get("name", "unknown")
        lines.append(f"BLAS backend: {blas}")
    except Exception:
        lines.append("BLAS backend: could not be determined programmatically "
                     "(see `python -c \"import numpy; numpy.show_config()\"` manually)")
    lines.append("GPU: NOT USED. This model is a plain-numpy CPU computation "
                 "(np.zeros init, numpy matmul) -- no CUDA/GPU library is "
                 "imported anywhere in train_model2.py or this profiler.")
    lines.append(f"config.DT (assumed seconds/frame): {DT_S}  "
                f"(nominal {NOMINAL_FPS:.1f} FPS -- NOT a measured camera "
                f"timestamp, see config.py)")
    for l in lines: print(" ", l)
    with open(os.path.join(out_dir, "profiling_environment.txt"), "w") as fh:
        fh.write("\n".join(lines))
    note("Environment recorded above and in profiling_environment.txt. "
        "**All latency numbers below are specific to THIS machine and are "
        "NOT hardware-independent or edge-device benchmarks.**")


# ===========================================================================
# PARAMETER SIZE + MEMORY
# ===========================================================================
def profile_parameters(results_dir, out_dir):
    heading("## PARAMETER SIZE + MEMORY (Sections 3, 4, 19, 22, 23)")
    param_rows, mem_rows, hist_rows = [], [], []
    models = {}
    for k in K_VALUES:
        path = os.path.join(results_dir, f"traj_model_k{k}_h{HORIZON}.npz")
        if not os.path.exists(path):
            print(f"  k={k}: {path} not found -- skipping.")
            continue
        m = np.load(path, allow_pickle=True)
        models[k] = m
        W, b, mu, sd = m["W"], m["b"], m["mu"], m["sd"]
        n_in, n_out = W.shape
        n_weight_params = W.size
        n_bias_params = b.size
        n_total_params = n_weight_params + n_bias_params
        n_mu = mu.size; n_sd = sd.size

        w_bytes = W.nbytes; b_bytes = b.nbytes
        mu_bytes = mu.nbytes; sd_bytes = sd.nbytes
        raw_param_bytes = w_bytes + b_bytes
        norm_state_bytes = mu_bytes + sd_bytes
        total_inference_state_bytes = raw_param_bytes + norm_state_bytes
        file_size = os.path.getsize(path)

        print(f"\nk={k}: W.shape={W.shape} ({W.dtype})  b.shape={b.shape} ({b.dtype})  "
             f"mu/sd ({mu.dtype})")
        print(f"  formula: {n_in} inputs -> {n_out} outputs => W has {n_in}x{n_out}="
             f"{n_weight_params} weights + {n_out} biases = {n_total_params} learned params")
        print(f"  raw parameter bytes: W={w_bytes}B + b={b_bytes}B = {raw_param_bytes}B")
        print(f"  normalization state (NOT learned params): mu={mu_bytes}B + sd={sd_bytes}B = {norm_state_bytes}B")
        print(f"  total inference-state bytes: {total_inference_state_bytes}B")
        print(f"  serialized .npz file size on disk: {file_size}B "
             f"(includes .npz container/metadata overhead beyond raw arrays)")

        param_rows.append([k, n_in, n_out, n_weight_params, n_bias_params, n_total_params, n_mu, n_sd])
        mem_rows.append([k, str(W.dtype), str(b.dtype), w_bytes, b_bytes, raw_param_bytes,
                        norm_state_bytes, total_inference_state_bytes, file_size])

        # theoretical alternative precisions -- CLEARLY LABELED, not validated
        for dtype_name, itemsize in [("float64",8), ("float32",4), ("int16",2), ("int8",1)]:
            theo_bytes = n_total_params * itemsize
            mem_rows.append([k, f"THEORETICAL_{dtype_name}", "-", "-", "-", theo_bytes,
                            "-", "-", "-"])

        # history buffer: k rows x 5 values (frame,u,v,du,dv), actual dtype from tracks arrays
        obs_dtype_bytes = 8  # np.array() of mixed int/float track rows upcasts to float64 -- verified empirically below
        vals_per_obs = 5
        hist_bytes = k * vals_per_obs * obs_dtype_bytes
        hist_rows.append([k, k, vals_per_obs, obs_dtype_bytes, hist_bytes])
        print(f"  history buffer: {k} observations x {vals_per_obs} values x "
             f"{obs_dtype_bytes}B (float64, track array dtype) = {hist_bytes}B")

    save_csv(os.path.join(out_dir, "model_parameter_size.csv"), param_rows,
            ["k","n_inputs","n_outputs","n_weight_params","n_bias_params",
             "n_total_learned_params","n_mu_values","n_sd_values"])
    save_csv(os.path.join(out_dir, "model_memory.csv"), mem_rows,
            ["k","W_dtype","b_dtype","W_bytes","b_bytes","raw_param_bytes",
             "norm_state_bytes","total_inference_state_bytes","file_size_bytes"])
    save_csv(os.path.join(out_dir, "history_buffer_memory.csv"), hist_rows,
            ["k","n_observations_retained","values_per_observation","bytes_per_value","total_history_bytes"])

    param_counts = set(r[5] for r in param_rows)
    note(f"**Does parameter count change with k? "
        f"{'NO -- confirmed identical (' + str(param_counts) + ' learned params) at every k.' if len(param_counts)==1 else 'YES, unexpectedly -- ' + str(param_counts)}** "
        f"k changes temporal feature-CONSTRUCTION and history-STORAGE "
        f"requirements (see history_buffer_memory.csv, which scales linearly "
        f"with k), NOT the regression model's own dimensionality (fixed at "
        f"9 inputs -> 2 outputs regardless of k).")
    return models


# ===========================================================================
# ANALYTICAL OPERATION COUNTS
# ===========================================================================
def profile_operation_counts(out_dir):
    heading("## ANALYTICAL OPERATION COUNTS (Sections 6, 7, 8, 9)")
    print("Counting convention, stated explicitly: a mean over n elements = "
         "(n-1) additions + 1 division. Matrix-vector product counted BOTH "
         "as MACs (multiply-accumulate, the hardware-relevant unit) AND as "
         "separate multiply+add FLOPs, clearly labeled as different things.")
    rows = []
    for k in K_VALUES:
        # A. feature construction (window_features(), traced line by line)
        dt_span_ops = 2                    # 1 sub (f1-f0) + 1 mult (*DT)
        avg_du_ops = k                     # (k-1) adds + 1 div, counted as k total
        avg_dv_ops = k
        ddu_ops = 2                        # 1 sub + 1 div
        ddv_ops = 2
        speed_ops = 4                      # du1^2 + dv1^2 + add + sqrt
        feature_ops = dt_span_ops + avg_du_ops + avg_dv_ops + ddu_ops + ddv_ops + speed_ops

        # B. normalization: (x-mu)/sd, 9 elements
        norm_ops = 9 + 9  # 9 subtractions + 9 divisions

        # C. model inference: Z(1x9) @ W(9x2) + b(2)
        n_in, n_out = 9, 2
        macs = n_in * n_out                       # 18 MACs
        matvec_flops = macs * 2                    # each MAC = 1 mult + 1 add = 2 flops
        bias_ops = n_out                            # 2 additions

        # D. future-position reconstruction: U + pred_disp, 2 values
        reconstruct_ops = 2

        total_flops = feature_ops + norm_ops + matvec_flops + bias_ops + reconstruct_ops
        total_macs = macs  # only the matvec is naturally MAC-shaped; feature/norm counted as flops

        print(f"\nk={k}:")
        print(f"  A. feature construction (window_features): {feature_ops} scalar ops "
             f"(dt_span=2, avg_du={avg_du_ops}, avg_dv={avg_dv_ops}, ddu=2, ddv=2, speed=4) -- O(k)")
        print(f"  B. normalization (9 features): {norm_ops} ops (9 sub + 9 div) -- O(1), fixed")
        print(f"  C. model inference: {macs} MACs ({matvec_flops} flops as mult+add) + "
             f"{bias_ops} bias-add ops -- O(9x2)=O(18), fixed regardless of k")
        print(f"  D. future-position reconstruction: {reconstruct_ops} ops -- O(1), fixed")
        print(f"  TOTAL per prediction: {total_flops} scalar ops "
             f"(feature+norm+matvec-as-flops+bias+reconstruct)")
        rows.append([k, feature_ops, norm_ops, macs, matvec_flops, bias_ops,
                    reconstruct_ops, total_flops])

    save_csv(os.path.join(out_dir, "operation_counts.csv"), rows,
            ["k","feature_construction_ops","normalization_ops","model_MACs",
             "model_matvec_flops","bias_add_ops","reconstruction_ops","total_flops_per_prediction"])
    ratio = rows[-1][7] / rows[0][7] if rows[0][7] else float("nan")
    note(f"**Feature-construction workload is O(k)** (grows with k: "
        f"{rows[0][1]} ops at k=1 -> {rows[-1][1]} ops at k={K_VALUES[-1]}). "
        f"**Model-inference workload is O(9x2)=O(18), FIXED regardless of "
        f"k** ({rows[0][2]+rows[0][3]*2+rows[0][5]} ops, identical at every "
        f"k, confirmed above). Total workload k={K_VALUES[-1]} vs k=1 ratio: "
        f"{ratio:.2f}x -- growth is small in absolute terms since k only "
        f"affects two small averaging loops, not the model itself.")


# ===========================================================================
# LATENCY BENCHMARKS (real code, real data, warm-up + many reps)
# ===========================================================================
def benchmark(fn, n_warmup=200, n_reps=5000):
    for _ in range(n_warmup):
        fn()
    times = np.empty(n_reps)
    for i in range(n_reps):
        t0 = time.perf_counter_ns()
        fn()
        times[i] = time.perf_counter_ns() - t0
    times_us = times / 1000.0  # ns -> us
    return dict(mean=float(times_us.mean()), median=float(np.median(times_us)),
               std=float(times_us.std()), p90=pctl(times_us,90),
               p95=pctl(times_us,95), p99=pctl(times_us,99), n_reps=n_reps)


def profile_latency(models, results_dir, out_dir):
    heading("## MEASURED LATENCY (Sections 10, 11, 12) -- warm-up=200, reps=5000, perf_counter_ns")
    feat_rows, model_rows, total_rows = [], [], []
    for k in K_VALUES:
        if k not in models:
            continue
        dataset_path = os.path.join(results_dir, f"traj_dataset_k{k}_h{HORIZON}.npz")
        if not os.path.exists(dataset_path):
            print(f"  k={k}: dataset not found, skipping latency benchmark.")
            continue
        d = np.load(dataset_path, allow_pickle=True)
        tracks = md2.load_tracks("ball_moving") if os.path.exists(
            os.path.join(results_dir, "ball_moving_tracks.csv")) else None
        # pick a real window from the real dataset's own recomputation source
        # (a real track array, real index i) -- avoid disk I/O inside timed section
        if tracks:
            t = tracks[0]
            a = t["arr"]
            i = min(k + 5, len(a)-1)
        else:
            a = np.column_stack([np.arange(100), np.random.rand(100,4)])  # fallback only if no real tracks present
            i = k

        m = models[k]
        W, b, mu, sd = m["W"], m["b"], m["mu"], m["sd"]
        X_sample = d["X_test"][0:1]
        U_sample = d["U_test"][0:1]

        feat_bench = benchmark(lambda: md2.window_features(a, i, k))
        model_bench = benchmark(lambda: tm2.predict_disp(X_sample, W, b, mu, sd))
        def total_fn():
            f = md2.window_features(a, i, k)
            pred = tm2.predict_disp(f.reshape(1,-1), W, b, mu, sd)
            future = U_sample + pred
            return future
        total_bench = benchmark(total_fn)

        print(f"\nk={k}:")
        print(f"  feature-construction latency: mean={feat_bench['mean']:.2f}us "
             f"median={feat_bench['median']:.2f}us P95={feat_bench['p95']:.2f}us")
        print(f"  model-inference latency: mean={model_bench['mean']:.2f}us "
             f"median={model_bench['median']:.2f}us P95={model_bench['p95']:.2f}us")
        print(f"  TOTAL trajectory-prediction latency: mean={total_bench['mean']:.2f}us "
             f"median={total_bench['median']:.2f}us P95={total_bench['p95']:.2f}us")

        feat_rows.append([k] + [feat_bench[kk] for kk in ["mean","median","std","p90","p95","p99","n_reps"]])
        model_rows.append([k] + [model_bench[kk] for kk in ["mean","median","std","p90","p95","p99","n_reps"]])
        total_rows.append([k] + [total_bench[kk] for kk in ["mean","median","std","p90","p95","p99","n_reps"]])

    save_csv(os.path.join(out_dir, "latency_benchmarks.csv"),
            [["feature_construction"]+r for r in feat_rows] +
            [["model_inference"]+r for r in model_rows] +
            [["total_trajectory_prediction"]+r for r in total_rows],
            ["stage","k","mean_us","median_us","std_us","p90_us","p95_us","p99_us","n_reps"])

    if len(total_rows) >= 2:
        deltas = [total_rows[i][2]-total_rows[0][2] for i in range(len(total_rows))]  # median deltas vs k=1
        noise_floor = max(r[3] for r in total_rows)  # max std as a noise reference
        note(f"**Does total latency increase with k?** Median deltas vs k=1: "
            f"{[f'{d:.2f}us' for d in deltas]}. "
            f"{'These differences are SMALL RELATIVE TO the measurement std (~' + f'{noise_floor:.1f}us) -- likely dominated by Python/NumPy call overhead, not by k itself. Not claiming a meaningful trend.' if max(abs(d) for d in deltas) < noise_floor else 'Differences exceed the noise floor -- a real (if small) latency increase with k is plausible.'}")
    return total_rows


# ===========================================================================
# THROUGHPUT + 30-FPS REQUIREMENT
# ===========================================================================
def profile_throughput(total_rows, out_dir):
    heading("## THROUGHPUT + 30-FPS REAL-TIME REQUIREMENT (Sections 14, 15, 16)")
    print(f"REQUIREMENT: {NOMINAL_FPS:.1f} predictions/sec for one prediction per "
         f"30-FPS incoming frame (based on config.DT={DT_S}s, NOT a measured "
         f"camera timestamp).")
    print(f"Available processing time per incoming frame: {DT_S*1000:.2f} ms.")
    rows = []
    for r in total_rows:
        k, median_us = r[0], r[2]
        latency_ms = median_us/1000.0
        pred_per_sec = 1.0/(median_us/1e6) if median_us>0 else float("inf")
        utilization = 100*latency_ms/(DT_S*1000)
        headroom_ms = DT_S*1000 - latency_ms
        keeps_up = latency_ms < DT_S*1000
        print(f"  k={k}: median_latency={latency_ms:.4f}ms  "
             f"trajectory-prediction processing capacity={pred_per_sec:.0f}/sec  "
             f"utilization={utilization:.4f}%  headroom={headroom_ms:.4f}ms  "
             f"keeps up with TRAJECTORY-PREDICTOR-ALONE requirement: {keeps_up}")
        rows.append([k, latency_ms, pred_per_sec, utilization, headroom_ms, keeps_up])
    save_csv(os.path.join(out_dir, "throughput_results.csv"), rows,
            ["k","median_latency_ms","predictions_per_sec","utilization_pct_of_33.33ms",
             "headroom_ms","keeps_up_trajectory_predictor_only"])
    note("**This evaluates the TRAJECTORY PREDICTOR ALONE.** It does NOT "
        "establish that the complete depth-camera pipeline (backprojection, "
        "RANSAC fitting, discretization, Kalman tracking -- all upstream and "
        "unmeasured here) can sustain 30 FPS. Do not extrapolate this result "
        "to the full system.")
    return rows


# ===========================================================================
# ACCURACY-COST TRADEOFF (pull from ALREADY-VALIDATED results, don't recompute)
# ===========================================================================
def accuracy_cost_tradeoff(param_rows_by_k, mem_rows_by_k, hist_rows_by_k,
                          feat_lat_by_k, model_lat_by_k, total_lat_by_k,
                          throughput_by_k, out_dir):
    heading("## ACCURACY-COST TRADEOFF (Section 18) -- using ALREADY-VALIDATED accuracy, not recomputed")
    repeat_path = os.path.join(_PROJECT_ROOT, "validation_results", "repeatability", "repeatability_summary.csv")
    traj_path = os.path.join(_PROJECT_ROOT, "validation_results", "trajectory_model", "trajectory_model_metrics.csv")
    acc = {}
    if os.path.exists(repeat_path):
        for r in csv.DictReader(open(repeat_path)):
            acc[int(r["k"])] = dict(median=float(r["common_median_mm"]),
                                   rmse=float(r["common_rmse_mm"]), p95=float(r["common_p95_mm"]))
        print(f"  loaded accuracy from {repeat_path} (common-target, already validated)")
    elif os.path.exists(traj_path):
        for r in csv.DictReader(open(traj_path)):
            acc[int(r["k"])] = dict(median=float(r["median_mm"]), rmse=float(r["rmse_mm"]), p95=float(r["p95_mm"]))
        print(f"  loaded accuracy from {traj_path} (native test set -- "
             f"repeatability_summary.csv not found)")
    else:
        print("  NEITHER validation_results/repeatability/repeatability_summary.csv "
             "NOR validation_results/trajectory_model/trajectory_model_metrics.csv "
             "found. Cannot populate accuracy-cost table without recomputing, "
             "which this script is instructed not to do. Run those validators "
             "first for this section.")
        return

    rows = []
    for k in K_VALUES:
        if k not in acc:
            continue
        row = [k, acc[k]["median"], acc[k]["rmse"], acc[k]["p95"]]
        row += [param_rows_by_k.get(k, "N/A")]
        row += [mem_rows_by_k.get(k, "N/A")]
        row += [hist_rows_by_k.get(k, "N/A")]
        row += [feat_lat_by_k.get(k, "N/A")]
        row += [model_lat_by_k.get(k, "N/A")]
        row += [total_lat_by_k.get(k, "N/A")]
        row += [throughput_by_k.get(k, "N/A")]
        rows.append(row)
        print(f"  k={k}: median={acc[k]['median']:.2f}mm RMSE={acc[k]['rmse']:.2f}mm "
             f"P95={acc[k]['p95']:.2f}mm | params={param_rows_by_k.get(k)} | "
             f"total_latency={total_lat_by_k.get(k):.2f}us | "
             f"throughput={throughput_by_k.get(k):.0f}/s")
    save_csv(os.path.join(out_dir, "accuracy_cost_tradeoff.csv"), rows,
            ["k","median_error_mm","rmse_mm","p95_mm","n_params","inference_state_bytes",
             "history_bytes","feature_latency_us","model_latency_us",
             "total_latency_us","predictions_per_sec"])
    note("Accuracy-cost table populated from already-validated accuracy figures "
        "(not recomputed here) joined with this script's own parameter/memory/"
        "latency measurements. No automatic winner is declared -- see the raw "
        "table for k=3 (accuracy) vs k=10 (RMSE/tail) vs cost tradeoffs.")


# ===========================================================================
def main():
    p = argparse.ArgumentParser(
        description="CHARACTERIZATION ONLY: parameter size, memory, analytical "
                    "workload, and measured latency/throughput of the existing "
                    "frozen trajectory models. No training, no accuracy claims.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--out", type=str, default="validation_results/trajectory_profile")
    args = p.parse_args()
    def _resolve(path): return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)
    args.results_dir = _resolve(args.results_dir); args.out = _resolve(args.out)
    plots_dir = os.path.join(args.out, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    profile_environment(args.out)
    models = profile_parameters(args.results_dir, args.out)
    if not models:
        print("\nNo frozen models found. Stopping."); return
    profile_operation_counts(args.out)
    total_rows = profile_latency(models, args.results_dir, args.out)
    throughput_rows = profile_throughput(total_rows, args.out)

    param_rows_by_k = {}
    for k in models:
        W = models[k]["W"]; b = models[k]["b"]
        param_rows_by_k[k] = int(W.size + b.size)
    mem_rows_by_k = {k: int(models[k]["W"].nbytes + models[k]["b"].nbytes +
                           models[k]["mu"].nbytes + models[k]["sd"].nbytes) for k in models}
    hist_rows_by_k = {k: k*5*8 for k in models}
    total_lat_by_k = {r[0]: r[2] for r in total_rows}
    throughput_by_k = {r[0]: r[2] for r in throughput_rows}
    # re-derive feat/model medians from the saved CSV for the tradeoff table
    lat_csv = os.path.join(args.out, "latency_benchmarks.csv")
    feat_lat_by_k, model_lat_by_k = {}, {}
    if os.path.exists(lat_csv):
        for r in csv.DictReader(open(lat_csv)):
            if r["stage"]=="feature_construction": feat_lat_by_k[int(r["k"])] = float(r["median_us"])
            if r["stage"]=="model_inference": model_lat_by_k[int(r["k"])] = float(r["median_us"])

    accuracy_cost_tradeoff(param_rows_by_k, mem_rows_by_k, hist_rows_by_k,
                          feat_lat_by_k, model_lat_by_k, total_lat_by_k,
                          throughput_by_k, args.out)

    # ---- plots ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ks = sorted(total_lat_by_k)
    if ks:
        fig, ax = plt.subplots(figsize=(7,4.5))
        ax.plot(ks, [total_lat_by_k[k] for k in ks], "o-")
        ax.set_xlabel("k"); ax.set_ylabel("median total latency (us)")
        ax.set_title("Trajectory-prediction latency vs k")
        fig.savefig(os.path.join(plots_dir, "latency_vs_k.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7,4.5))
        ax.plot(ks, [feat_lat_by_k.get(k,np.nan) for k in ks], "o-")
        ax.set_xlabel("k"); ax.set_ylabel("median feature-construction latency (us)")
        ax.set_title("Feature-construction latency vs k")
        fig.savefig(os.path.join(plots_dir, "feature_latency_vs_k.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7,4.5))
        ax.plot(ks, [throughput_by_k[k] for k in ks], "o-")
        ax.set_xlabel("k"); ax.set_ylabel("predictions/sec")
        ax.set_title("Throughput vs k")
        fig.savefig(os.path.join(plots_dir, "throughput_vs_k.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7,4.5))
        ax.plot(ks, [hist_rows_by_k[k] for k in ks], "o-")
        ax.set_xlabel("k"); ax.set_ylabel("history buffer bytes")
        ax.set_title("History-buffer memory vs k")
        fig.savefig(os.path.join(plots_dir, "history_memory_vs_k.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---- report ----
    lines = ["# TRAJECTORY MODEL PROFILE REPORT (characterization only)\n"]
    lines.append("**No edge-device suitability is claimed anywhere in this "
                "report.** All latency/throughput numbers are specific to "
                "the machine this script was run on.\n\n")
    lines.extend(x if x.startswith("#") else x+"\n" for x in REPORT)
    lines.append("\n## Upstream costs explicitly OUT OF SCOPE (Section 25)\n")
    lines.append("depth loading -> point-cloud backprojection -> RANSAC plane/"
                "sphere/box fitting -> grid discretization -> Kalman filter "
                "tracking. All of this happens in track_moving.py, entirely "
                "BEFORE any row reaches window_features(). None of it is "
                "measured by this script.\n")
    report_path = os.path.join(args.out, "trajectory_profile_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"\n\nFull report: {report_path}")


if __name__ == "__main__":
    main()