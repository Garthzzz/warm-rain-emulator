# build_viz_report.py
# Purpose: end-to-end visualization pipeline for (time, IC, sims, feature) data.
# 1) Build a lightweight subset from data_mini/ into viz_sample/
# 2) Produce multiple plots (PNG) into viz_out/
# 3) Write a text summary (report.txt)
#
# Expected inputs (already prepared in your workflow):
#   E:/warm rain/warm-rain-emulator/data_mini/inputs_all.npy      # (T, IC, N, 9)
#   E:/warm rain/warm-rain-emulator/data_mini/outputs_all.npy     # (T, IC, N, 4)
#   E:/warm rain/warm-rain-emulator/data_mini/tendencies.npy      # (T, IC, N, 4)
#
# This script is safe: it uses slicing and memory mapping, never loads full huge arrays into RAM.

import os
import numpy as np

# Use non-interactive backend for PNG saving
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# Config (edit paths if needed)
# -----------------------------
ROOT = r"E:/warm rain/warm-rain-emulator"
SRC_DIR = os.path.join(ROOT, "data_mini")
VIZ_DIR = os.path.join(ROOT, "viz_sample")
OUT_DIR = os.path.join(ROOT, "viz_out")

# Sampling config for building viz_sample
TIME_MAX  = 600     # take first 600 time steps (reduce if needed)
IC_STRIDE = 10      # take every 10th vertical level (~719 -> ~72)
SIM_TAKE  = 5       # take first 5 ensemble members

# Which indices to plot (you can tweak)
HOV_FEATS =  [0, 1, 2]   # features to show in Hovmoller
VERT_FEAT =  0           # vertical profile feature
TS_FEAT   =  0           # time-series feature
HIST_FEAT = 0            # histogram feature
SIM_IDX   = 0            # which sim to pick when a single sim is needed
TIME_IDX  = 0            # which time to pick for vertical/hist/corr
IC_IDX    = 5            # which downsampled IC index for time-series/hist/corr

# Files to process
FILES = {
    "inputs":     "inputs_all.npy",
    "outputs":    "outputs_all.npy",
    "tendencies": "tendencies.npy",
}

# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def build_viz_subset():
    """Create lightweight subset into viz_sample/."""
    ensure_dir(VIZ_DIR)
    for key, fname in FILES.items():
        src_path = os.path.join(SRC_DIR, fname)
        if not os.path.exists(src_path):
            print(f"[WARN] Missing: {src_path} (skip)")
            continue
        arr = np.load(src_path, mmap_mode="r")   # (time, IC, sims, feature)
        t = slice(0, min(TIME_MAX, arr.shape[0]))
        ic_idx = np.arange(0, arr.shape[1], IC_STRIDE, dtype=int)
        s = slice(0, min(SIM_TAKE, arr.shape[2]))
        sub = arr[t][:, ic_idx][:, :, s]         # (time, IC_sub, sims_sub, feature)
        out_path = os.path.join(VIZ_DIR, fname.replace(".npy", "_viz.npy"))
        np.save(out_path, sub)
        print(f"[VIZ-SAMPLE] saved: {out_path}  shape: {sub.shape}")

def load_viz(name: str):
    """Load viz-sample array by logical name ('inputs'/'outputs'/'tendencies')."""
    path = os.path.join(VIZ_DIR, FILES[name].replace(".npy", "_viz.npy"))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}. Run sampling first.")
    return np.load(path)  # (time, IC_sub, sims_sub, feature)

def savefig_safe(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] saved: {path}")

def summary_line(arr_name: str, arr: np.ndarray) -> str:
    return f"{arr_name}: shape={tuple(arr.shape)}  (time, IC_sub, sims_sub, feature)"

# -----------------------------
# Plots
# -----------------------------
def plot_hovmoller(arr: np.ndarray, title_prefix: str, out_prefix: str, sim_idx: int, feat_list):
    """Time-height heatmap: field = arr[:, :, sim_idx, feat_idx].T"""
    T, ICs, Ns, F = arr.shape
    for feat_idx in feat_list:
        if feat_idx >= F:
            continue
        field = arr[:, :, sim_idx, feat_idx]  # (time, IC_sub)
        plt.figure()
        plt.imshow(field.T, aspect='auto', origin='lower')
        plt.title(f"{title_prefix} | Hovmöller (sim={sim_idx}, feat={feat_idx})")
        plt.xlabel("time index (downsampled)")
        plt.ylabel("IC level (downsampled)")
        plt.colorbar(label="value")
        out_path = os.path.join(OUT_DIR, f"{out_prefix}_hov_sim{sim_idx}_feat{feat_idx}.png")
        savefig_safe(out_path)

def plot_vertical_profile(arr: np.ndarray, time_idx: int, feat_idx: int, title_prefix: str, out_prefix: str):
    """At given time, vertical profile with mean and 5-95 percentile across sims."""
    if time_idx >= arr.shape[0]:
        return
    if feat_idx >= arr.shape[-1]:
        return
    slice_t = arr[time_idx, :, :, feat_idx]  # (IC_sub, sims_sub)
    mean_prof = slice_t.mean(axis=1)
    p5  = np.percentile(slice_t, 5, axis=1)
    p95 = np.percentile(slice_t, 95, axis=1)
    ic = np.arange(slice_t.shape[0])

    plt.figure()
    plt.plot(mean_prof, ic, label="mean across sims")
    plt.fill_betweenx(ic, p5, p95, alpha=0.3, label="5–95% band")
    plt.gca().invert_yaxis()  # optional: put surface at bottom
    plt.xlabel("value")
    plt.ylabel("IC level (downsampled)")
    plt.title(f"{title_prefix} | Vertical profile @ time={time_idx}, feat={feat_idx}")
    plt.legend()
    out_path = os.path.join(OUT_DIR, f"{out_prefix}_vprof_t{time_idx}_feat{feat_idx}.png")
    savefig_safe(out_path)

def plot_time_series_spread(arr: np.ndarray, ic_idx: int, feat_idx: int, title_prefix: str, out_prefix: str):
    """At given IC, time series mean with 25–75% band across sims."""
    if ic_idx >= arr.shape[1]:
        return
    if feat_idx >= arr.shape[-1]:
        return
    ts = arr[:, ic_idx, :, feat_idx]  # (time, sims_sub)
    mean_ts = ts.mean(axis=1)
    p25 = np.percentile(ts, 25, axis=1)
    p75 = np.percentile(ts, 75, axis=1)
    t = np.arange(ts.shape[0])

    plt.figure()
    plt.plot(t, mean_ts, label="mean across sims")
    plt.fill_between(t, p25, p75, alpha=0.3, label="25–75% band")
    plt.xlabel("time index (downsampled)")
    plt.ylabel("value")
    plt.title(f"{title_prefix} | Time series @ IC={ic_idx}, feat={feat_idx}")
    plt.legend()
    out_path = os.path.join(OUT_DIR, f"{out_prefix}_ts_ic{ic_idx}_feat{feat_idx}.png")
    savefig_safe(out_path)

def plot_histogram(arr: np.ndarray, time_idx: int, ic_idx: int, feat_idx: int, title_prefix: str, out_prefix: str):
    """Histogram across sims at a fixed (time, IC) for one feature."""
    if time_idx >= arr.shape[0] or ic_idx >= arr.shape[1] or feat_idx >= arr.shape[-1]:
        return
    vals = arr[time_idx, ic_idx, :, feat_idx]  # (sims_sub,)
    plt.figure()
    plt.hist(vals, bins=30)
    plt.title(f"{title_prefix} | Hist across sims @ t={time_idx}, IC={ic_idx}, feat={feat_idx}")
    plt.xlabel("value")
    plt.ylabel("count")
    out_path = os.path.join(OUT_DIR, f"{out_prefix}_hist_t{time_idx}_ic{ic_idx}_feat{feat_idx}.png")
    savefig_safe(out_path)

def plot_feature_corr_inputs(arr_inputs: np.ndarray, time_idx: int, ic_idx: int):
    """Correlation among input features at (time, IC) across sims."""
    if time_idx >= arr_inputs.shape[0] or ic_idx >= arr_inputs.shape[1]:
        return
    X = arr_inputs[time_idx, ic_idx, :, :]  # (sims_sub, 9)
    if X.shape[-1] < 2:
        return
    C = np.corrcoef(X, rowvar=False)

    plt.figure()
    plt.imshow(C, vmin=-1, vmax=1)
    plt.colorbar(label="corr")
    plt.title(f"Inputs | Feature correlation @ time={time_idx}, IC={ic_idx}")
    plt.xlabel("feature")
    plt.ylabel("feature")
    out_path = os.path.join(OUT_DIR, f"inputs_corr_t{time_idx}_ic{ic_idx}.png")
    savefig_safe(out_path)

# -----------------------------
# Main
# -----------------------------
def main():
    ensure_dir(OUT_DIR)

    # 1) Build viz subset (lightweight)
    build_viz_subset()

    # 2) Load viz arrays
    try:
        a_in  = load_viz("inputs")
        a_out = load_viz("outputs")
        a_ten = load_viz("tendencies")
    except FileNotFoundError as e:
        print("[ERROR]", e)
        return

    # 3) Report basic summary
    report_lines = []
    report_lines.append(summary_line("inputs_viz", a_in))
    report_lines.append(summary_line("outputs_viz", a_out))
    report_lines.append(summary_line("tend_viz", a_ten))

    # 4) Hovmöller (time-height) for selected features
    plot_hovmoller(a_in,  "Inputs",     "inputs",     SIM_IDX, HOV_FEATS)
    plot_hovmoller(a_out, "Outputs",    "outputs",    SIM_IDX, HOV_FEATS[:min(2, a_out.shape[-1])])
    plot_hovmoller(a_ten, "Tendencies", "tendencies", SIM_IDX, HOV_FEATS[:min(2, a_ten.shape[-1])])

    # 5) Vertical profiles @ TIME_IDX
    plot_vertical_profile(a_in,  TIME_IDX, VERT_FEAT, "Inputs",     "inputs")
    plot_vertical_profile(a_out, TIME_IDX, min(VERT_FEAT, a_out.shape[-1]-1), "Outputs",    "outputs")
    plot_vertical_profile(a_ten, TIME_IDX, min(VERT_FEAT, a_ten.shape[-1]-1), "Tendencies", "tendencies")

    # 6) Time series with spread @ IC_IDX
    plot_time_series_spread(a_in,  IC_IDX, TS_FEAT, "Inputs",     "inputs")
    plot_time_series_spread(a_out, IC_IDX, min(TS_FEAT, a_out.shape[-1]-1), "Outputs",    "outputs")
    plot_time_series_spread(a_ten, IC_IDX, min(TS_FEAT, a_ten.shape[-1]-1), "Tendencies", "tendencies")

    # 7) Histogram @ (TIME_IDX, IC_IDX)
    plot_histogram(a_in,  TIME_IDX, IC_IDX, HIST_FEAT, "Inputs",     "inputs")
    plot_histogram(a_out, TIME_IDX, IC_IDX, min(HIST_FEAT, a_out.shape[-1]-1), "Outputs",    "outputs")
    plot_histogram(a_ten, TIME_IDX, IC_IDX, min(HIST_FEAT, a_ten.shape[-1]-1), "Tendencies", "tendencies")

    # 8) Feature correlation (inputs only)
    plot_feature_corr_inputs(a_in, TIME_IDX, IC_IDX)

    # 9) Write report
    ensure_dir(OUT_DIR)
    rpt = os.path.join(OUT_DIR, "report.txt")
    with open(rpt, "w", encoding="utf-8") as f:
        f.write("Visualization report (lightweight subset)\n")
        f.write(f"ROOT = {ROOT}\nSRC_DIR = {SRC_DIR}\nVIZ_DIR = {VIZ_DIR}\nOUT_DIR = {OUT_DIR}\n")
        f.write(f"\nSampling: TIME_MAX={TIME_MAX}, IC_STRIDE={IC_STRIDE}, SIM_TAKE={SIM_TAKE}\n")
        f.write("\nShapes:\n")
        for line in report_lines:
            f.write("  " + line + "\n")
        f.write("\nNotes:\n  - Hovmöller: time-height for selected (sim, feature)\n")
        f.write("  - Vertical profile: mean across sims with 5–95% band\n")
        f.write("  - Time series: mean with 25–75% band across sims\n")
        f.write("  - Histogram: per (time, IC) across sims\n")
        f.write("  - Correlation: inputs feature correlation per (time, IC)\n")
    print(f"[REPORT] {rpt}")

if __name__ == "__main__":
    main()
