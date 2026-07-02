import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_CSV = "../benchmark/results_cuda_global.csv"
OUTPUT_PNG = "benchmark_barplot.png"

# Metrics to plot (in the order you want them grouped per N)
METRICS = ["TimeHostToDevice", "ExecutionTime", "TimeDeviceToHost"]#, "TotalTime"]
METRIC_LABELS = ["Host -> Device", "Execution", "Device -> Host"]#, "Total"]

# ---- Load data ----
df = pd.read_csv(INPUT_CSV)

# Convert seconds -> milliseconds for readability
for m in METRICS:
    df[m] = df[m] * 1e3

# ---- Aggregate: mean and std per N ----
grouped = df.groupby("N")[METRICS].agg(["mean", "std"])
N_values = grouped.index.to_list()

means = {m: grouped[m]["mean"].values for m in METRICS}
stds = {m: grouped[m]["std"].values for m in METRICS}

# ---- Plot ----
x = np.arange(len(N_values))          # one group per N
n_metrics = len(METRICS)
bar_width = 0.8 / n_metrics

fig, ax = plt.subplots(figsize=(9, 6))

colors = plt.cm.Greens(np.linspace(0.15, 0.85, n_metrics))


for i, (metric, label) in enumerate(zip(METRICS, METRIC_LABELS)):
    offset = (i - (n_metrics - 1) / 2) * bar_width
    ax.bar(
        x + offset,
        means[metric],
        width=bar_width,
        yerr=stds[metric],
        capsize=4,
        label=label,
        color=colors[i],
        edgecolor="black",
        linewidth=0.5,
        zorder=2
    )

ax.set_xlabel("N (FFT size)")
ax.set_ylabel("Time (ms)")
ax.set_yscale("log")  # timings span more than an order of magnitude across N
ax.set_title("CUDA FFT Timing Breakdown vs N (mean +/- std over runs)")
ax.set_xticks(x)
ax.set_xticklabels([str(n) for n in N_values])
ax.legend(title="Metric")
ax.grid(axis="y", linestyle="--", alpha=0.5)

# fig.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=600)