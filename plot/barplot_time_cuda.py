import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_CSV = "../benchmark/results_cuda.csv"
OUTPUT_PNG = "benchmark_barplot.png"

METRICS = ["TimeHostToDevice", "ExecutionTime", "TimeDeviceToHost"]
METRIC_LABELS = ["Host -> Device", "Execution", "Device -> Host"]

df = pd.read_csv(INPUT_CSV)

# Convert seconds -> milliseconds
for m in METRICS:
    df[m] = df[m] * 1e3

grouped = df.groupby("N")[METRICS].agg(["mean", "std"])
N_values = np.unique(df['N'])

means = {m: grouped[m]["mean"].values for m in METRICS}
stds = {m: grouped[m]["std"].values for m in METRICS}

x = np.arange(len(N_values))       
n_metrics = len(METRICS)
bar_width = 0.8 / n_metrics

fig, ax = plt.subplots(figsize=(9, 6))

colors = plt.cm.YlGn(np.linspace(0.15, 0.85, n_metrics))

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

ax.set_xlabel("Matrix Size (N)")
ax.set_ylabel("Time (ms)")
ax.set_yscale("log")  # timings span more than an order of magnitude across N
ax.set_title("CUDA FFT Timing Breakdown vs N (mean +/- std)")
ax.set_xticks(x)
ax.set_xticklabels([str(n) for n in N_values])
ax.legend(fontsize=14)
ax.grid(axis="y", linestyle="--", alpha=0.5)

fig.savefig(OUTPUT_PNG, dpi=600)