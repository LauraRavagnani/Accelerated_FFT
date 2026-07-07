import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np
import pandas as pd
import scipy

try:
    df = pd.read_csv('../benchmark/results_cuda.csv')
    
except Exception as e:
    print(f"Error reading file: {e}")
    
n = np.unique(df['N'])
x = np.arange(512, 4096)

max_gflops = [254.4 for i in range(len(x))]
max_gbps = [320 for i in range(len(x))]

mean_gflops = df.groupby('N')['GFlops'].mean().values
std_gflops = df.groupby('N')['GFlops'].std().values

mean_gbps1 = df.groupby('N')['GbpsHtD'].mean().values
std_gbps1 = df.groupby('N')['GbpsHtD'].std().values

mean_gbps2 = df.groupby('N')['GbpsDtH'].mean().values
std_gbps2 = df.groupby('N')['GbpsDtH'].std().values

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
 
ax1.errorbar(
    n, mean_gflops, yerr=std_gflops,
    marker="o", capsize=4, color="forestgreen", linestyle='--', label="GFlops"
)
ax1.plot(
    x, max_gflops, color="red", linestyle='--', label="max FLOPS: 254.4 GFLOPS (FP64)"
)
ax1.set_xlabel("N", fontsize=12)
ax1.set_ylabel("GFlops", fontsize=12)
ax1.set_xscale('log', base=2)
ax1.set_yscale('log')
ax1.set_xticks(n, labels=n)
ax1.set_title("Compute throughput vs N")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(fontsize=14)

ax2.errorbar(
    n, mean_gbps1, yerr=std_gbps1,
    marker="o", capsize=4, color="yellowgreen", linestyle='--', label="Host to Device"
)
ax2.errorbar(
    n, mean_gbps2, yerr=std_gbps2,
    marker="s", capsize=4, color="teal", linestyle='--', label="Device to Host"
)
ax2.plot(
    x, max_gbps, color="black", linestyle='--', label="max Bandwidth: 320 GB/s"
)
ax2.set_xlabel("N", fontsize=12)
ax2.set_ylabel("GB/s", fontsize=12)
ax2.set_xscale('log', base=2)
ax2.set_yscale('log')
ax2.set_xticks(n, labels=n)
ax2.set_title("Transfer bandwidth vs N")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(fontsize=14)

fig.tight_layout()
fig.savefig('throughput_bandwidth.png', dpi=600)
