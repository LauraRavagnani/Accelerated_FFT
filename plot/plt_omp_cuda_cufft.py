import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np
import pandas as pd
import scipy

try:
    df1 = pd.read_csv('../benchmark/results_openmp.csv')
    df2 = pd.read_csv('../benchmark/results_cuda.csv')
    df3 = pd.read_csv('../benchmark/results_cufft.csv')
except Exception as e:
    print(f"Error reading file: {e}")

n = np.unique(df1['N'])

mean_time1 = df1[df1['NThreads']==22].groupby('N')['ExecutionTime'].mean().values
std_time1 = df1[df1['NThreads']==22].groupby('N')['ExecutionTime'].std().values

mean_time2 = df2.groupby('N')['ExecutionTime'].mean().values
std_time2 = df2.groupby('N')['ExecutionTime'].std().values

mean_time3 = df3.groupby('N')['ExecutionTime'].mean().values
std_time3 = df3.groupby('N')['ExecutionTime'].std().values

plt.figure(figsize=(10, 6))
plt.errorbar(n, mean_time1, yerr=std_time1,
            fmt='o--',
            ms=5,
            capsize=3,
            linewidth=1,
            color='#5983c2', 
            label='openMP iterative',
            zorder=2)

plt.errorbar(n, mean_time2, yerr=std_time2,
            fmt='*--',
            ms=7,
            capsize=3,
            linewidth=1,
            color='#5da272', 
            label='CUDA custom',
            zorder=2)

plt.errorbar(n, mean_time3, yerr=std_time3,
fmt='s--',
ms=5,
capsize=3,
linewidth=1,
color='#db7a57', 
label='cuFFT library',
zorder=2)

plt.title(f'Algorithm Performance Analysis', fontsize=14)
plt.xlabel('Dataset size N', fontsize=12)
plt.ylabel('Execution Time (seconds)', fontsize=12)
plt.xscale('log', base=2)
plt.xticks(n, labels=n)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
	
plt.savefig(f'benchmark_omp_cuda_cufft.png', dpi=600)