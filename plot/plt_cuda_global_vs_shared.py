import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np
import pandas as pd
import scipy

try:
    df1 = pd.read_csv('../benchmark/results_cuda.csv')
    df2 = pd.read_csv('../benchmark/results_cuda_shared.csv')
    
except Exception as e:
    print(f"Error reading file: {e}")
    
n = np.unique(df1['N'])
x = np.arange(512, 4096)

mean_time1 = df1.groupby('N')['ExecutionTime'].mean().values
mean_time2 = df2.groupby('N')['ExecutionTime'].mean().values

std_time1 = df1.groupby('N')['ExecutionTime'].std().values
std_time2 = df2.groupby('N')['ExecutionTime'].std().values

plt.figure(figsize=(10, 6))
plt.errorbar(n, mean_time1, yerr=std_time1,
            fmt='o',
            ms=5,
            capsize=3,
            color='red', 
            label='global',
            zorder=2)
plt.errorbar(n, mean_time2, yerr=std_time2,
            fmt='o',
            ms=5,
            capsize=3,
            color='blue', 
            label='shared',
            zorder=2)

plt.title('Benchmark CUDA, TPB = 256', fontsize=14)
plt.xlabel('Dataset size N', fontsize=12)
plt.ylabel('Execution Time (seconds)', fontsize=12)
plt.xscale('log', base=2)
plt.xticks(n, labels=n)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
	
plt.savefig('benchmark_cuda_global_vs_shared.png')