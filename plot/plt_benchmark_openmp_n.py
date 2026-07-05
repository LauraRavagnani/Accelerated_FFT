import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np
import pandas as pd
import scipy

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Plot benchmark performance for a given N.')
parser.add_argument('--nth', type=int)
parser.add_argument('--color', type=str, default='blue',
                     help='Color for the data points (e.g. blue, red, #ff5733)')

args = parser.parse_args()
point_color = args.color
n_threads = args.nth

try:
    df = pd.read_csv('../benchmark/results_openmp.csv')
except Exception as e:
    print(f"Error reading file: {e}")

n = np.unique(df['N'])
mean_time = df[df['NThreads']==n_threads].groupby('N')['ExecutionTime'].mean().values
std_time = df[df['NThreads']==n_threads].groupby('N')['ExecutionTime'].std().values

plt.figure(figsize=(10, 6))
plt.errorbar(n, mean_time, yerr=std_time,
            fmt='o--',
            ms=5,
            capsize=3,
            linewidth=1,
            color=point_color, 
            label='iterative FFT openMP',
            zorder=2)


plt.title(f'Algorithm Performance Analysis, NTh = {n_threads}', fontsize=14)
plt.xlabel('Dataset size N', fontsize=12)
plt.ylabel('Execution Time (seconds)', fontsize=12)
plt.xscale('log', base=2)
plt.xticks(n, labels=n)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
	
plt.savefig(f'benchmark_openmp_{n_threads}.png', dpi=600)