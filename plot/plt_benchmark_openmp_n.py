import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np
import pandas as pd
import scipy

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Plot benchmark performance for a given N.')
# parser.add_argument('--n', type=int, help='Value of N to filter and plot (e.g. 512)')
parser.add_argument('--nth', type=int)
parser.add_argument('--color', type=str, default='blue',
                     help='Color for the data points (e.g. blue, red, #ff5733)')

args = parser.parse_args()
# n = args.n
point_color = args.color
n_threads = args.nth

try:
    df = pd.read_csv('../benchmark/results_openmp.csv')
except Exception as e:
    print(f"Error reading file: {e}")

# x = np.arange(1, 28.1, 0.1)

# ### Amdahl's law
# def model(x, a, b):
#     return a / x + b

# n_threads = np.unique(df['NThreads'])
n = np.unique(df['N'])
mean_time = df[df['NThreads']==n_threads].groupby('N')['ExecutionTime'].mean().values
std_time = df[df['NThreads']==n_threads].groupby('N')['ExecutionTime'].std().values

# print(np.min(mean_time), n_threads[np.argmin(mean_time)])

# (a, b), _ = scipy.optimize.curve_fit(model, n_threads, mean_time, sigma=std_time)

# 2. Create the visualization
plt.figure(figsize=(10, 6))
plt.errorbar(n, mean_time, yerr=std_time,
            fmt='o--',
            ms=5,
            capsize=3,
            linewidth=1,
            color=point_color, 
            label='iterative FFT openMP',
            zorder=2)

# plt.plot(x, a / x + b, 
#         linestyle='-',
#         color=point_color,
#         alpha=0.3,
#         linewidth=1, 
#         label='Amdahl\'s law',
#         zorder=2)

# 3. Add labels and styling
plt.title(f'Algorithm Performance Analysis, NTh = {n_threads}', fontsize=14)
plt.xlabel('Dataset size N', fontsize=12)
plt.ylabel('Execution Time (seconds)', fontsize=12)
plt.xscale('log', base=2)
plt.xticks(n, labels=n)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

# 4. Show the result	
plt.savefig(f'benchmark_openmp_{n_threads}.png', dpi=600)