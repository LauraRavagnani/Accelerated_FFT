import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np
import pandas as pd
import scipy

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Plot benchmark performance for a given N.')
parser.add_argument('--n', type=int, help='Value of N to filter and plot (e.g. 512)')
parser.add_argument('--color', type=str, default='blue',
                     help='Color for the data points (e.g. blue, red, #ff5733)')
# parser.add_argument('--line-color', type=str, default='aquamarine',
#                      help='Color for the Amdahl\'s law fit line')
args = parser.parse_args()
n = args.n
point_color = args.color
# line_color = args.line_color

try:
    df = pd.read_csv('../benchmark/results_openmp.csv')
except Exception as e:
    print(f"Error reading file: {e}")

x = np.arange(1, 28.1, 0.1)

### Amdahl's law
def model(x, a, b):
    return a / x + b

n_threads = np.unique(df['NThreads'])
mean_time = df[df['N']==n].groupby('NThreads')['ExecutionTime'].mean().values
std_time = df[df['N']==n].groupby('NThreads')['ExecutionTime'].std().values
(a, b), _ = scipy.optimize.curve_fit(model, n_threads, mean_time, sigma=std_time)

# 2. Create the visualization
plt.figure(figsize=(10, 6))
plt.errorbar(n_threads, mean_time, yerr=std_time,
            fmt='o',
            ms=5,
            capsize=3,
            color=point_color, 
            label='iterative FFT openMP',
            zorder=2)
plt.plot(x, a / x + b, 
        linestyle='-',
        color=point_color,
        alpha=0.3,
        linewidth=1, 
        label='Amdahl\'s law',
        zorder=2)

# 3. Add labels and styling
plt.title(f'Algorithm Performance Analysis, N = {n}', fontsize=14)
plt.xlabel('Number of threads', fontsize=12)
plt.ylabel('Execution Time (seconds)', fontsize=12)
# plt.xscale('log', base=2)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

# 4. Show the result	
plt.savefig(f'benchmark_openmp_{n}.png')