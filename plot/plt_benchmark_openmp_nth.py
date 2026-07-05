import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np
import pandas as pd
import scipy

parser = argparse.ArgumentParser(description='Plot benchmark performance for a given N.')
parser.add_argument('--n', type=int, help='Value of N to filter and plot (e.g. 512)')
parser.add_argument('--color', type=str, default='blue',
                     help='Color for the data points (e.g. blue, red, #ff5733)')
args = parser.parse_args()
n = args.n
point_color = args.color

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

print(np.min(mean_time), n_threads[np.argmin(mean_time)])

(a, b), _, infodict, _, _ = scipy.optimize.curve_fit(model, n_threads, mean_time, sigma=std_time, full_output=True)
chi2 = np.sum(infodict['fvec']**2)

n_params = 2  # a, b
dof = len(n_threads) - n_params
red_chi2 = chi2 / dof   # reduced chi2

plt.figure(figsize=(10, 6))
plt.errorbar(n_threads, mean_time, yerr=std_time,
            fmt='o',
            ms=5,
            capsize=3,
            color=point_color, # deepskyblue, cornflowerblue, royalblue, slateblue
            label='iterative FFT openMP',
            zorder=2)
plt.plot(x, a / x + b, 
        linestyle='-',
        color=point_color,
        alpha=0.3,
        linewidth=1, 
        label=f"Amdahl's law fit\n"
             f"$\\chi^2_\\nu$={red_chi2:.3f}",
        zorder=2)

plt.title(f'Algorithm Performance Analysis, N = {n}', fontsize=14)
plt.xlabel('Number of threads', fontsize=12)
plt.ylabel('Execution Time (seconds)', fontsize=12)
# plt.xscale('log', base=2)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
	
plt.savefig(f'benchmark_openmp_{n}.png', dpi=600)