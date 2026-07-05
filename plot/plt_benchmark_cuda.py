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

mean_time = df.groupby('N')['ExecutionTime'].mean().values
std_time = df.groupby('N')['ExecutionTime'].std().values

def model(x, a):
		return a * x**2 * np.log(x)

a, _ = scipy.optimize.curve_fit(model, n, mean_time, sigma=std_time)

plt.figure(figsize=(10, 6))
plt.errorbar(n, mean_time, yerr=std_time,
            fmt='o',
            ms=5,
            capsize=3,
            color='red', 
            #label='iterative FFT openMP',
            zorder=2)

plt.title('Benchmark CUDA, TPB = 256', fontsize=14)
plt.plot(x, a*x*x*np.log(x), 
			linestyle='-',        
			color="lightcoral",     
			linewidth=1, 
			label='$\\mathcal{O}(N^2\\log N)$',
			zorder=2)
plt.xlabel('Dataset size N', fontsize=12)
plt.ylabel('Execution Time (seconds)', fontsize=12)
plt.xscale('log', base=2)
plt.xticks(n, labels=n)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.savefig('benchmark_cuda.png')