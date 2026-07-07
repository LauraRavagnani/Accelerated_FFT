import matplotlib.pyplot as plt
import matplotlib.colors as col
from mpl_toolkits.mplot3d import Axes3D
import json
import numpy as np
import pandas as pd
import scipy

df1 = pd.read_csv('../benchmark/results_iterative_fft.csv')
df2 = pd.read_csv('../benchmark/results_ditfft2.csv')

x = np.arange(64, 4096)

def model(x, a):
    return a * x**2 * np.log(x)

N = np.unique(df1['N'])

mean_time1 = df1.groupby('N')['ExecutionTime'].mean().values
mean_time2 = df2.groupby('N')['ExecutionTime'].mean().values

std_time1 = df1.groupby('N')['ExecutionTime'].std().values
std_time2 = df2.groupby('N')['ExecutionTime'].std().values

# print(mean_time1)
# print(mean_time2)

a1, _ = scipy.optimize.curve_fit(model, N, mean_time1, sigma=std_time1)
a2, _ = scipy.optimize.curve_fit(model, N, mean_time2, sigma=std_time2)

# Best-fit values
fit1 = model(N, *a1)
fit2 = model(N, *a2)

# Chi-squared
chi2_1 = np.sum(((mean_time1 - fit1) / std_time1)**2)
chi2_2 = np.sum(((mean_time2 - fit2) / std_time2)**2)

# Degrees of freedom
ndof = len(N) - len(a1)   # one fitted parameter

# Reduced chi-squared
chi2_red_1 = chi2_1 / ndof
chi2_red_2 = chi2_2 / ndof

# 2. Create the visualization
plt.figure(figsize=(10, 6))
plt.errorbar(N, mean_time1, yerr=std_time1,
            fmt='o',
            ms=5,
            capsize=3,
            color="#ac7bbd", 
            label='iterative FFT serial',
            zorder=2)
plt.errorbar(N, mean_time2, yerr=std_time2,
            fmt='s',
            ms=5,
            capsize=3,
            color="#cdb94e", 
            label='ditfft2 FFT serial',
            zorder=2)
plt.plot(x, a1*x*x*np.log(x), 
        linestyle='-',        # Solid line
        color="#ac7bbd",      # Red color
        alpha=0.5,
        linewidth=1, 
        label='$\\mathcal{O}(N^2\\log N)$,' f'$\\chi^2_\\nu={chi2_red_1:.2f}$',
        zorder=2)
plt.plot(x, a2*x*x*np.log(x), 
        linestyle='-',        # Solid line
        color="#cdb94e",      # Red color
        alpha=0.5,
        linewidth=1, 
        label='$\\mathcal{O}(N^2\\log N)$,' f'$\\chi^2_\\nu={chi2_red_2:.2f}$',
        zorder=1)



# 3. Add labels and styling
plt.title('Out-of-place vs In-place Performance', fontsize=14)
plt.xlabel('Matrix Size (N)', fontsize=12)
plt.ylabel('Execution Time (seconds)', fontsize=12)
plt.xscale('log', base=2)
plt.xticks(N, labels=N)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=14)

# 4. Show the result	
plt.savefig('serial.png', dpi=300)