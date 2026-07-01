import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np
import pandas as pd


try:
	df = pd.read_csv('../benchmark/results_tune_tpb.csv')
	
except Exception as e:
	print(f"Error reading file: {e}")
	
n_tpb = np.unique(df['NThreadsPerBlock'])
mean_time = df.groupby('NThreadsPerBlock')['ExecutionTime'].mean().values
std_time = df.groupby('NThreadsPerBlock')['ExecutionTime'].std().values
# 2. Create the visualization
plt.figure(figsize=(10, 6))
plt.errorbar(n_tpb, mean_time, yerr=std_time,
			fmt='o',
			ms=5,
			capsize=3,
			color='red', 
			#label='iterative FFT openMP',
			zorder=2)
# 3. Add labels and styling
plt.title('Tuning Number of Thread per Block, N = 512', fontsize=14)
plt.xlabel('Number of threads per block', fontsize=12)
plt.ylabel('Execution Time (seconds)', fontsize=12)
plt.xscale('log', base=2)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.xticks(n_tpb, labels=['32', '64', '128', '256'])
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
# 4. Show the result	
plt.savefig('tune_tpb.png')