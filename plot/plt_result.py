import matplotlib.pyplot as plt
import matplotlib.colors as col
from mpl_toolkits.mplot3d import Axes3D
import json
import numpy as np
import pandas as pd

with open('../fft_res_omp.json', 'r') as f:
    data = json.load(f)
    
x = [item['x'] for item in data]
y = [item['y'] for item in data]
z = [item['z'] for item in data]

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Draw the 'heads' of the stems
ax.scatter(x, y, z, color="#ac7bbd", s=50, label='FFT Peaks')

# Draw the 'stems' (lines from the floor to the peak)
for xi, yi, zi in zip(x, y, z):
    ax.plot([xi, xi], [yi, yi], [0, zi], color='black', alpha=0.6, linewidth=1)

# Formatting
ax.set_title("3D Stem Plot of 2D FFT ($X_{kl}$)")
ax.set_xlabel('Frequency $f_x$')
ax.set_ylabel('Frequency $f_y$')
ax.set_zlabel('Magnitude')
ax.set_zlim(0, 17)
ax.view_init(elev=25, azim=50)

plt.savefig('result.png', dpi=300)

