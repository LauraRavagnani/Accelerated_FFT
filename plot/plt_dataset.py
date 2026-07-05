import matplotlib.pyplot as plt
import matplotlib.colors as col
from mpl_toolkits.mplot3d import Axes3D
import json
import numpy as np
import pandas as pd

with open('../Gxy_vs_xy.json', 'r') as f:
    data = json.load(f)
    
    x = [item['x'] for item in data]
    y = [item['y'] for item in data]
    Gxy = [item['z'] for item in data]

    colors = ["#4a6fa5", 'white', "#db7a57"]
    cmap = col.LinearSegmentedColormap.from_list('custom', colors, N=256)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(np.array(x).reshape((int(np.sqrt(len(x))), int(np.sqrt(len(x))))),
                            np.array(y).reshape((int(np.sqrt(len(y))), int(np.sqrt(len(y))))),
                            np.array(Gxy).reshape((int(np.sqrt(len(Gxy))), int(np.sqrt(len(Gxy))))),
                            cmap=cmap, antialiased=True, shade=True)

    ax.set_title('$G(x, y) = \\cos(2\\pi f_x x) \\cos(3\\pi f_y y)$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('G(x, y)')
    fig.colorbar(surf, shrink=0.5, aspect=10)
    ax.set_facecolor('white')

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor('lightgray')

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo['grid'].update({
            'color': 'lightgray',
            'linestyle': '--',
            'linewidth': 0.8
        })
    plt.savefig('dataset.png', dpi=600)

    