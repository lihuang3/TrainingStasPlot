import sys, os, math
dir = os.getcwd()

import matplotlib.pyplot as plt
import numpy as np, pandas

# map0318data = [[3709.3, 2520.0],[1658.6, 337.1],[674.2, 23.5], [328.9, 18.7]]
# map0122data = [[24000, 10000], [11598.2, 1590.5], [5861.4, 652.5], [1405.0, 188.5]]
map0318data = [[1658.6, 337.1],[674.2, 23.5], [328.9, 18.7]]
map0122data = [[11598.2, 1590.5], [5861.4, 652.5], [1405.0, 188.5]]

map0122data = np.array(map0122data, dtype=np.float32)
map0318data = np.array(map0318data, dtype=np.float32)

fig, axs = plt.subplots(1, 2, figsize=(16, 8))
x = np.arange(3)
# xticks = ['Heuristics\nRRT','Heuristics\nowRRT', 'DNQ\nowRRT', 'DRL']
xticks = ['Heuristics\nowRRT', 'DNQ\nowRRT', 'DRL']
# Set the ticks and ticklabels for all axes
plt.setp(axs, xticks=x, xticklabels=xticks)

axs[0].errorbar(x, map0318data[:,0], map0318data[:,1], fmt='o', linestyle="None",
                color='black', markersize='10', capsize=6, elinewidth=3)
axs[1].errorbar(x, map0122data[:,0], map0122data[:,1], fmt='o', linestyle="None",
                color='black', markersize='10', capsize=6, elinewidth=3)
axs[0].set_ylim(bottom=0.0)
axs[1].set_ylim(bottom=0.0)

for i in range(2):
  axs[i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
  axs[i].set_ylabel('num_steps', fontsize = 16)
  axs[i].tick_params(labelsize=16)

fig.tight_layout()
plt.show()
