import sys, os, math
dir = os.getcwd()

import matplotlib.pyplot as plt
import numpy as np, pandas

scale_dict = {'0318': np.sqrt(55167/2869), '0318v1':np.sqrt(55167/2869),'1203v2':np.sqrt(60321/7169), '1203v3':np.sqrt(60321/7169)}
# DRL: 64 agents, ICM
eplen_py = {'0318': 60.3, '0318v1':52.1, '1203v2':93.8, '1203v3':112.0}
eplen_std_py = {'0318':0.553, '0318v1':1.51, '1203v2': 3.15, '1203v3':8.57}

# Heuristics: 64 trials
eplen_ml = {'0318': 4374.9, '1203v2':3528.2}
eplen_std_ml = {'0318':407.5, '1203v2':370.1}

# Divid-N-conquer: 64 trials
eplen_ml_dc = {'0318':2813.0, '1203v2':2851.9}
eplen_std_ml_dc = {'0318':216.1, '1203v2':280.7}
# map0318data = [[3709.3, 2520.0],[1658.6, 337.1],[674.2, 23.5], [328.9, 18.7]]
# map0122data = [[24000, 10000], [11598.2, 1590.5], [5861.4, 652.5], [1405.0, 188.5]]

# map0318data = [[1658.6, 337.1],[674.2, 23.5], [328.9, 18.7]]
# map0122data = [[11598.2, 1590.5], [5861.4, 652.5], [1405.0, 188.5]]



fig, axs = plt.subplots(1, 1, figsize=(8, 8))
x = np.arange(2)
xticks = ['maze1', 'maze2']
colors = ['black', 'magenta','blue']
# Set the ticks and ticklabels for all axes
plt.setp(axs, xticks=x, xticklabels=xticks)

for id, map in enumerate(['0318', '1203v2']):
    if id < 1:
        label = ['Benchmark', 'DNC', 'ICM']
    else:
        label = [None, None, None]
    axs.errorbar(x[id], eplen_ml[map], eplen_std_ml[map], fmt='o', linestyle="None",
                    color='black', markersize='8', capsize=10, elinewidth=3, label=label[0])
    axs.errorbar(x[id], eplen_ml_dc[map], eplen_std_ml_dc[map], fmt='o', linestyle="None",
                    color='blue', markersize='8', capsize=10, elinewidth=3, label=label[1])
    axs.errorbar(x[id], 4 * scale_dict[map] * eplen_py[map], 4 * scale_dict[map] * eplen_std_py[map], fmt='o', linestyle="None",
                    color='red', markersize='8', capsize=10, elinewidth=3, label=label[2])
axs.set_ylim(bottom=0.0)
axs.set_xlim(left=-1.5, right=2.5)
axs.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
axs.set_ylabel('Num Steps', fontsize = 16)
axs.tick_params(labelsize=16)
axs.yaxis.offsetText.set_fontsize(14)
axs.legend(loc="upper right", prop={'size': 16})
fig.tight_layout()
plt.show()
