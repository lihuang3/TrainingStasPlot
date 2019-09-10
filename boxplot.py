import sys, os, math
dir = os.getcwd()

import matplotlib.pyplot as plt
import numpy as np, pandas

scale_dict = {'0318': np.sqrt(55167/2869), '0318v1':np.sqrt(55167/2869),'1203v2':np.sqrt(60321/7169), \
    '1203v3':np.sqrt(60321/7169), '0122v1':np.sqrt(35055/22593)}
# DRL: 64 agents, ICM
eplen_py = {'0318': 60.3, '0318v1':52.1, '1203v2':93.8, '1203v3':112.0, '0122v1':351.25}
eplen_std_py = {'0318':0.553, '0318v1':1.51, '1203v2': 3.15, '1203v3':8.57, '0122v1':47.13}

# Heuristics: 64 trials
eplen_ml = {'0318': 4374.9, '1203v2':3528.2, '0122v1':11598.2}
eplen_std_ml = {'0318':407.5, '1203v2':370.1,'0122v1': 1590.5}

# Divid-N-conquer: 64 trials
eplen_ml_dc = {'0318':2813.0, '1203v2':2851.9, '0122v1':5861.4}
eplen_std_ml_dc = {'0318':216.1, '1203v2':280.7, '0122v1':652.5}
# map0318data = [[3709.3, 2520.0],[1658.6, 337.1],[674.2, 23.5], [328.9, 18.7]]
# map0122data = [[24000, 10000], [11598.2, 1590.5], [5861.4, 652.5], [1405.0, 188.5]]

# map0318data = [[1658.6, 337.1],[674.2, 23.5], [328.9, 18.7]]
# map0122data = [[11598.2, 1590.5], [5861.4, 652.5], [1405.0, 188.5]]



fig, axs = plt.subplots(1, 1, figsize=(10, 8))
x = [0.5, 1.0, 1.5, 2.5, 3.0,3.5, 4.5, 5, 5.5]
xticks = ['maze1', 'maze2']
colors = ['black', 'magenta','blue']
# Set the ticks and ticklabels for all axes
plt.setp(axs, xticks=x, xticklabels=xticks)

for id, map in enumerate(['0318', '1203v2','0122v1']):
    if id < 1:
        label = ['Benchmark', 'D&C', 'RL']
    else:
        label = [None, None, None]
    axs.errorbar(x[3*id], eplen_ml[map], eplen_std_ml[map], fmt='o', linestyle="None",
                    color='black', markersize='8', capsize=10, elinewidth=3, label=label[0])
    axs.errorbar(x[3*id+1], eplen_ml_dc[map], eplen_std_ml_dc[map], fmt='o', linestyle="None",
                    color='blue', markersize='8', capsize=10, elinewidth=3, label=label[1])
    axs.errorbar(x[3*id+2], 4 * scale_dict[map] * eplen_py[map], 4 * scale_dict[map] * eplen_std_py[map], fmt='o', linestyle="None",
                    color='red', markersize='8', capsize=10, elinewidth=3, label=label[2])

    plt.plot([x[3*id], x[3*id+1], x[3*id+2]], [eplen_ml[map], eplen_ml_dc[map],4 * scale_dict[map] * eplen_py[map]], 'k.-')

axs.set_ylim(bottom=-1000)
axs.set_xlim(left=-.5, right=7)
axs.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
axs.set_ylabel('Num Steps', fontsize = 24)
axs.tick_params(labelsize=24)
axs.yaxis.offsetText.set_fontsize(24)
axs.legend(loc="upper left", prop={'size': 24})
fig.tight_layout()
plt.show()
