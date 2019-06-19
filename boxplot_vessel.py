import sys, os, math
dir = os.getcwd()

import matplotlib.pyplot as plt
import numpy as np, pandas as pd

df = pd.read_csv(os.path.join(dir, 'ves_stats.csv'))
mazes = np.unique(df['maze'].values)

x = np.arange(3)
xticks = ['Long gap', 'Short gap', 'Continuous']
labels = ['priority', 'default', 'RL', 'none']
markers = ['o', 'D', 'd', 's']
colors = ['b', 'cyan', 'r', 'k']
goals = [['v0', 'v2', 'v4'], ['v1', 'v3', 'v5']]

# Set the ticks and ticklabels for all axes

for id, maze in enumerate(mazes):
    for goal_idx, goal in enumerate(goals):
        fig, axs = plt.subplots(1, 1, figsize=(10, 8))
        plt.setp(axs, xticks=x, xticklabels=xticks)

        for case_idx, case in enumerate(goal):
            if case_idx<1:
                label = labels
            else:
                label = [None] * len(labels)
            cdf = df[np.logical_and(df['maze'].values == mazes[id], df['case'].values == case)]
            stats = np.array(cdf.values[0][2:], dtype=np.float32)
            for idx in range( np.shape(stats)[0]//2):
                axs.errorbar(case_idx, stats[2*idx], stats[2*idx+1], fmt=markers[idx], linestyle="None",
                         color=colors[idx], markersize='10', capsize=10, elinewidth=2, label=label[idx])

        axs.set_ylim(bottom=0.0)
        axs.set_xlim(left=-1, right=len(xticks)+0.5)
        # axs.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        axs.set_ylabel('Delivery rate (%)', fontsize = 24)
        axs.tick_params(labelsize=24)
        axs.yaxis.offsetText.set_fontsize(24)
        axs.legend(loc="upper right", prop={'size': 24})
        fig.tight_layout()
        fig.subplots_adjust(
            top = 0.977,
        bottom = 0.073,
        left = 0.106,
        right = 0.982,
        hspace = 0.2,
        wspace = 0.2
        )
        plt.savefig(os.path.join(dir, 'stats_%s_g%d.png'%(maze, goal_idx)), pad_inches=0.0, dpi=100)
        # plt.show()
