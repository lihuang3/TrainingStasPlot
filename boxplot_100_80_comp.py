import sys, os, math
dir = os.getcwd()

import matplotlib.pyplot as plt
import numpy as np, pandas

# DRL: 64 agents, ICM
scale = 4
# eplen_idf = {'0519p100': 113.96, '0519p80':86.23, '0519p60': 65.21, '1203p100':93.06, '1203p80':78.45, '1203p60':64.69}
# eplen_std_idf = {'0519p100': 1.62, '0519p80':1.22, '0519p60': 4.29,'1203p100':2.43, '1203p80':2.24, '1203p60':5.37}
#
# eplen_idf.update({'0122p100':337.6, '0122p80': 233.16, '0122p60': 222.21})
# eplen_std_idf.update({'0122p100':24.5, '0122p80': 31.72, '0122p60': 39.72})

# eplen_rnd = {'0519p100': 125.10, '0519p80':95.51, '1203p100':96.76, '1203p80':80.54}
# eplen_std_rnd = {'0519p100': 2.03, '0519p80':8.68, '1203p100':3.04, '1203p80':3.76}


eplen_idf = {'0519p100': 113.96, '0519p80':116, '0519p60': 106, '1203p100':93.06, '1203p80':95.6, '1203p60':95.8}
eplen_std_idf = {'0519p100': 1.62, '0519p80':1.55, '0519p60': 2.45, '1203p100':2.43, '1203p80':3, '1203p60':4.24}

eplen_idf.update({'0122p100':337.6, '0122p80': 344, '0122p60': 349})
eplen_std_idf.update({'0122p100':24.5, '0122p80': 48.5, '0122p60': 76.4})





mazes = ['0519','1203', '0122']
fig, axs = plt.subplots(1, 1, figsize=(10, 8))
x = [0.5, 1.0, 1.5, 2.5, 3.0,3.5, 4.5, 5, 5.5]

colors = ['red', 'black','blue']
# Set the ticks and ticklabels for all axes
plt.setp(axs, xticks=x)

for id, _map in enumerate(mazes):
    if id < 1:
        label = ['100%', '80%', '60%']
    else:
        label = [None, None, None]
    print(_map)
    map = _map + 'p100'
    axs.errorbar(x[len(mazes)*id], scale*eplen_idf[map], scale*eplen_std_idf[map], fmt='o', linestyle="None",
                    color='black', markersize='8', capsize=10, elinewidth=3, label=label[0])
    map = _map + 'p80'
    axs.errorbar(x[len(mazes)*id+1], scale*eplen_idf[map], scale*eplen_std_idf[map], fmt='o', linestyle="None",
                 color='red', markersize='8', capsize=10, elinewidth=3, label=label[1])

    map = _map + 'p60'
    axs.errorbar(x[len(mazes)*id+2], scale*eplen_idf[map], scale*eplen_std_idf[map], fmt='o', linestyle="None",
                 color='blue', markersize='8', capsize=10, elinewidth=3, label=label[2])

    """
        Connect points with lines
    """
    plt.plot([x[len(mazes)*id], x[len(mazes)*id+1], x[len(mazes)*id+2]], [scale*eplen_idf[_map+'p100'], scale*eplen_idf[_map+'p80'], scale*eplen_idf[_map+'p60']], 'k.-')

    # map = _map + 'p100'
    # axs.errorbar(x[2*id+1], scale*eplen_rnd[map], scale*eplen_std_rnd[map], fmt='o', linestyle="None",
    #                 color='black', markersize='8', capsize=10, elinewidth=3)
    # map = _map + 'p80'
    # axs.errorbar(x[2*id+1], scale*eplen_rnd[map], scale*eplen_std_rnd[map], fmt='o', linestyle="None",
    #                 color='red', markersize='8', capsize=10, elinewidth=3)
axs.set_ylim(bottom=0.0, top=1750)
axs.set_xlim(left=-1, right=7)
# axs.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
axs.set_ylabel('Avg Episode Length', fontsize = 24)
axs.tick_params(labelsize=24)
axs.yaxis.offsetText.set_fontsize(24)
axs.legend(loc="upper left", prop={'size': 24})
fig.tight_layout()
plt.show()
