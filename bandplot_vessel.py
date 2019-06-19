import sys, os, math
dir = os.getcwd()

import matplotlib.pyplot as plt
import numpy as np, pandas


def big_fcn(filename, subdir='vessel'):
    def moving_avg(arr, window=10):
        pad_arr = np.append(arr[0]*np.ones(int(window/2)), arr)
        pad_arr = np.append(pad_arr, arr[-1]*np.ones(int(window/2)))
        pad_arr2 = np.empty((np.shape(arr)[0], window))

        for i in np.arange(np.shape(arr)[0]):
            pad_arr2[i, :] = pad_arr[i:i+window]

        mean = np.mean(pad_arr2, axis=1)
        std = np.std(pad_arr2, axis=1)
        return mean, std



    # Load data from log
    rawdata3 = pandas.read_csv(os.path.join(dir,'logger',subdir, filename), header=0, engine='python')
    variables3 = ['n_updates', 'total_secs', 'tcount', \
                  'epcount','eprew','best_ext_ret']
    var_dict3 = {}
    for i in np.arange(len(variables3)):
        var_dict3[variables3[i]] = i
    df3 = rawdata3[variables3]
    num_features = len(variables3)
    data_size = len(df3[variables3[0]].values)
    dataset3 = np.empty((data_size, num_features))

    for idx, feature in enumerate(variables3):
        dataset3[:, idx] = np.array(df3[feature].values, dtype=np.float32)

    # Plot stats vs episodes
    for x_label in ['epcount', 'tcount']:
        y_label = 'eprew'
        y2_label = 'best_ext_ret'
        f, axs = plt.subplots(1, 1, figsize=(8, 6))

        mean, std = moving_avg(dataset3[:,var_dict3[y_label]], window=4)
        axs.plot(dataset3[:,var_dict3[x_label]],  mean, linewidth=3 ,color='orange',label='mean')
        axs.fill_between(dataset3[:,var_dict3[x_label]],  mean-std, mean+std, color='moccasin')
        axs.plot(dataset3[:,var_dict3[x_label]],  dataset3[:,var_dict3[y2_label]], '--', linewidth=2 ,color='orange',label='best')

        """Post processing"""
        #============================================================

        ylabel = 'Episode Reward'
        # axs.set_title('Training Stats', fontsize=16)
        axs.set_xlabel(x_label, fontsize=24)
        axs.set_ylabel(ylabel, fontsize=24)
        axs.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        axs.xaxis.offsetText.set_fontsize(24)

        axs.tick_params(labelsize=24)
        axs.legend(loc="lower right",prop={'size': 18})

        f.tight_layout()
        f.subplots_adjust(
            top=0.969,
            bottom=0.162,
            left=0.167,
            right=0.977,
            hspace=0.2,
            wspace=0.233
        )
        # plt.show()
        plt.savefig(os.path.join(figpath, filename[:-4]+'-%s.png'%(x_label) ))


#==========================================================
""""Configurations"""
logs = os.listdir(os.path.join(dir, 'logger','vessel'))

for env_id,  log in enumerate(logs):
    if '.csv' in log:
        print(log)
        os.makedirs(os.path.join(dir, 'fig'), exist_ok=True)
        figpath = os.path.join(dir, 'fig')
        big_fcn(filename=log)
        #-----------------------------------------------------------