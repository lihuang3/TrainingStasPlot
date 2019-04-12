import sys, os, math
dir = os.getcwd()

import matplotlib.pyplot as plt
import numpy as np, pandas

def moving_avg(arr, window=32):
  pad_arr = np.append(arr[0]*np.ones(int(window/2)), arr)
  pad_arr = np.append(pad_arr, arr[-1]*np.ones(int(window/2)))
  pad_arr2 = np.empty((np.shape(arr)[0], window))

  for i in np.arange(np.shape(arr)[0]):
    pad_arr2[i, :] = pad_arr[i:i+window]

  mean = np.mean(pad_arr2, axis=1)
  std = np.std(pad_arr2, axis=1)
  return mean, std


#==========================================================
""""Configurations"""
f, axs = plt.subplots(1, 2, figsize=(20, 8))
env ='Maze1203-v2'
os.makedirs(os.path.join(dir, 'fig', env), exist_ok=True)
scope = 'eplen'
include_best = True
tag=  env + '_%s.png'%(scope)
figpath = os.path.join(dir, 'fig', env, tag)
#-----------------------------------------------------------

""" 
Part I: PPO
"""
filename = env + '-PPO.csv'
algo = 'PPO'
# Load data from log
rawdata = pandas.read_csv(os.path.join(dir,'logger', filename), header=0, engine='python')
variables = ['nupdates', 'time_elapsed', 'total_timesteps', 'serial_timesteps',\
             'eprewmean', 'epbestlen', \
             'eplenmean','curbestrew', 'epbestrew']
var_dict = {}
for i in np.arange(len(variables)):
  var_dict[variables[i]] = i
df = rawdata[variables]
num_features = len(variables)
data_size = len(df[variables[0]].values)
dataset = np.empty((data_size, num_features))
for idx, feature in enumerate(variables):
  dataset[:, idx] = np.array(df[feature].values, dtype=np.float32)
dataset[:, var_dict['time_elapsed']]/= 3600.0

# Plot stats vs time
x_label = 'time_elapsed'
if scope == 'eplen':
  y_label = 'eplenmean'
elif scope == 'eprew':
  y_label = 'eprewmean'
else:
  print('Scope %s not found!'%(scope))
  exit()
mean, std = moving_avg(dataset[:,var_dict[y_label]])
axs[0].plot(dataset[:,var_dict[x_label]],  mean, linewidth=4 ,color='navy',label=algo)
axs[0].fill_between(dataset[:,var_dict[x_label]],  mean-std, mean+std, color='lightcyan')

# Plot stats vs num updates
x_label = 'nupdates'
if scope == 'eplen':
  y_label = 'eplenmean'
elif scope == 'eprew':
  y_label = 'eprewmean'
else:
  print('Scope %s not found!'%(scope))
  exit()
mean, std = moving_avg(dataset[:,var_dict[y_label]])
axs[1].plot(dataset[:,var_dict[x_label]],  mean, linewidth=4 ,color='navy',label=algo)
axs[1].fill_between(dataset[:,var_dict[x_label]],  mean-std, mean+std, color='lightcyan')


"""
Part II: RND
"""
filename = env + '-RND.csv'
algo = 'RND'
# Load data from log
rawdata2 = pandas.read_csv(os.path.join(dir,'logger', filename), header=0, engine='python')
variables2 = ['n_updates', 'total_secs', 'tcount', \
             'eplen', 'best_eplen',\
             'eprew','recent_best_ext_ret', 'best_ext_ret', 'eprew_recent']
var_dict2 = {}
for i in np.arange(len(variables2)):
  var_dict2[variables2[i]] = i
df2 = rawdata2[variables2]
num_features = len(variables2)
data_size = len(df2[variables2[0]].values)
dataset2 = np.empty((data_size, num_features))
for idx, feature in enumerate(variables2):
  dataset2[:, idx] = np.array(df2[feature].values, dtype=np.float32)
dataset2[:, var_dict2['total_secs']]/= 3600.0
dataset2[:, var_dict2['eplen']]*= 4
dataset2[:, var_dict2['best_eplen']]*= 4
dataset2[:, var_dict2['recent_best_eplen']]*= 4

# Plot stats vs time
x_label = 'total_secs'
if scope == 'eplen':
  y_label = 'eplen'
elif scope == 'eprew':
  y_label = 'eprew'
else:
  print('Scope %s not found!'%(scope))
  exit()
mean, std = moving_avg(dataset2[:,var_dict2[y_label]], window=8)
axs[0].plot(dataset2[:,var_dict2[x_label]],  mean, linewidth=4 ,color='orange',label=algo)
axs[0].fill_between(dataset2[:,var_dict2[x_label]],  mean-std, mean+std, color='moccasin')

# Plot stats vs num updates
x_label = 'n_updates'
if scope == 'eplen':
  y_label = 'eplen'
elif scope == 'eprew':
  y_label = 'eprew'
else:
  print('Scope %s not found!'%(scope))
  exit()
mean, std = moving_avg(dataset2[:,var_dict2[y_label]], window=8)
axs[1].plot(dataset2[:,var_dict2[x_label]],  mean, linewidth=4 , color='orange',label=algo)
axs[1].fill_between(dataset2[:,var_dict2[x_label]],  mean-std, mean+std, color='moccasin')






"""
Part III: ICM
"""
filename = env + '-ICM.csv'
algo = 'ICM'
# Load data from log
rawdata3 = pandas.read_csv(os.path.join(dir,'logger', filename), header=0, engine='python')
variables3 = ['n_updates', 'total_secs', 'tcount', \
             'eplen', 'best_eplen', 'recent_best_eplen',\
             'eprew','recent_best_ext_ret', 'best_ext_ret', 'eprew_recent']
var_dict3 = {}
for i in np.arange(len(variables3)):
  var_dict3[variables3[i]] = i
df3 = rawdata3[variables3]
num_features = len(variables3)
data_size = len(df3[variables3[0]].values)
dataset3 = np.empty((data_size, num_features))
for idx, feature in enumerate(variables3):
  dataset3[:, idx] = np.array(df3[feature].values, dtype=np.float32)
dataset3[:, var_dict3['total_secs']]/= 3600.0
dataset3[:, var_dict3['eplen']]*= 4
dataset3[:, var_dict3['best_eplen']]*= 4
dataset3[:, var_dict3['recent_best_eplen']]*= 4

# Plot stats vs time
x_label = 'total_secs'
if scope == 'eplen':
  y_label = 'eplen'
elif scope == 'eprew':
  y_label = 'eprew'
else:
  print('Scope %s not found!'%(scope))
  exit()
mean, std = moving_avg(dataset3[:,var_dict3[y_label]], window=8)
axs[0].plot(dataset3[:,var_dict3[x_label]],  mean, linewidth=4 ,color='purple',label=algo)
axs[0].fill_between(dataset3[:,var_dict3[x_label]],  mean-std, mean+std, color='plum')

# Plot stats vs num updates
x_label = 'n_updates'
if scope == 'eplen':
  y_label = 'eplen'
elif scope == 'eprew':
  y_label = 'eprew'
else:
  print('Scope %s not found!'%(scope))
  exit()
mean, std = moving_avg(dataset3[:,var_dict3[y_label]], window=8)
axs[1].plot(dataset3[:,var_dict3[x_label]],  mean, linewidth=4 , color='purple',label=algo)
axs[1].fill_between(dataset3[:,var_dict3[x_label]],  mean-std, mean+std, color='plum')



"""Post processing"""
#============================================================
axs[0].set_title('Training Stats', fontsize=16)
axs[0].set_xlabel('Time (h)', fontsize=16)
axs[0].set_ylabel('Episode Length', fontsize=16)
axs[0].tick_params(labelsize=16)
axs[0].legend(loc="center right",prop={'size': 16})

axs[1].set_title('Training Stats', fontsize=16)
axs[1].set_xlabel('num updates', fontsize=16)
axs[1].set_ylabel('Episode Length', fontsize=16)
axs[1].tick_params(labelsize=16)
axs[1].legend(loc="center right",prop={'size': 16})

f.tight_layout()
f.subplots_adjust(
top=0.948,
bottom=0.096,
left=0.05,
right=0.991,
hspace=0.2,
wspace=0.117
)
# plt.show()
plt.savefig(figpath)


