import numpy as np, os
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize_3d as skel_3d
import random, skimage
ROOT_PATH = os.path.abspath('./')

mapfile = 'map1203'
filename = 'map1203'

raw_img = plt.imread(os.path.join(ROOT_PATH, mapfile+'.png'))
mazeData  =np.loadtxt(os.path.join(ROOT_PATH, mapfile+'.csv'))
h, w = np.shape(raw_img)[:2]
h1, w1 = mazeData.shape

import matplotlib as mpl
## map0318
# goal1 = [82, 82]
# goal2 = [42, 48]

## map1203
goal1 = [130, 61]
goal2 = [75, 100]
plt.imshow(raw_img)
fig = plt.gcf()
fig.set_size_inches(7.5, 6.25)


circle = plt.Circle((w/w1*goal1[1], h/h1*goal1[0]), 25, linestyle='-', color='red',
                    linewidth=2, fill=False)

plt.gcf().gca().add_artist(circle)

circle = plt.Circle((w/w1*goal2[1], h/h1*goal2[0]), 25, linestyle='-', color='blue',
                    linewidth=2, fill=False)

plt.gcf().gca().add_artist(circle)

plt.axis('off')
# plt.show()
fig.tight_layout()
fig.subplots_adjust \
    (top=0.973,
bottom=0.057,
left=0.068,
right=0.98,
hspace=0.2,
wspace=0.2)
plt.savefig( os.path.join(ROOT_PATH, filename+'_goal.png'), pad_inches=0.0, dpi=100)

##++++++++++++++++++++++++++++++++++++++++++++




