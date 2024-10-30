import os 
import numpy as np
from read_pfm import *
import matplotlib.pyplot as plt

root_path = r'F:\11.09_later\LFsynfromSingleView\est_disps\ex_div'
files = sorted([i for i in os.listdir(root_path) if i.endswith('.pfm')])

# est padded_inria range
#dispmin = [-3.9776373, -2.6604526, -0.5838925, -1.8149893, -0.902449]
#dispmax = [1.2473879, 0.78691596,  0.54742336, 3.7186515, 0.44722313]

# est stfgantry range
#ispmin = [-3.891059, -2.0988493]
#dispmax = [2.916655, 2.1692533]

# est hciold range
dispmin = [-1] * 5
dispmax = [1] * 5

i = 0
for name in files:
    unscaled = read_pfm(os.path.join(root_path, name))
    zero_one = (unscaled - unscaled.min()) / (unscaled.max() - unscaled.min())
    scaled = zero_one * (dispmax[i] - dispmin[i]) + dispmin[i]
    print(scaled.min(), scaled.max())
    i += 1


    np.save(os.path.join(r'F:\11.09_later\LFsynfromSingleView\est_disps\rescaled_ex_div', name[:-4] + '.npy'), scaled)
