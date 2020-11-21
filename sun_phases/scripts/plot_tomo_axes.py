#!/usr/bin/env python3

import os, glob
import numpy as np
import matplotlib.pyplot as plt

data_dir = "../data/qudit_tomo/"
fig_dir = "../figures/qudit_tomo/"

if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

params = { "font.size" : 10,
           "text.usetex" : True }
plt.rcParams.update(params)

figsize = (3,2)

dims = [ 10, 20, 40, 80 ]
colors = [ "k", "tab:blue", "tab:orange", "tab:green" ]

##################################################

plt.figure(figsize = figsize)

for dim, color in zip(dims, colors):
    files = glob.glob(data_dir + f"axes_d{dim}.txt")
    assert(len(files) == 1)
    axes, _, scales = np.loadtxt(files[0], unpack = True)

    scales *= np.sqrt(axes)
    excess_axes = axes - (2*dim-1)
    normed_scales = scales / scales[0]
    plt.plot(excess_axes/dim, normed_scales, ".", label = dim, color = color)

plt.gca().set_ylim(bottom = 0)
plt.xlabel(r"$p/d$")
plt.ylabel(r"$\beta(r)$")
plt.legend(loc = "best", ncol = 2)
plt.tight_layout()
plt.savefig(fig_dir + "qudit_axes.pdf")
