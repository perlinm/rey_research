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

dims = [ 10, 40, 20, 80 ]
colors = [ "k", "tab:orange", "tab:blue", "tab:green" ]

##################################################

plt.figure(figsize = figsize)

for dim, color in zip(dims, colors):
    files = glob.glob(data_dir + f"axes_d{dim}.txt")
    assert(len(files) == 1)
    axes, _, scales = np.loadtxt(files[0], unpack = True)

    scales *= np.sqrt(axes)
    excess_axes = axes - (2*dim-1)
    plt.plot(excess_axes/dim, scales/scales[0],
             ".", label = f"$d={dim}$", color = color)

plt.gca().set_ylim(bottom = 0)
plt.gca().tick_params(right = True)

plt.xlabel(r"$p/d$")
plt.ylabel(r"$\tilde\beta(p)/\tilde\beta(0)$")
spacing_kwargs = dict( handlelength = 1, columnspacing = 1, labelspacing = 0.2 )
plt.legend(loc = "best", ncol = 2, **spacing_kwargs)
plt.tight_layout()
plt.savefig(fig_dir + "qudit_axes.pdf")
