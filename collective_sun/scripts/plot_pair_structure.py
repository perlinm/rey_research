#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from multibody_methods import dist_method, get_multibody_states

spin_num = 16
alpha = 3
manifold = 2
periodic = True

font_size = 8
figsize = (1.5,1.5)
color_map = "RdBu"

fig_dir = "../figures/pair_structure/"

params = { "font.family" : "serif",
           "font.serif" : "Computer Modern",
           "text.usetex" : True,
           "font.size" : font_size }
plt.rcParams.update(params)

plot_args = dict( cmap = color_map, aspect = "equal", interpolation = "nearest" )

##########################################################################################
# build SU(n) interaction matrix, and build generators of interaction eigenstates
##########################################################################################

lattice_shape = (spin_num,)
dist = dist_method(lattice_shape, periodic = periodic)

sunc = {}
spin_num = np.prod(lattice_shape)
sunc["mat"] = np.zeros((spin_num,spin_num))
for pp in range(spin_num):
    for qq in range(pp+1,spin_num):
        sunc["mat"][pp,qq] = sunc["mat"][qq,pp] = -1/dist(pp,qq)**alpha
sunc["TI"] = periodic

sunc["shells"], sunc["energies"], sunc_tensors \
    = get_multibody_states(lattice_shape, sunc["mat"], manifold, sunc["TI"])
sunc.update(sunc_tensors)

shells = sunc["shells"][manifold]
max_mag = max( max(abs(sunc[shell].flatten())) for shell in shells )
norm = mpl.colors.Normalize(vmin = -max_mag, vmax = max_mag)

for shell in shells:
    plt.figure(figsize = figsize)
    energy = sunc["energies"][shell]
    plt.title(r"$\Delta\approx" + f"{energy:.2f}" + "$")
    plt.imshow(sunc[shell], norm = norm, **plot_args)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout(pad = 0.1)
    plt.savefig(fig_dir + f"pair_L{spin_num}_a{alpha}_s{shell}.pdf")
    plt.close()
