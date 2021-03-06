#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from multibody_methods import dist_method, get_multibody_states

spin_num = 21
alpha = 2
columns = 4
shell_cutoff = 20

font_size = 9
color_map = "RdBu"

panel_figsize = (6,3.3)
single_figsize = (1.5,1.5)

fig_dir = "../figures/structure/"

params = { "font.family" : "serif",
           "font.serif" : "Computer Modern",
           "text.usetex" : True,
           "font.size" : font_size,
           "axes.titlesize" : font_size }
plt.rcParams.update(params)

norm = mpl.colors.Normalize(vmin = -1, vmax = 1)
plot_args = dict( cmap = color_map, norm = norm,
                  origin = "lower", interpolation = "none", aspect = "equal" )

##########################################################################################
# build SU(n) interaction matrix, and build generators of interaction eigenstates
##########################################################################################

max_manifold = 3
periodic = True
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
    = get_multibody_states(lattice_shape, sunc["mat"],
                           max_manifold, sunc["TI"], shell_cutoff = shell_cutoff)
sunc.update(sunc_tensors)

shell_num = len(sunc["energies"])
shell_digits = len(str(shell_num))

##########################################################################################
# plot generators within the 2- and 3-excitation manifolds
##########################################################################################

def get_matrix(tensor):
    assert( tensor.ndim in [ 2, 3] )
    if tensor.ndim == 2:
        mat = tensor
        sign = np.sign(mat[0,1])
    elif tensor.ndim == 3:
        mat = tensor[:,:,0]
        sign = np.sign(mat[1,2])
    mat *= sign / abs(max(mat.flatten()))
    mat = np.roll(mat, spin_num//2, axis = 0)
    mat = np.roll(mat, spin_num//2, axis = 1)
    return mat

def plot_matrix(axis, matrix, energy):
    title = r"$\Delta\approx" + f"{energy:.2f}" + "$"
    axis.set_title(title)
    axis.imshow(matrix, **plot_args)
    axis.set_xticks([])
    axis.set_yticks([])

# make figure showing several shells
figure, axes = plt.subplots(2, columns, figsize = panel_figsize)

for manifold, row_axes in zip([ 2, 3 ], axes):

    # make figure showing each individual shell
    fig, ax = plt.subplots(1, figsize = single_figsize)

    shells = sunc["shells"][manifold]
    for idx, shell in enumerate(shells):
        if idx > shell_cutoff: break
        energy = sunc["energies"][shell]
        matrix = get_matrix(sunc[shell])
        plot_matrix(ax, matrix, energy)

        if idx < columns:
            plot_matrix(row_axes[idx], matrix, energy)

        shell_str = str(shell).rjust(shell_digits, "0")
        fig.tight_layout(pad = 0.1)
        fig.savefig(fig_dir + f"mat_L{spin_num}_a{alpha}_s{shell_str}.pdf")
        plt.close(fig)

    row_axes[0].set_ylabel(f"$r={manifold}$")

figure.tight_layout(pad = 0.2, h_pad = 1, w_pad = 1)
figure.savefig(fig_dir + f"mat_L{spin_num}_a{alpha}.pdf")
