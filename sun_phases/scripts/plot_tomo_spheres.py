#!/usr/bin/env python3

import os, sys
import numpy as np
import scipy, scipy.stats
import matplotlib.pyplot as plt

from dicke_methods import plot_dicke_state, coherent_spin_state

np.set_printoptions(linewidth = 200)

fig_dpi = 1000
fig_dir = "../figures/qudit_tomo/"

##################################################

dim = 10
kwargs = dict( single_sphere = True, shade = False, grid_size = 501 )

# plot a random
np.random.seed(7)
random_U = scipy.stats.unitary_group.rvs(dim)
qudit_state = random_U @ coherent_spin_state("+X", dim-1)
figure, axes = plot_dicke_state(qudit_state, view_angles = (0,0), **kwargs)

# plot points
np.random.seed(1)
point_num = 2*dim-1
polar = np.arccos(2 * np.random.rand(point_num) - 1)
azimuth = 2*np.pi*np.random.rand(point_num)
points = np.array([ np.sin(polar) * np.cos(azimuth),
                    np.sin(polar) * np.sin(azimuth),
                    np.cos(polar) ])
in_front = points[0,:] >= 0
axes[0].plot(*points[:,in_front], "o", markersize = 15, zorder = 5)

plt.savefig(fig_dir + "sphere_points.pdf", dpi = fig_dpi)
