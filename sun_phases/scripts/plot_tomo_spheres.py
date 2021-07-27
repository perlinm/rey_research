#!/usr/bin/env python3

import os, sys
import numpy as np
import scipy, scipy.stats
import matplotlib.pyplot as plt

from dicke_methods import plot_dicke_state
from multilevel_methods import drive_op

np.set_printoptions(linewidth = 200)

fig_dpi = 1000
grid_size = 200
fig_dir = "../figures/qudit_tomo/"

##################################################
# plot a random state

dim = 50
state_seed = 857054
np.random.seed(state_seed)

unitary = scipy.stats.unitary_group.rvs(dim)
state = np.zeros(dim)
state[-1] = 1
state = unitary @ state

plot_dicke_state(state, grid_size = grid_size)
plt.savefig(fig_dir + "rnd_state.pdf", dpi = fig_dpi)
plt.close("all")
