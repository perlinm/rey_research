#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from dicke_methods import *
from squeezing_methods import spin_squeezing


fig_dir = "../figures/spheres/"
params = { "text.usetex" : True }
plt.rcParams.update(params)
fig_dpi = 600
grid_size = 201

N = 50
max_tau = 2
time_steps = 1000
ivp_steps = 100


max_time = max_tau * N**(-2/3)
times = np.linspace(0, max_time, time_steps)

methods = [ "OAT", "TVF", "TAT" ]
OAT, TVF, TAT = methods

def save_state(state, file_name):
    plot_dicke_state(state, grid_size = grid_size)
    plt.gca().set_rasterization_zorder(1)
    plt.savefig(fig_dir + file_name, rasterized = True, dpi = fig_dpi)


S_op_vec, SS_op_mat = spin_op_vec_mat_dicke(N)

H = {}
H[OAT] = SS_op_mat[0][0]
H[TVF] = SS_op_mat[0][0] + N/2 * S_op_vec[1]
H[TAT] = 1/3 * ( SS_op_mat[0][0] - SS_op_mat[2][2] )

def time_deriv(state, H):
    return -1j * H.dot(state)
def ivp_deriv(H):
    return lambda time, state : time_deriv(state, H)


init_state = coherent_spin_state([0,1,0], N)
save_state(init_state, f"sphere_init.pdf")

for method in methods:
    print(method)
    states = solve_ivp(ivp_deriv(H[method]), (0,times[-1]), init_state,
                       t_eval = times, max_step = times[1]/ivp_steps).y
    sqz = np.array([ spin_squeezing(N, states[:,tt], S_op_vec, SS_op_mat)
                     for tt in range(times.size) ])
    opt_idx = sqz.argmax()
    save_state(states[:,opt_idx], f"sphere_{method}_opt.pdf")
    save_state(states[:,opt_idx//2], f"sphere_{method}_half.pdf")
