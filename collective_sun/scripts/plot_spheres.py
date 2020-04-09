#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from dicke_methods import spin_op_vec_mat_dicke, coherent_spin_state, plot_dicke_state
from squeezing_methods import spin_squeezing

fig_dir = "../figures/"
params = { "text.usetex" : True }
plt.rcParams.update(params)
fig_dpi = 301
grid_size = 201

N = 50
max_tau = 2
time_steps = 1000
ivp_tolerance = 1e-5


max_time = max_tau * N**(-2/3)
times = np.linspace(0, max_time, time_steps)

methods = [ "OAT", "TAT", "TNT" ]
OAT, TAT, TNT = methods

def save_state(state, file_name):
    plot_dicke_state(state, grid_size = grid_size, figsize = (2,2), white_sphere = True)
    plt.gca().set_rasterized(True)
    plt.gca().set_rasterization_zorder(1)
    plt.savefig(fig_dir + file_name, rasterized = True, dpi = fig_dpi)
    plt.close()

S_op_vec, SS_op_mat = spin_op_vec_mat_dicke(N)
S_z = S_op_vec[0]
H_OAT = S_z @ S_z

##################################################
# simulate and plot

print("plotting initial state")
init_state = coherent_spin_state([0,1,0], N)
save_state(init_state, f"sphere_init.pdf")

def time_deriv(state, H):
    return -1j * H.dot(state)
def ivp_deriv(H):
    return lambda time, state : time_deriv(state, H)

print("simulating")
states = solve_ivp(ivp_deriv(H_OAT), (0,times[-1]), init_state,
                   t_eval = times, rtol = ivp_tolerance, atol = ivp_tolerance).y
sqz = np.array([ spin_squeezing(N, states[:,tt], S_op_vec, SS_op_mat)
                 for tt in range(times.size) ])
opt_idx = sqz.argmin()

print("plotting squeezed states")
save_state(states[:,opt_idx//3], f"sphere_OAT_start.pdf")
save_state(states[:,opt_idx], f"sphere_OAT_opt.pdf")

print("completed")
