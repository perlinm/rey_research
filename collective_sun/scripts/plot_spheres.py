#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from dicke_methods import spin_op_vec_mat_dicke, coherent_spin_state, plot_dicke_state
from squeezing_methods import spin_squeezing

params = { "font.size" : 10,
           "text.usetex" : True }
plt.rcParams.update(params)

figsize = (1,1)
fig_dir = "../figures/"
fig_dpi = 600
grid_size = 1000

N = 40
max_tau = 2
time_steps = 1000
ivp_tolerance = 1e-5

max_time = max_tau * N**(-2/3)
times = np.linspace(0, max_time, time_steps)

S_op_vec, SS_op_mat = spin_op_vec_mat_dicke(N)
S_z = S_op_vec[0]
H_OAT = S_z @ S_z

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

##################################################
# simulate and plot

print("plotting initial state")
init_state = coherent_spin_state([0,1,0], N)
plot_dicke_state(init_state, grid_size = grid_size, figsize = figsize)
arrow = Arrow3D([0,0], [-1.05]*2, [0.6,1.05], color = "k",
                arrowstyle = "-|>", mutation_scale = 5)
plt.gca().add_artist(arrow)
plt.gca().text(0,-0.95, 0.75, "$z$")
plt.savefig(fig_dir + "sphere_init.pdf", dpi = fig_dpi)

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

plot_dicke_state(states[:,opt_idx//3], grid_size = grid_size, figsize = figsize)
for sign in [ +1, -1 ]:
    for height, length in [ (.1,.3), (.3,.6) ]:
        arrow = Arrow3D([1.2]*2, [0,sign*length], [sign*height]*2, color = "#1f77b4",
                        arrowstyle = "-|>", mutation_scale = 5, zorder = 3)
        plt.gca().add_artist(arrow)
plt.savefig(fig_dir + "sphere_OAT_start.pdf", dpi = fig_dpi)

plot_dicke_state(states[:,opt_idx], grid_size = grid_size, figsize = figsize)
plt.savefig(fig_dir + "sphere_OAT_opt.pdf", dpi = fig_dpi)

print("completed")
