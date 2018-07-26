#!/usr/bin/env python3

import sys, itertools
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar

from mathieu_methods import mathieu_solution
from overlap_methods import tunneling_1D, pair_overlap_1D
from sr87_olc_constants import g_int_LU, recoil_energy_NU, recoil_energy_Hz

from dicke_methods import spin_op_vec_mat_dicke, coherent_spin_state, squeezing_OAT
from squeezing_methods import spin_vec_mat_vals, spin_squeezing, evolve
from fermi_hubbard_methods import spatial_basis

show = "show" in sys.argv
save = "save" in sys.argv
assert(show or save)

figsize = (4,3)
fig_dir = "../figures/"
params = { "text.usetex" : True }
plt.rcParams.update(params)

min_tau = 0
max_tau = 1.6
time_steps = 200

N_min = 5
N_max = 1e4
N_vals = 100

# value of N at which we switch methods for determining optimal TAT parameters
N_crossover = 1500

tau_vals = np.linspace(min_tau, max_tau, time_steps)
particle_nums = np.logspace(np.log10(N_min), np.log10(N_max), N_vals)
particle_nums = np.unique(particle_nums.round().astype(int))

def to_dB(x): return 10*np.log10(x)

squeezing_TAT_vals = np.zeros(time_steps)
squeezing_opt_OAT_vals = np.zeros(len(particle_nums))
squeezing_opt_TAT_vals = np.zeros(len(particle_nums))
tau_opt_OAT_vals = np.zeros(len(particle_nums))
tau_opt_TAT_vals = np.zeros(len(particle_nums))
for nn in range(len(particle_nums)):
    N = particle_nums[nn]
    print("N:",N)

    # determine optimal OAT squeezing parameters
    def squeezing_OAT_val(tau):
        chi_t = tau * N**(-2/3)
        return to_dB(squeezing_OAT(chi_t, N))
    optimum_OAT = minimize_scalar(squeezing_OAT_val, method = "bounded",
                                  bounds = (min_tau, max_tau))
    tau_opt_OAT_vals[nn] = optimum_OAT.x
    squeezing_opt_OAT_vals[nn] = -optimum_OAT.fun

    # construct TAT Hamiltonian and spin operator vector / matrix
    S_op_vec, SS_op_mat = spin_op_vec_mat_dicke(N)
    H_TAT = 1/3 * np.real( SS_op_mat[0][0] - SS_op_mat[2][2] )
    state = coherent_spin_state([0,1,0], N)

    if N < N_crossover:
        vals, vecs = linalg.eigh(H_TAT.toarray())
        state = vecs.T @ state
        S_op_vec = np.array([ vecs.T @ X @ vecs for X in S_op_vec ])
        SS_op_mat = np.array([ [ vecs.T @ X @ vecs for X in XS ] for XS in SS_op_mat ])

        def squeezing_TAT_val(tau):
            chi_t = tau * N**(-2/3)
            state_t =  np.exp(-1j * chi_t * vals) * state
            return to_dB(spin_squeezing(state_t, S_op_vec, SS_op_mat, N)[0])
        optimum_TAT = minimize_scalar(squeezing_TAT_val, method = "bounded",
                                      bounds = (min_tau, max_tau))
        tau_opt_TAT_vals[nn] = optimum_TAT.x
        squeezing_opt_TAT_vals[nn] = -optimum_TAT.fun

    else:
        chi_times = tau_vals * N**(-2/3)
        d_chi_t = ( chi_times[-1] - chi_times[0] ) / time_steps
        for ii in range(time_steps):
            squeezing_TAT_vals[ii], _ = spin_squeezing(state, S_op_vec, SS_op_mat, N)
            state = evolve(state, H_TAT, d_chi_t)
        squeezing_opt_TAT_vals[nn] = -to_dB(squeezing_TAT_vals.min())
        tau_opt_TAT_vals[nn] = tau_vals[squeezing_TAT_vals.argmin()]

plt.figure(figsize = figsize)
plt.semilogx(particle_nums, squeezing_opt_OAT_vals, label = "OAT")
plt.semilogx(particle_nums, squeezing_opt_TAT_vals, label = "TAT")
plt.xlim(N_min, N_max)
plt.xlabel(r"$N$")
plt.ylabel(r"$-10\log_{10}(\xi_{\mathrm{opt}}^2)$")
plt.legend(loc = "best")
plt.tight_layout()
if save: plt.savefig(fig_dir + "optimal_squeezing.pdf")

plt.figure(figsize = figsize)
plt.semilogx(particle_nums, tau_opt_OAT_vals, label = "OAT")
plt.semilogx(particle_nums, tau_opt_TAT_vals, label = "TAT")
plt.xlim(N_min, N_max)
plt.xlabel(r"$N$")
plt.ylabel(r"$N^{2/3} \chi t_{\mathrm{opt}}$")
plt.legend(loc = "best")
plt.tight_layout()
if save: plt.savefig(fig_dir + "optimal_times.pdf")

if show: plt.show()
