#!/usr/bin/env python3

import os, sys, itertools
import numpy as np
import pandas as pd
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
params = { "text.usetex" : True }
plt.rcParams.update(params)

data_dir = "../data/"
fig_dir = "../figures/"

min_tau = 0
max_tau = 2
time_steps = 1000

N_min = 5
N_max = 1e4
N_vals = 200

# value of N at which we switch methods for determining optimal TAT parameters
N_crossover = 1500

tau_vals = np.linspace(min_tau, max_tau, time_steps)
particle_nums = np.logspace(np.log10(N_min), np.log10(N_max), N_vals)
particle_nums = np.unique(particle_nums.round().astype(int))


##########################################################################################
# method to compute optimal squeezing values and times
##########################################################################################

def to_dB(x): return 10*np.log10(x)

def compute_squeezing():
    zero_frame = pd.DataFrame(np.zeros(len(particle_nums)), index = particle_nums)
    squeezing_OAT_vals = zero_frame.copy(deep = True)
    squeezing_TAT_vals = zero_frame.copy(deep = True)
    time_OAT_vals = zero_frame.copy(deep = True)
    time_TAT_vals = zero_frame.copy(deep = True)
    for N in particle_nums:
        print("N:",N)

        time_bounds = np.array([min_tau, max_tau]) * N**(-2/3)

        # determine optimal OAT squeezing parameters
        def squeezing_OAT_val(chi_t):
            return to_dB(squeezing_OAT(chi_t, N))
        optimum_OAT = minimize_scalar(squeezing_OAT_val,
                                      method = "bounded", bounds = time_bounds)
        time_OAT_vals.at[N] = optimum_OAT.x
        squeezing_OAT_vals.at[N] = -optimum_OAT.fun

        # construct TAT Hamiltonian and spin operator vector / matrix
        S_op_vec, SS_op_mat = spin_op_vec_mat_dicke(N)
        H_TAT = 1/3 * np.real( SS_op_mat[0][0] - SS_op_mat[2][2] )
        state = coherent_spin_state([0,1,0], N)

        if N < N_crossover:
            vals, vecs = linalg.eigh(H_TAT.toarray())
            state = vecs.T @ state
            S_op_vec = np.array([ vecs.T @ X @ vecs for X in S_op_vec ])
            SS_op_mat = np.array([ [ vecs.T @ X @ vecs for X in XS ]
                                   for XS in SS_op_mat ])

            def squeezing_TAT_val(chi_t):
                state_t =  np.exp(-1j * chi_t * vals) * state
                return to_dB(spin_squeezing(state_t, S_op_vec, SS_op_mat, N)[0])
            optimum_TAT = minimize_scalar(squeezing_TAT_val,
                                          method = "bounded", bounds = time_bounds)
            time_TAT_vals.at[N] = optimum_TAT.x
            squeezing_TAT_vals.at[N] = -optimum_TAT.fun

        else:
            chi_times = tau_vals * N**(-2/3)
            d_chi_t = ( chi_times[-1] - chi_times[0] ) / time_steps
            last_squeezing_val = 2
            for ii in range(time_steps):
                squeezing_val, _ = spin_squeezing(state, S_op_vec, SS_op_mat, N)
                if squeezing_val > last_squeezing_val:
                    squeezing_TAT_vals.at[N] = -to_dB(last_squeezing_val)
                    time_TAT_vals.at[N] = chi_times[ii-1]
                    break
                state = evolve(state, H_TAT, d_chi_t)
                last_squeezing_val = squeezing_val

    return squeezing_OAT_vals, squeezing_TAT_vals, time_OAT_vals, time_TAT_vals


##########################################################################################
# plot optimal squeezing values and times
##########################################################################################

squeezing_OAT_fname = data_dir + "squeezing_OAT.txt"
squeezing_TAT_fname = data_dir + "squeezing_TAT.txt"
time_OAT_fname = data_dir + "time_OAT.txt"
time_TAT_fname = data_dir + "time_TAT.txt"

header_common = "# first column = particle number\n"
header_chi_t = "# time in units with OAT twisting strength chi = 1\n"

def pd_read_csv(fname):
    return pd.read_csv(fname, comment = "#", squeeze = True, header = None, index_col = 0)

# determine whether we can compute squeezing and time values
compute_optima = False
if not os.path.isfile(squeezing_OAT_fname):
    compute_optima = True
else:
    test_vals = pd_read_csv(time_OAT_fname)
    f_particle_nums = test_vals.index.values
    if ( f_particle_nums[0] > particle_nums[0] or
         f_particle_nums[-1] < particle_nums[-1] or
         f_particle_nums.size < particle_nums.size):
        compute_optima = True

if compute_optima:
    squeezing_OAT_vals, squeezing_TAT_vals, time_OAT_vals, time_TAT_vals = \
        compute_squeezing()
    if save:
        for frame, fname in zip([ squeezing_OAT_vals, squeezing_TAT_vals,
                                  time_OAT_vals, time_TAT_vals ],
                                [ squeezing_OAT_fname, squeezing_TAT_fname,
                                  time_OAT_fname, time_TAT_fname ]):
            with open(fname, "w") as f:
                f.write(header_common)
                if "time" in fname:
                    f.write(header_chi_t)
            frame.to_csv(fname, header = False, mode = "a")
else:
    squeezing_OAT_vals = pd_read_csv(squeezing_OAT_fname)
    squeezing_TAT_vals = pd_read_csv(squeezing_TAT_fname)
    time_OAT_vals = pd_read_csv(time_OAT_fname)
    time_TAT_vals = pd_read_csv(time_TAT_fname)

N_min, N_max = time_OAT_vals.index[0], time_OAT_vals.index[-1]

plt.figure(figsize = figsize)
plt.semilogx(squeezing_OAT_vals, label = "OAT")
plt.semilogx(squeezing_TAT_vals, label = "TAT")
plt.xlim(N_min, N_max)
plt.xlabel(r"$N$")
plt.ylabel(r"$-10\log_{10}(\xi_{\mathrm{opt}}^2)$")
plt.legend(loc = "best")
plt.tight_layout()
if save: plt.savefig(fig_dir + "optimal_squeezing.pdf")

plt.figure(figsize = figsize)
plt.loglog(time_OAT_vals, label = "OAT")
plt.loglog(time_TAT_vals, label = "TAT")
plt.xlim(N_min, N_max)
plt.xlabel(r"$N$")
plt.ylabel(r"$\chi t_{\mathrm{opt}}$")
plt.legend(loc = "best")
plt.tight_layout()
if save: plt.savefig(fig_dir + "optimal_times.pdf")

if show: plt.show()
