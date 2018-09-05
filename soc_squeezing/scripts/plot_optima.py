#!/usr/bin/env python3

# FILE CONTENTS: plots optimal squeezing times and values as a function of system size

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
from squeezing_methods import spin_vec_mat_vals, spin_squeezing, evolve, val
from fermi_hubbard_methods import spatial_basis

show = "show" in sys.argv
save = "save" in sys.argv
assert(show or save)

figsize = (4,3)
params = { "text.usetex" : True }
plt.rcParams.update(params)

data_dir = "../data/"
fig_dir = "../figures/"

max_tau = 2
time_steps = 1000

N_min = 5
N_max = 1e4
N_vals = 200

# value of N at which we switch methods for determining optimal TAT parameters
N_crossover = 700

tau_vals = np.linspace(0, max_tau, time_steps)
particle_nums = np.logspace(np.log10(N_min), np.log10(N_max), N_vals)
particle_nums = np.unique(particle_nums.round().astype(int))


##########################################################################################
# method to compute optimal squeezing values and times
##########################################################################################

def compute_squeezing():
    zero_frame = pd.Series(np.zeros(len(particle_nums)), index = particle_nums)
    squeezing_OAT_vals = zero_frame.copy(deep = True)
    squeezing_TVF_vals = zero_frame.copy(deep = True)
    squeezing_TAT_vals = zero_frame.copy(deep = True)
    time_OAT_vals = zero_frame.copy(deep = True)
    time_TVF_vals = zero_frame.copy(deep = True)
    time_TAT_vals = zero_frame.copy(deep = True)
    for N in particle_nums:
        print("N:",N)

        time_bound = max_tau * N**(-2/3)

        # determine optimal OAT squeezing parameters
        def squeezing_OAT_nval(chi_t):
            return -squeezing_OAT(N,chi_t)
        optimum_OAT = minimize_scalar(squeezing_OAT_nval, method = "bounded",
                                      bounds = (0,time_bound))
        time_OAT_vals.at[N] = optimum_OAT.x
        squeezing_OAT_vals.at[N] = -optimum_OAT.fun

        # compute spin vectors, spin-spin matrices, Hamiltonians, and initial states,
        #   exploiting parity symmetries to reduce the size of the Hilbert space
        S_op_vec, SS_op_mat = spin_op_vec_mat_dicke(N)
        S_op_vec = [ X[::2,::2] for X in S_op_vec ]
        SS_op_mat = [ [ X[::2,::2] for X in XS ] for XS in SS_op_mat ]

        H_TVF = SS_op_mat[1][1] - N/2 * S_op_vec[0]
        H_TAT = 1/3 * ( SS_op_mat[1][2] + SS_op_mat[2][1] )

        state_TVF = np.zeros(N//2+1)
        state_TVF[0] = 1
        state_TAT = np.copy(state_TVF)

        if N < N_crossover:
            H_TVF = H_TVF.toarray().real
            diags_TVF = np.diag(H_TVF)
            off_diags_TVF = np.diag(H_TVF,1)
            vals_TVF, vecs_TVF = linalg.eigh_tridiagonal(diags_TVF, off_diags_TVF)
            S_op_vec_TVF = np.array([ vecs_TVF.T @ X @ vecs_TVF for X in S_op_vec ])
            SS_op_mat_TVF = np.array([ [ vecs_TVF.T @ X @ vecs_TVF for X in XS ]
                                      for XS in SS_op_mat ])
            state_TVF = vecs_TVF.T @ state_TVF

            def squeezing_TVF_nval(chi_t):
                state_t =  np.exp(-1j * chi_t * vals_TVF) * state_TVF
                return -spin_squeezing(N, state_t, S_op_vec_TVF, SS_op_mat_TVF)
            optimum_TVF = minimize_scalar(squeezing_TVF_nval, method = "bounded",
                                         bounds = (0,time_OAT_vals.at[N]))
            time_TVF_vals.at[N] = optimum_TVF.x
            squeezing_TVF_vals.at[N] = -optimum_TVF.fun

            H_TAT = H_TAT.toarray().real
            diags_TAT = np.diag(H_TAT)
            off_diags_TAT = np.diag(H_TAT,1)
            vals_TAT, vecs_TAT = linalg.eigh_tridiagonal(diags_TAT, off_diags_TAT)
            S_op_vec_TAT = np.array([ vecs_TAT.T @ X @ vecs_TAT for X in S_op_vec ])
            SS_op_mat_TAT = np.array([ [ vecs_TAT.T @ X @ vecs_TAT for X in XS ]
                                       for XS in SS_op_mat ])
            state_TAT = vecs_TAT.T @ state_TAT

            def squeezing_TAT_nval(chi_t):
                state_t =  np.exp(-1j * chi_t * vals_TAT) * state_TAT
                return -spin_squeezing(N, state_t, S_op_vec_TAT, SS_op_mat_TAT)
            optimum_TAT = minimize_scalar(squeezing_TAT_nval, method = "bounded",
                                          bounds = (0,time_bound))
            time_TAT_vals.at[N] = optimum_TAT.x
            squeezing_TAT_vals.at[N] = -optimum_TAT.fun

        else:
            chi_times = tau_vals * N**(-2/3)
            d_chi_t = ( chi_times[-1] - chi_times[0] ) / time_steps

            last_squeezing_val = 2
            for ii in range(time_steps):
                squeezing_val = spin_squeezing(N, state_TVF, S_op_vec, SS_op_mat)
                if squeezing_val > last_squeezing_val:
                    squeezing_TVF_vals.at[N] = last_squeezing_val
                    time_TVF_vals.at[N] = chi_times[ii-1]
                    break
                state_TVF = evolve(state_TVF, H_TVF, d_chi_t)
                last_squeezing_val = squeezing_val

            last_squeezing_val = 2
            for ii in range(time_steps):
                XX_TAT = np.real(val(SS_op_mat[1][1],state_TAT))
                Z_TAT = np.real(val(S_op_vec[0],state_TAT))
                squeezing_val = XX_TAT * N / Z_TAT**2
                if squeezing_val > last_squeezing_val:
                    squeezing_TAT_vals.at[N] = last_squeezing_val
                    time_TAT_vals.at[N] = chi_times[ii-1]
                    break
                state_TAT = evolve(state_TAT, H_TAT, d_chi_t)
                last_squeezing_val = squeezing_val

    return [ squeezing_OAT_vals, squeezing_TVF_vals, squeezing_TAT_vals,
             time_OAT_vals, time_TVF_vals, time_TAT_vals ]


##########################################################################################
# plot optimal squeezing values and times
##########################################################################################

squeezing_OAT_fname = data_dir + "squeezing_OAT.txt"
squeezing_TVF_fname = data_dir + "squeezing_TVF.txt"
squeezing_TAT_fname = data_dir + "squeezing_TAT.txt"
time_OAT_fname = data_dir + "time_OAT.txt"
time_TVF_fname = data_dir + "time_TVF.txt"
time_TAT_fname = data_dir + "time_TAT.txt"

header_common = "# first column = particle number\n"
header_chi_t = "# time in units with OAT twisting strength chi = 1\n"

def pd_read_csv(fname):
    return pd.read_csv(fname, comment = "#", squeeze = True, header = None, index_col = 0)

# determine whether we can compute squeezing and time values
if not os.path.isfile(squeezing_OAT_fname):
    squeezing_OAT_vals, squeezing_TVF_vals, squeezing_TAT_vals, \
        time_OAT_vals, time_TVF_vals, time_TAT_vals = \
            compute_squeezing()
    if save:
        for frame, fname in [ [ squeezing_OAT_vals, squeezing_OAT_fname ],
                              [ squeezing_TVF_vals, squeezing_TVF_fname ],
                              [ squeezing_TAT_vals, squeezing_TAT_fname ],
                              [ time_OAT_vals, time_OAT_fname ],
                              [ time_TVF_vals, time_TVF_fname ],
                              [ time_TAT_vals, time_TAT_fname ] ]:
            with open(fname, "w") as f:
                f.write(header_common)
                if "time" in fname:
                    f.write(header_chi_t)
            frame.to_csv(fname, header = False, mode = "a")
else:
    squeezing_OAT_vals = pd_read_csv(squeezing_OAT_fname)
    squeezing_TVF_vals = pd_read_csv(squeezing_TVF_fname)
    squeezing_TAT_vals = pd_read_csv(squeezing_TAT_fname)
    time_OAT_vals = pd_read_csv(time_OAT_fname)
    time_TVF_vals = pd_read_csv(time_TVF_fname)
    time_TAT_vals = pd_read_csv(time_TAT_fname)

N_min, N_max = time_OAT_vals.index[0], time_OAT_vals.index[-1]

plt.figure(figsize = figsize)
plt.semilogx(squeezing_OAT_vals, label = "OAT")
plt.semilogx(squeezing_TVF_vals, label = "TVF")
plt.semilogx(squeezing_TAT_vals, label = "TAT")
plt.xlim(N_min, N_max)
plt.xlabel(r"$N$")
plt.ylabel(r"$-10\log_{10}(\xi_{\mathrm{opt}}^2)$")
plt.legend(loc = "best")
plt.tight_layout()
if save: plt.savefig(fig_dir + "optimal_squeezing.pdf")

plt.figure(figsize = figsize)
plt.loglog(time_OAT_vals, label = "OAT")
plt.loglog(time_TVF_vals, label = "TVF")
plt.loglog(time_TAT_vals, label = "TAT")
plt.xlim(N_min, N_max)
plt.xlabel(r"$N$")
plt.ylabel(r"$\chi t_{\mathrm{opt}}$")
plt.legend(loc = "best")
plt.tight_layout()
if save: plt.savefig(fig_dir + "optimal_times.pdf")

if show: plt.show()
