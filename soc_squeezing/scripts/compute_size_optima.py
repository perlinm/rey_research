#!/usr/bin/env python3

# FILE CONTENTS: computes optimal squeezing times and values as a function of system size

import sys
import numpy as np
import pandas as pd
import scipy.linalg as linalg

from scipy.optimize import minimize_scalar
from scipy.integrate import solve_ivp

from dicke_methods import spin_op_vec_mat_dicke
from squeezing_methods import spin_squeezing, squeezing_OAT, evolve

data_dir = "../data/"
sqz_fname = data_dir + "sqz_{}.txt"
time_fname = data_dir + "time_{}.txt"

methods = [ "OAT", "TVF", "TAT" ]
OAT, TVF, TAT = methods

max_tau = 2
time_steps = 1000

max_time = lambda N : max_tau * N**(-2/3)

N_min = 5
N_max = 1e4
N_vals = 200

# value of N at which we switch methods for determining optimal TVF/TAT parameters
N_crossover = 900

particle_nums = np.logspace(np.log10(N_min), np.log10(N_max), N_vals)
particle_nums = np.unique(particle_nums.round().astype(int))


##########################################################################################
# compute optimal squeezing values and times
##########################################################################################

# get optimal squeezing parameters by exact diagonalization
def get_optima_diagonalization(N, H, S_op_vec, SS_op_mat, init_state):
    this_H = H.toarray()
    diags_H = np.diag(this_H)
    off_diags_H = np.diag(this_H,1)
    eig_vals, eig_vecs = linalg.eigh_tridiagonal(diags_H, off_diags_H)
    this_S_op_vec = np.array([ eig_vecs.T @ X @ eig_vecs for X in S_op_vec ])
    this_SS_op_mat = np.array([ [ eig_vecs.T @ X @ eig_vecs for X in XS ]
                              for XS in SS_op_mat ])
    this_init_state = eig_vecs.T @ init_state

    def squeezing_TVF_nval(chi_t):
        state_t =  np.exp(-1j * chi_t * eig_vals) * this_init_state
        return -spin_squeezing(N, state_t, this_S_op_vec, this_SS_op_mat)
    optimum_TVF = minimize_scalar(squeezing_TVF_nval, method = "bounded",
                                  bounds = (0, max_time(N)))
    return -optimum_TVF.fun, optimum_TVF.x

# get optimal squeezing parameters by simulation
def get_optima_simulation(N, H, S_op_vec, SS_op_mat, init_state):
    chi_times = np.linspace(0, max_time(N), time_steps)
    d_chi_t = ( chi_times[-1] - chi_times[0] ) / time_steps
    state = init_state.copy()

    sqz_val = -1
    for tt in range(time_steps):
        last_sqz_val = sqz_val
        state = evolve(state, H, d_chi_t)
        sqz_val = spin_squeezing(N, state, S_op_vec, SS_op_mat)
        if sqz_val < last_sqz_val:
            return last_sqz_val, chi_times[tt-1]

    return None, None

# initialize optimal squeezing parameters
sqz_vals = {}
time_vals = {}
zero_frame = pd.Series(np.zeros(len(particle_nums)), index = particle_nums)
for method in methods:
    sqz_vals[method] = zero_frame.copy(deep = True)
    time_vals[method] = zero_frame.copy(deep = True)

# compute optimal squeezing parameters for all system sizes
for N in particle_nums:
    print("N:",N)

    ### use exact analytical results for one-axis twisting (OAT)

    def sqz_OAT_nval(chi_t): return -squeezing_OAT(N, chi_t)
    optimum_OAT = minimize_scalar(sqz_OAT_nval, method = "bounded",
                                  bounds = (0, max_time(N)))
    time_vals[OAT].at[N] = optimum_OAT.x
    sqz_vals[OAT].at[N] = -optimum_OAT.fun

    ### compute spin vectors, spin-spin matrices, Hamiltonians, and initial state,
    ###   exploiting conservation of Dicke state parity to reduce the Hilbert space
    S_op_vec, SS_op_mat = spin_op_vec_mat_dicke(N)
    S_op_vec = [ X[::2,::2] for X in S_op_vec ]
    SS_op_mat = [ [ X[::2,::2] for X in XS ] for XS in SS_op_mat ]

    H_TVF = SS_op_mat[1][1] - N/2 * S_op_vec[0]
    H_TAT = 1/3 * ( SS_op_mat[2][2] - SS_op_mat[1][1] ).real

    state_nZ = np.zeros(N//2+1)
    state_nZ[0] = 1

    if N < N_crossover:
        get_optima = get_optima_diagonalization
    else:
        get_optima = get_optima_simulation

    sqz_vals[TVF].at[N], time_vals[TVF].at[N] = \
        get_optima(N, H_TVF, S_op_vec, SS_op_mat, state_nZ)

    sqz_vals[TAT].at[N], time_vals[TAT].at[N] = \
        get_optima(N, H_TAT, S_op_vec, SS_op_mat, state_nZ)


##########################################################################################
# save results to data files
##########################################################################################

header_common = "# first column = particle number\n"
header_time = "# time in units with OAT twisting strength chi = 1\n"

for method in methods:
    for frame, fname in [ [ sqz_vals[method], sqz_fname.format(method) ],
                          [ time_vals[method], time_fname.format(method) ] ]:
        with open(fname, "w") as f:
            f.write(header_common)
            if "time" in fname: f.write(header_time)
        frame.to_csv(fname, header = False, mode = "a")
