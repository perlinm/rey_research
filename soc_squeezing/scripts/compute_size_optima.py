#!/usr/bin/env python3

# FILE CONTENTS: computes optimal squeezing times and values as a function of system size

import sys
import numpy as np
import pandas as pd
import scipy.linalg as linalg
import scipy.optimize  as optimize

from dicke_methods import spin_op_vec_mat_dicke
from squeezing_methods import squeezing_OAT, \
    get_optima_diagonalization, get_optima_simulation

data_dir = "../data/"
sqz_fname = data_dir + "sqz_{}.txt"
time_fname = data_dir + "time_{}.txt"

methods = [ "OAT", "TAT", "TNT" ]
OAT, TAT, TNT = methods

max_tau = 2
max_time = lambda N : max_tau * N**(-2/3)

N_min = 5
N_max = 1e4
N_vals = 200

# value of N at which we switch methods for determining optimal TAT/TNT parameters
N_crossover = 900

particle_nums = np.logspace(np.log10(N_min), np.log10(N_max), N_vals)
particle_nums = np.unique(particle_nums.round().astype(int))


##########################################################################################
# compute optimal squeezing values and times
##########################################################################################

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
    optimum_OAT = optimize.minimize_scalar(sqz_OAT_nval, method = "bounded",
                                           bounds = (0, max_time(N)))
    sqz_vals[OAT].at[N] = -optimum_OAT.fun
    time_vals[OAT].at[N] = optimum_OAT.x

    ### compute spin vectors, spin-spin matrices, Hamiltonians, and initial state,
    ###   exploiting conservation of Dicke state parity to reduce the Hilbert space
    S_op_vec, SS_op_mat = spin_op_vec_mat_dicke(N)
    S_op_vec = [ X[::2,::2] for X in S_op_vec ]
    SS_op_mat = [ [ X[::2,::2] for X in XS ] for XS in SS_op_mat ]

    H_TAT = 1/3 * ( SS_op_mat[2][2] - SS_op_mat[1][1] ).real
    H_TNT = SS_op_mat[1][1] - N/2 * S_op_vec[0]

    state_nZ = np.zeros(N//2+1)
    state_nZ[0] = 1

    if N < N_crossover:
        get_optima = get_optima_diagonalization
    else:
        get_optima = get_optima_simulation

    sqz_vals[TAT].at[N], time_vals[TAT].at[N] = \
        get_optima(N, H_TAT, S_op_vec, SS_op_mat, state_nZ, max_time(N))

    sqz_vals[TNT].at[N], time_vals[TNT].at[N] = \
        get_optima(N, H_TNT, S_op_vec, SS_op_mat, state_nZ, max_time(N))


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
