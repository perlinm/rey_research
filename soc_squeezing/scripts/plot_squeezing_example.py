#!/usr/bin/env python3

##########################################################################################
# FILE CONTENTS:
# makes an example squeezing vs. time figure for the SOC/squeezing paper
##########################################################################################

import os, sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from correlator_methods import squeezing_ops
from squeezing_methods import squeezing_OAT, spin_squeezing, squeezing_from_correlators
from dicke_methods import spin_op_vec_mat_dicke, coherent_spin_state
from sr87_olc_constants import recoil_energy_NU, colors


lattice_size = 100 # linear size of 2-D lattice
lattice_depth = 4 # recoil energies
confining_depth = 60 # recoil energies
dec_rate_SI = 0.1 # 1/sec
max_time_SI = 2 # seconds; maximum time in squeezing plot
ymax = 36

# WARNING: manually retrieved from print_olc_squeezing_info.py (!!!)
# for (lattice_depth, confining_depth) = (4, 60) E_R
if lattice_size == 40:
    title = r"{\bf (a)} $\ell=40$"
    chi = 2.731876693400312e-07 # in lattice units with recoil_energy_NU = 1
elif lattice_size == 100:
    title = r"{\bf (b)} $\ell=100$"
    chi = 4.3687077035181746e-08
else:
    print(f"invalid lattice size: {lattice_size}")
    exit()

ivp_tolerance = 1e-10 # relative error tolerance in numerical integrator

data_dir = "../data/"
fig_dir = "../figures/"

figsize = (3,2.5)
params = { "font.family" : "sans-serif",
           "font.serif" : "Computer Modern",
           "text.usetex" : True }
plt.rcParams.update(params)

sqz_label = r"Squeezing (dB)"
time_label = r"Time (sec)"

def to_dB(vals): return 10*np.log10(vals) # convert values to decibels


##########################################################################################
# set some variables we will need

N = lattice_size**2 # total number of particles
chi_NU = chi * recoil_energy_NU # chi (OAT squeezing strength) in natural units

# times to plot
max_time = 2 * N**(-2/3)
time_steps = 200
times = np.linspace(0, max_time, time_steps)
times_SI = times / chi_NU

# decoherence rates: dephasing and decay
dec_rate = dec_rate_SI / chi_NU
dec_rates = (0, dec_rate, dec_rate)


##########################################################################################
# compute squeezing

# squeezing via OAT, both with and without decoherence
sqz_OAT = -to_dB(squeezing_OAT(N, times))
sqz_OAT_dec = -to_dB(squeezing_OAT(N, times, dec_rates))

# compute squeezing via TAT without decoherence
S_op_vec, SS_op_mat = spin_op_vec_mat_dicke(N)
S_op_vec = [ X.todok()[::2,::2] for X in S_op_vec ]
SS_op_mat = [ [ X.todok()[::2,::2] for X in XS ] for XS in SS_op_mat ]

H_TAT = 1/3 * ( SS_op_mat[1][2] + SS_op_mat[2][1] )
init_nZ = np.zeros(S_op_vec[0].shape[0], dtype = complex)
init_nZ[0] = 1

states_TAT = solve_ivp(lambda time, state : -1j * H_TAT.dot(state),
                       (0,times[-1]), init_nZ, t_eval = times,
                       rtol = ivp_tolerance, atol = ivp_tolerance).y
sqz_TAT = np.array([ spin_squeezing(N, states_TAT[:,tt], S_op_vec, SS_op_mat))
                     for tt in range(times.size) ])
sqz_TAT = -to_dB(sqz_TAT)

# extract squeezing via TAT with decoherence
file_name = data_dir + f"TAT/exact_{lattice_depth:.1f}_{lattice_size}.txt"
derivs = np.loadtxt(file_name, dtype = complex)

order_cap = derivs.shape[1]
times_k = np.array([ times**kk for kk in range(order_cap) ])

orders = np.arange(order_cap//2, order_cap+1)
sqz_vals = np.zeros((orders.size, time_steps))
peak_idx = np.zeros(orders.size, dtype = int)
for order_idx, order in enumerate(orders):
    correlators = {}
    for op_idx, op in enumerate(squeezing_ops):
        op = squeezing_ops[op_idx]
        correlators[op] = derivs[op_idx,:order] @ times_k[:order,:]

    sqz = squeezing_from_correlators(N, correlators, in_dB = True)
    sqz_vals[order_idx,:] = sqz

    # get index of first local maximum and inflection point
    d_sqz = sqz[1:] - sqz[:-1] # first derivative
    d_sqz_idx = np.argmax(d_sqz < 0) # index of first local maximum
    dd_sqz = d_sqz[1:] - d_sqz[:-1] # second derivative
    dd_sqz_idx = np.argmax(dd_sqz > 0) # index of first inflection point

    # if either index is zero, it means we did not actually find what we needed
    if  d_sqz_idx == 0:  d_sqz_idx = sqz.size
    if dd_sqz_idx == 0: dd_sqz_idx = sqz.size

    # the peak is at whichever event occurred first
    peak_idx[order_idx] = min(d_sqz_idx, dd_sqz_idx+1)

max_time_TAT_dec_idx = peak_idx.max()
sqz_TAT_dec = sqz_vals[peak_idx.argmax(),:]


##########################################################################################
# make and save figure

plt.figure(figsize = figsize)
plt.title(title)
plt.plot(times_SI, sqz_OAT, "-", color = colors[0], label = "OAT")
plt.plot(times_SI, sqz_OAT_dec, "--", color = colors[0])

plt.plot(times_SI, sqz_TAT, "-", color = colors[1], label = "TAT")
plt.plot(times_SI[:max_time_TAT_dec_idx], sqz_TAT_dec[:max_time_TAT_dec_idx],
         "--", color = colors[1])

plt.xlim(times_SI[0], max_time_SI)
plt.ylim(0, ymax)
plt.xlabel(time_label)
plt.ylabel(sqz_label)

plt.legend(loc = "best")
plt.tight_layout()

plt.savefig(fig_dir + f"squeezing_example_L{lattice_size}.pdf")
