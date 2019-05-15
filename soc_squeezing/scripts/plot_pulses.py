#!/usr/bin/env python3

##########################################################################################
# FILE CONTENTS:
# plots optimal squeezing as a function of CPMG pulse number for several protocols
##########################################################################################

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy

from scipy.integrate import solve_ivp
from matplotlib.ticker import MaxNLocator

from dicke_methods import spin_op_vec_mat_dicke
from squeezing_methods import spin_squeezing

np.set_printoptions(linewidth = 200)

show = "show" in sys.argv
save = "save" in sys.argv

fig_dir = "../figures/squeezing/"

figsize = (5,2.5)
font = { "family" : "serif",
         "sans-serif" : "Computer Modern Sans serif" }
plt.rc("font",**font)
params = { "text.usetex" : True,
           "font.size" : 9 }
plt.rcParams.update(params)

OAT, TAT, TNT = "OAT", "TAT", "TNT"
TAT_X, TAT_Z = r"TAT$_{+,\mathrm{z}}$", r"TAT$_{-,\mathrm{z}}$"
methods = [ OAT, TAT, TAT_X, TAT_Z ]

max_pulses = 15
time_steps = 200 # time steps in plot
ivp_tolerance = 1e-10 # relative error tolerance in numerical integrator
max_tau = 2 # for simulation: chi * max_time = max_tau * N **(-2/3)

N_vals = [ 100, 1000 ]

# < \bar{B} / \tilde{B} >^{rms}_{f} at filling = 5/6 as a function of N
# computed via Monte Carlo sampling (output of sample_field_statistics.py)
filling = 5/6
B_fac = { 100 : 0.041222217002695134,
          1000 : 0.01497047524310684 }

# suppression factors on S_z from the rotating frame of the TAT protocol
# i.e. J_0(\beta_\pm) for \beta_\pm satisfying J_0(2\beta_\pm) = \pm 1/3
sup_x = 0.8051961132022772 # for \beta_+
sup_z = 0.09852764489020495 # for \beta_-

# strength of the S_z term in units with the OAT strength \chi = 1
# equal to 2 f ( \tilde{B} / U ) ( B_fac ) (N/2)
field_strength = { N : 40 * filling * B_fac[N] * N/2 for N in N_vals }

# initial state for each protocol
init_state = { OAT : "+X", TAT : "+X", TAT_X : "+X", TAT_Z : "-Z" }

ax = {}
_, ( ax[N_vals[0]], ax[N_vals[1]] ) = plt.subplots(figsize = figsize, ncols = 2)

for N in N_vals:
    max_time = max_tau * N**(-2/3)
    times = np.linspace(0, max_time, time_steps)

    # compute Hamiltonian, spin vector, spin-spin matrix, and initial states for TAT and TNT,
    #   exploiting a parity symmetry in both cases to reduce the size of the Hilbert space
    S_op_vec, SS_op_mat = spin_op_vec_mat_dicke(N)
    S_op_vec = [ X for X in S_op_vec ]
    SS_op_mat = [ [ X for X in XS ] for XS in SS_op_mat ]
    def spin_sqz(state):
        return spin_squeezing(N, state, S_op_vec, SS_op_mat)

    # construct all Hamiltonians
    H = { OAT : SS_op_mat[1][1],
          TAT : 1/3 * ( SS_op_mat[1][1] - SS_op_mat[2][2] ) }
    H.update({ TAT_X : H[TAT] + field_strength[N] * sup_x * S_op_vec[1],
               TAT_Z : H[TAT] + field_strength[N] * sup_z * S_op_vec[0] })

    # define initial state in -Z
    init_nZ = np.zeros(S_op_vec[0].shape[0], dtype = complex)
    init_nZ[0] = 1

    # initialize minimum squeezing values for all methods
    min_sqz = { method : np.zeros(max_pulses+1) for method in methods }

    # compute minimal squeezing for OAT and TAT without pulses
    for method in [ OAT, TAT ]:
        print(method)
        states = solve_ivp(lambda time, state : -1j * H[method].dot(state),
                           (0,times[-1]), init_nZ, t_eval = times,
                           rtol = ivp_tolerance, atol = ivp_tolerance).y
        sqz = np.array([ spin_sqz(states[:,tt]) for tt in range(times.size) ])
        min_sqz[method] += sqz.min()
        if method == TAT:
            max_TAT_time = times[np.argmin(sqz)]

    # compute minimal squeezing for TAT_X and TAT_Z with pulses
    for method in [ TAT_X, TAT_Z ]:
        print(method)
        for pulses in range(max_pulses+1):
            print("", pulses)
            if pulses > 0:
                pulse_period = max_TAT_time / pulses
            else:
                pulse_period = 2*max_time # long enough that we simulate until max_time
            time = 0
            pulse = 0
            state = init_nZ.copy()
            sqz = np.zeros(times.size) # initialize squeezing values at sampled times
            sqz[0] = spin_squeezing(N, state, S_op_vec, SS_op_mat)
            while time < max_time:
                pulse_time = min(pulse_period * (pulse+1/2), max_time)
                save_idx = (times > time) & (times <= pulse_time)
                save_times = times[save_idx]
                if save_times.size == 0 or save_times[-1] < pulse_time:
                    save_times = np.append(save_times, pulse_time)
                states = solve_ivp(lambda time, state : -1j * H[method].dot(state),
                                   (time,pulse_time), state, t_eval = save_times,
                                   rtol = ivp_tolerance, atol = ivp_tolerance).y
                sqz[save_idx] = np.array([ spin_sqz(states[:,tt])
                                           for tt in range(np.sum(save_idx)) ])
                state = states[:,-1]
                if init_state[method] == "-Z": # apply a pi-pulse about X
                    state = state[::-1]
                elif init_state[method] == "+X": # apply a pi-pulse about Z
                    state = np.array([ (-1)**nn for nn in range(N+1) ]) * state
                else:
                    print("invalid initial state:",init_state[method])
                time = pulse_time
                pulse = pulse + 1
            min_sqz[method][pulses] = sqz.min()

    for method in methods:
        ax[N].plot(range(max_pulses+1), -10*np.log10(min_sqz[method]), "o", label = method)
    ax[N].set_xlabel(r"Pulses")

ax[N_vals[0]].set_ylabel(r"Squeezing (dB)")
ax[N_vals[1]].legend(loc = "best", framealpha = 1)

ax[N_vals[0]].set_title(r"{\bf (a)} " + f"$N={N_vals[0]}$")
ax[N_vals[1]].set_title(r"{\bf (b)} " + f"$N={N_vals[1]}$")

for val in N_vals:
    ax[val].set_xticks(range(0,max_pulses+1,5))

plt.tight_layout()
plt.savefig(fig_dir + "pulsed_squeezing.pdf")
