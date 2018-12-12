#!/usr/bin/env python3

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy

from scipy.integrate import solve_ivp

from dicke_methods import spin_op_vec_mat_dicke, coherent_spin_state
from squeezing_methods import spin_squeezing, squeezing_from_correlators, squeezing_OAT
from correlator_methods import compute_correlators, convert_zxy_mat, convert_zxy
from jump_methods import correlators_from_trajectories

np.set_printoptions(linewidth = 200)

show = "show" in sys.argv
save = "save" in sys.argv

data_dir = "../data/squeezing/"
fig_dir = "../figures/squeezing/"

figsize = (6.5,3)
font = { "family" : "serif",
         "sans-serif" : "Computer Modern Sans serif" }
plt.rc("font",**font)
params = { "text.usetex" : True }
plt.rcParams.update(params)

OAT, TAT, TNT = "OAT", "TAT", "TNT"
TAT_X, TAT_Z = r"TAT$_{+,\mathrm{z}}$", r"TAT$_{-,\mathrm{z}}$"
methods = [ OAT, TAT, TAT_X, TAT_Z ]

max_pulses = 20
time_steps = 200 # time steps in plot
ivp_tolerance = 1e-10 # relative error tolerance in numerical integrator
max_tau = 2 # for simulation: chi * max_time = max_tau * N **(-2/3)

N_vals = [ 100, 1000 ]

# < \bar{B} / \tilde{B} >^{rms}_{f} at f = 5/6 as a function of N
B_fac = { 100 : 0.057603548563242575,
          1000 : 0.01798638849064955 }

# suppression factors on S_z from the rotating frame of the TAT protocol
sup_x = 0.8051961132022772
sup_z = 0.09852764489020495

# strength of the S_z term in units with the OAT strength \chi = 1
# equal to 2 f ( \tilde{B} / U ) ( B_fac ) (N/2)
field_strength = { N : 40*5/6 * B_fac[N] * N/2 for N in N_vals }

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

    H = { OAT : SS_op_mat[1][1],
          TAT : 1/3 * ( SS_op_mat[1][1] - SS_op_mat[2][2] ) }
    H.update({ TAT_X : H[TAT] + field_strength[N] * sup_x * S_op_vec[1],
               TAT_Z : H[TAT] + field_strength[N] * sup_z * S_op_vec[0] })

    init_nZ = np.zeros(S_op_vec[0].shape[0], dtype = complex)
    init_nZ[0] = 1

    min_sqz = { method : np.zeros(21) for method in methods }
    for method in [ OAT, TAT ]:
        print(method)
        states = solve_ivp(lambda time, state : -1j * H[method].dot(state),
                           (0,times[-1]), init_nZ, t_eval = times,
                           rtol = ivp_tolerance, atol = ivp_tolerance).y
        sqz = np.array([ spin_sqz(states[:,tt]) for tt in range(times.size) ])
        min_sqz[method] += sqz.min()
        if method == TAT:
            max_TAT_time = times[np.argmin(sqz)]

    for method in [ TAT_X, TAT_Z ]:
        print(method)
        for pulses in range(max_pulses+1):
            print("", pulses)
            if pulses > 0:
                pulse_period = max_TAT_time / pulses
            else:
                pulse_period = 2*max_time
            time = 0
            pulse = 0
            state = init_nZ.copy()
            sqz = np.zeros(times.size)
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
                if init_state[method] == "-Z":
                    state = state[::-1]
                elif init_state[method] == "+X":
                    state = np.array([ (-1)**nn for nn in range(N+1) ]) * state
                else:
                    print("invalid initial state:",init_state[method])
                time = pulse_time
                pulse = pulse + 1
            min_sqz[method][pulses] = sqz.min()

    for method in methods:
        ax[N].plot(range(max_pulses+1), -10*np.log10(min_sqz[method]), label = method,
                   linestyle = "--", marker = "o")
    ax[N].set_xlabel(r"$\pi$-pulses")

ax[N_vals[0]].set_ylabel(r"Squeezing (dB)")
ax[N_vals[1]].legend(loc = "best")

ax[N_vals[0]].set_title(r"{\bf (a)} " + f"$N={N_vals[0]}$")
ax[N_vals[1]].set_title(r"{\bf (b)} " + f"$N={N_vals[1]}$")

plt.tight_layout()
plt.savefig(f"pulsed_squeezing.pdf")
