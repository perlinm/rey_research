#!/usr/bin/env python3

import sys, scipy
import numpy as np
import matplotlib.pyplot as plt

from dicke_methods import spin_op_vec_mat_dicke, coherent_spin_state
from squeezing_methods import spin_squeezing, squeezing_from_correlators, squeezing_OAT
from correlator_methods import compute_correlators, dec_mat_drive, convert_zxy
from jump_methods import correlators_from_trajectories

np.set_printoptions(linewidth = 200)

show = "show" in sys.argv
save = "save" in sys.argv

figsize = (4,3)
fig_dir = "../figures/squeezing/"
params = { "text.usetex" : True }
plt.rcParams.update(params)


##########################################################################################
# simulation options
##########################################################################################

N = 100 # number of spins
order_cap = 40 # order limit for short-time correlator expansion
trajectories = 500 # number of trajectories to use in quantum jump simulations

time_steps = 200 # time steps in plot
ivp_tolerance = 1e-10 # relative error tolerance in numerical integrator
max_tau = 2 # for simulation: chi * max_time = max_tau * N **(-2/3)

drive_mod_index_zy = 0.9057195866712102 # for TAT protocol about (z,y)
drive_mod_index_xy_1 = 1.6262104442160061 # for TAT protocol about (x,y)
drive_mod_index_xy_2 = 2.2213461342426544 # for TAT protocol about (x,y)

# (excitation, dephasing, decay) rates for single- and collective-spin operators
# in units of the OAT strength (i.e. \chi in \chi S_\z^2)
dec_rates = [ (1, 1, 1), (0, 0, 0) ]


##########################################################################################
# compute squeezing parameters without decoherence
##########################################################################################

# convert value to decibels (dB)
def to_dB(x): return 10*np.log10(x)

# determine simulation times and the size of a single time step
tau_vals = np.linspace(0, max_tau, time_steps)
times = tau_vals * N**(-2/3) # in units of the OAT strength \chi

# compute OAT squeezing parameters
sqz_OAT = squeezing_OAT(N, times)

# compute Hamiltonian, spin vector, spin-spin matrix, and initial states for TAT and TNT,
#   exploiting a parity symmetry in both cases to reduce the size of the Hilbert space
S_op_vec, SS_op_mat = spin_op_vec_mat_dicke(N)
S_op_vec = [ X[::2,::2] for X in S_op_vec ]
SS_op_mat = [ [ X[::2,::2] for X in XS ] for XS in SS_op_mat ]

H_TAT = 1/3 * ( SS_op_mat[1][2] + SS_op_mat[2][1] )
H_TNT = SS_op_mat[1][1] - N/2 * S_op_vec[0]

def deriv_TAT(time, state): return -1j * H_TAT.dot(state)
def deriv_TNT(time, state): return -1j * H_TNT.dot(state)

init_nZ = np.zeros(S_op_vec[0].shape[0], dtype = complex)
init_nZ[0] = 1

state_TAT = scipy.integrate.solve_ivp(deriv_TAT, (0,times[-1]), init_nZ,
                                      t_eval = times, rtol = ivp_tolerance).y
state_TNT = scipy.integrate.solve_ivp(deriv_TNT, (0,times[-1]), init_nZ,
                                      t_eval = times, rtol = ivp_tolerance).y

sqz_TAT = np.array([ spin_squeezing(N, state_TAT[:,tt], S_op_vec, SS_op_mat)
                     for tt in range(times.size) ])
sqz_TNT = np.array([ spin_squeezing(N, state_TNT[:,tt], S_op_vec, SS_op_mat)
                     for tt in range(times.size) ])

del S_op_vec, SS_op_mat, H_TAT, H_TNT, init_nZ, state_TAT, state_TNT


##########################################################################################
# compute squeezing parameters with decoherence
##########################################################################################

sqz_OAT_D = squeezing_OAT(N, times, dec_rates[0])

# construct Hamiltonians in (z,x,y) format
h_TAT = { (0,2,0) : +1/3,
          (0,0,2) : -1/3 }
h_TNT = { (0,2,0) : 1,
          (1,0,0) : -N/2 }
init_state = "-Z"

# compute correlators and squeezing without dechoerence for benchmarking
correlators_TAT_B = compute_correlators(N, order_cap, times, init_state, h_TAT)
correlators_TNT_B = compute_correlators(N, order_cap, times, init_state, h_TNT)

sqz_TAT_B = squeezing_from_correlators(N, correlators_TAT_B)
sqz_TNT_B = squeezing_from_correlators(N, correlators_TNT_B)

del correlators_TAT_B, correlators_TNT_B

# compute correlators and squeezing with decoherence
correlators_TAT_D = compute_correlators(N, order_cap, times, init_state, h_TAT, dec_rates)
correlators_TNT_D = compute_correlators(N, order_cap, times, init_state, h_TNT, dec_rates)

sqz_TAT_D = squeezing_from_correlators(N, correlators_TAT_D)
sqz_TNT_D = squeezing_from_correlators(N, correlators_TNT_D)

del correlators_TAT_D, correlators_TNT_D

compute correlators and squeezing using the quantum jump method
init_state_vec = coherent_spin_state(init_state, N)
def jump_args(hamiltonian):
    return [ N, trajectories, times, init_state_vec, hamiltonian, dec_rates ]

correlators_TAT_J = correlators_from_trajectories(*jump_args(h_TAT))
correlators_TNT_J = correlators_from_trajectories(*jump_args(h_TNT))

sqz_TAT_J = squeezing_from_correlators(N, correlators_TAT_J)
sqz_TNT_J = squeezing_from_correlators(N, correlators_TNT_J)

del correlators_TAT_J, correlators_TNT_J


##########################################################################################
# make squeezing plot
##########################################################################################

def plot_lim_idx(sqz_vals):
    where = np.where((sqz_vals < 0) | np.isnan(sqz_vals))[0]
    if where[0] != 0: return where[0]
    else: return where[1]

plt.figure(figsize = figsize)

plt.title(f"$N={N}$")

line_OAT, = plt.plot(times, sqz_OAT, label = "OAT")
plt.plot(times, sqz_OAT_D, ":", color = line_OAT.get_color())

line_TAT, = plt.plot(times, sqz_TAT, label = "TAT")
lim_TAT_B = plot_lim_idx(sqz_TAT_B)
plt.plot(times[:lim_TAT_B], sqz_TAT_B[:lim_TAT_B],
         "--", color = line_TAT.get_color())
lim_TAT_D = plot_lim_idx(sqz_TAT_D)
plt.plot(times[:lim_TAT_D], sqz_TAT_D[:lim_TAT_D],
         ":", color = line_TAT.get_color())

line_TNT, = plt.plot(times, sqz_TNT, label = "TNT")
lim_TNT_B = plot_lim_idx(sqz_TNT_B)
plt.plot(times[:lim_TNT_B], sqz_TNT_B[:lim_TNT_B],
         "--", color = line_TNT.get_color())
lim_TNT_D = plot_lim_idx(sqz_TNT_D)
plt.plot(times[:lim_TNT_D], sqz_TNT_D[:lim_TNT_D],
         ":", color = line_TNT.get_color())

plt.plot(times, sqz_TAT_J, ".", color = line_TAT.get_color())
plt.plot(times, sqz_TNT_J, ".", color = line_TAT.get_color())

plt.xlim(0,times[-1])
plt.ylim(0,sqz_TAT.max()*1.1)
plt.xlabel(r"$\chi t$")
plt.ylabel(r"$-10\log_{10}(\xi^2)$")
plt.legend(loc = "best")
plt.tight_layout()

if show: plt.show()
