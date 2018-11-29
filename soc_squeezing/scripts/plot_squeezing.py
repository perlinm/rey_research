#!/usr/bin/env python3

import sys, scipy
import numpy as np
import matplotlib.pyplot as plt

from dicke_methods import spin_op_vec_mat_dicke, coherent_spin_state
from squeezing_methods import spin_squeezing, squeezing_from_correlators, squeezing_OAT
from correlator_methods import compute_correlators, convert_zxy_mat, convert_zxy
from jump_methods import correlators_from_trajectories

np.set_printoptions(linewidth = 200)

show = "show" in sys.argv
save = "save" in sys.argv

figsize = (3,2.4)
fig_dir = "../figures/squeezing/"
params = { "text.usetex" : True,
           "font.size" : 8 }
plt.rcParams.update(params)


##########################################################################################
# simulation options
##########################################################################################

log10_N = 4
N = 10**log10_N # number of spins
order_cap = 40 # order limit for short-time correlator expansion
trajectories = 1000 # number of trajectories to use in quantum jump simulations

time_steps = 200 # time steps in plot
ivp_tolerance = 1e-10 # relative error tolerance in numerical integrator
max_tau = 2 # for simulation: chi * max_time = max_tau * N **(-2/3)

# (excitation, dephasing, decay) rates for single- and collective-spin operators
# in units of the OAT strength (i.e. \chi in \chi S_\z^2)
dec_rates = [ (1,1,1), (0,0,0) ]

methods = [ "OAT", "TAT", "TNT" ]


##########################################################################################
# compute squeezing parameters without decoherence
##########################################################################################

# determine simulation times in units of the OAT strength \chi
max_time = max_tau * N**(-2/3)
times = np.linspace(0, max_time, time_steps)

### exact calculations

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

state_TAT = scipy.integrate.solve_ivp(deriv_TAT, (0,times[-1]), init_nZ, t_eval = times,
                                      rtol = ivp_tolerance, atol = ivp_tolerance).y
state_TNT = scipy.integrate.solve_ivp(deriv_TNT, (0,times[-1]), init_nZ, t_eval = times,
                                      rtol = ivp_tolerance, atol = ivp_tolerance).y

sqz_TAT = np.array([ spin_squeezing(N, state_TAT[:,tt], S_op_vec, SS_op_mat)
                     for tt in range(times.size) ])
sqz_TNT = np.array([ spin_squeezing(N, state_TNT[:,tt], S_op_vec, SS_op_mat)
                     for tt in range(times.size) ])

del S_op_vec, SS_op_mat, H_TAT, H_TNT, init_nZ, state_TAT, state_TNT

sqz = { "OAT" : sqz_OAT, "TAT" : sqz_TAT, "TNT" : sqz_TNT }

sqz_max = max([ max(sqz[method]) for method in methods ])
sqz_min = min([ min(sqz[method]) for method in methods ])

### correlator expansions

correlators_OAT_B = compute_correlators(N, order_cap, times, init_state, h_OAT)
correlators_TAT_B = compute_correlators(N, order_cap, times, init_state, h_TAT)
correlators_TNT_B = compute_correlators(N, order_cap, times, init_state, h_TNT)

sqz_OAT_B = squeezing_from_correlators(N, correlators_OAT_B)
sqz_TAT_B = squeezing_from_correlators(N, correlators_TAT_B)
sqz_TNT_B = squeezing_from_correlators(N, correlators_TNT_B)

del correlators_OAT_B, correlators_TAT_B, correlators_TNT_B

sqz_B = { "OAT" : sqz_OAT_B, "TAT" : sqz_TAT_B, "TNT" : sqz_TNT_B }


##########################################################################################
# compute squeezing parameters with decoherence -- in a rotated basis (z,x) --> (x,-z)
##########################################################################################

# construct Hamiltonians in (z,x,y) format
h_OAT = { (0,2,0) : 1 }
h_TAT = { (0,2,0) : +1/3,
          (0,0,2) : -1/3 }
h_TNT = { (0,2,0) : 1,
          (1,0,0) : -N/2 }
init_state = "-Z"

# construct transformation matrix to rotate jump operators
dec_mat_zxy = np.array([ [ 0, -1, 0 ],
                         [ 1,  0, 0 ],
                         [ 0,  0, 1 ]])
dec_mat = convert_zxy_mat(dec_mat_zxy)

### exact and quantum trajectory calculations

sqz_OAT_D_exact = squeezing_OAT(N, times, dec_rates[0])

init_state_vec = coherent_spin_state(init_state, N)
def jump_args(hamiltonian):
    return [ N, trajectories, times, init_state_vec, hamiltonian, dec_rates, dec_mat ]
correlators_TAT_J = correlators_from_trajectories(*jump_args(h_TAT))
correlators_TNT_J = correlators_from_trajectories(*jump_args(h_TNT))

sqz_TAT_J = squeezing_from_correlators(N, correlators_TAT_J)
sqz_TNT_J = squeezing_from_correlators(N, correlators_TNT_J)

sqz_D_exact = { "OAT" : sqz_OAT_D_exact, "TAT" : sqz_TAT_J, "TNT" : sqz_TNT_J }

del init_state_vec, correlators_TAT_J, correlators_TNT_J

### correlator expansions

def correlator_args(hamiltonian):
    return [ N, order_cap, times, init_state, hamiltonian, dec_rates, dec_mat ]
correlators_OAT_D = compute_correlators(*correlator_args(h_OAT))
correlators_TAT_D = compute_correlators(*correlator_args(h_TAT))
correlators_TNT_D = compute_correlators(*correlator_args(h_TNT))

sqz_OAT_D = squeezing_from_correlators(N, correlators_OAT_D)
sqz_TAT_D = squeezing_from_correlators(N, correlators_TAT_D)
sqz_TNT_D = squeezing_from_correlators(N, correlators_TNT_D)

del correlators_TAT_D, correlators_OAT_D, correlators_TNT_D

sqz_D = { "OAT" : sqz_OAT_D, "TAT" : sqz_TAT_D, "TNT" : sqz_TNT_D }


##########################################################################################
# make squeezing plots
##########################################################################################

time_pad = 1/3 # fractional time to add past TAT squeezing minimum
sqz_pad = 1/10
trajectory_marker_size = 2

def positive(vals):
    idx = np.argmax(vals < 0)
    if idx == 0: return len(vals)
    else: return idx

max_plot_time = min(max_time, times[np.argmin(sqz_TAT)]*(1+time_pad))

### coherent evolution

plt.figure(figsize = figsize)
line = {}
for method in methods:
    line[method], = plt.semilogy(times, sqz[method], label = method)
    positive_vals = positive(sqz_B[method])
    plt.semilogy(times[:positive_vals], sqz_B[method][:positive_vals],
                 "--", color = line[method].get_color())
    if positive_vals < len(times):
        plt.semilogy(times[positive_vals-1],[sqz_B[method][positive_vals-1]],
                     "o", color = line[method].get_color())

plt.xlabel(r"$\chi t$")
plt.ylabel(r"$\xi^2$")

plt.xlim(0, max_plot_time)
plt.gca().ticklabel_format(axis = "x", style = "scientific", scilimits = (0,0))

ymin = min(sqz_TAT.min(), sqz_TAT_B[:positive(sqz_TAT_B)].min())
ymax = min(sqz_TNT.max(), sqz_TNT_B[:positive(sqz_TNT_B)].max())
yrange = ymax/ymin
yscale = yrange**sqz_pad
plt.ylim(ymin/yscale, ymax*yscale)

plt.legend(loc = "best")
plt.tight_layout()
if save: plt.savefig(fig_dir + "coherent.pdf")

### evolution with weak decoherence

plt.figure(figsize = figsize)
line = {}
for method in methods:
    linestyle = "-" if method == "OAT" else "."
    line[method], = plt.semilogy(times, sqz_D_exact[method], linestyle, label = method,
                                 markersize = trajectory_marker_size)
    positive_vals = positive(sqz_D[method])
    plt.semilogy(times[:positive_vals], sqz_D[method][:positive_vals],
                 "--", color = line[method].get_color())
    if positive_vals < len(times):
        plt.semilogy(times[positive_vals-1],[sqz_D[method][positive_vals-1]],
                     "o", color = line[method].get_color())

plt.xlabel(r"$\chi t$")
plt.ylabel(r"$\xi^2$")

plt.xlim(0, max_plot_time)
plt.gca().ticklabel_format(axis = "x", style = "scientific", scilimits = (0,0))

ymin = min(sqz_TAT_J.min(), sqz_TAT_D[:positive(sqz_TAT_D)].min())
ymax = min(sqz_TNT_J.max(), sqz_TNT_D[:positive(sqz_TNT_D)].max())
yrange = ymax/ymin
yscale = yrange**sqz_pad
plt.ylim(ymin/yscale, ymax*yscale)

plt.legend(loc = "best")
plt.tight_layout()
if save: plt.savefig(fig_dir + "decoherence_weak.pdf")

if show: plt.show()

# TODO:
# save/read squeezing from data files
# write script to submit this job to terra
