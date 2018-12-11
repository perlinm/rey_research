#!/usr/bin/env python3

import os, sys
import numpy as np
import matplotlib.pyplot as plt

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

figsize = (3,2.4)
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

# determine simulation times in units of the OAT strength \chi
max_time = max_tau * N**(-2/3)
times = np.linspace(0, max_time, time_steps)

# (excitation, dephasing, decay) rates for single- and collective-spin operators
# in units of the OAT strength (i.e. \chi in \chi S_\z^2)
dec_rates = [ (1,1,1), (0,0,0) ]


OAT, TAT, TNT = "OAT", "TAT", "TNT"
methods = [ OAT, TAT, TNT ]

# construct Hamiltonians in (z,x,y) format
init_state = "-Z"
h_OAT = { (0,2,0) : 1 }
h_TAT = { (0,2,0) : +1/3,
          (0,0,2) : -1/3 }
h_TNT = { (0,2,0) : 1,
          (1,0,0) : -N/2 }
h_vec = { OAT : h_OAT, TAT : h_TAT, TNT : h_TNT }

sqz_header = ""
sqz_header += r"# first column: time in units of the OAT stregth \chi" + "\n"
sqz_header += r"# second column: squeezing as \xi^2" + "\n"
order_header = f"# order_cap: {order_cap}\n"
trajectory_header = f"# trajectories: {trajectories}\n"

def write_sqz(sqz, sqz_path, extra_header = ""):
    with open(sqz_path, "w") as f:
        f.write(sqz_header)
        f.write(extra_header)
        for tt in range(times.size):
            f.write(f"{times[tt]},{sqz[tt]}\n")

def read_sqz(fname):
    sqz_times, sqz = np.loadtxt(sqz_path, delimiter = ",", unpack = True)
    assert(abs(sqz_times - times).max() < times[1]/1e10)
    return sqz


##########################################################################################
# compute squeezing parameters without decoherence
##########################################################################################

### exact calculations

sqz_C_exact = { OAT : squeezing_OAT(N, times) }

# compute Hamiltonian, spin vector, spin-spin matrix, and initial states for TAT and TNT,
#   exploiting a parity symmetry in both cases to reduce the size of the Hilbert space
S_op_vec, SS_op_mat = spin_op_vec_mat_dicke(N)
S_op_vec = [ X[::2,::2] for X in S_op_vec ]
SS_op_mat = [ [ X[::2,::2] for X in XS ] for XS in SS_op_mat ]

H = { OAT : SS_op_mat[1][1],
      TAT : 1/3 * ( SS_op_mat[1][2] + SS_op_mat[2][1] ),
      TNT : SS_op_mat[1][1] - N/2 * S_op_vec[0] }

init_nZ = np.zeros(S_op_vec[0].shape[0], dtype = complex)
init_nZ[0] = 1

for method in methods:
    if method == OAT: continue
    sqz_path = data_dir + f"sqz_C_exact_logN{log10_N}_{method}.txt"

    if not os.path.isfile(sqz_path):
        states = solve_ivp(lambda time, state : -1j * H[method].dot(state),
                           (0,times[-1]), init_nZ, t_eval = times,
                           rtol = ivp_tolerance, atol = ivp_tolerance).y
        sqz = np.array([ spin_squeezing(N, states[:,tt], S_op_vec, SS_op_mat)
                         for tt in range(times.size) ])
        write_sqz(sqz, sqz_path)
        del states, sqz

    sqz_C_exact[method] = read_sqz(sqz_path)

del S_op_vec, SS_op_mat, H, init_nZ

sqz_max = max([ max(sqz_C_exact[method]) for method in methods ])
sqz_min = min([ min(sqz_C_exact[method]) for method in methods ])

### correlator expansions

sqz_C_trunc = {}
for method in methods:
    sqz_path = data_dir + f"sqz_C_trunc_logN{log10_N}_{method}.txt"

    if not os.path.isfile(sqz_path):
        correlators = compute_correlators(N, order_cap, times, init_state, h_vec[method])
        sqz = squeezing_from_correlators(N, correlators)
        write_sqz(sqz, sqz_path, order_header)
        del correlators, sqz

    sqz_C_trunc[method] = read_sqz(sqz_path)


##########################################################################################
# compute squeezing parameters with decoherence -- in a rotated basis (z,x) --> (x,-z)
##########################################################################################

### exact and quantum trajectory calculations

sqz_D_exact = { OAT : squeezing_OAT(N, times, dec_rates[0]) }
if dec_rates[1] != (0,0,0):
    print("WARNING: 'exact' simulations do not account for collective decoherence!")

# construct transformation matrix to rotate jump operators
dec_mat_zxy = np.array([ [ 0, -1, 0 ],
                         [ 1,  0, 0 ],
                         [ 0,  0, 1 ]])
dec_mat = convert_zxy_mat(dec_mat_zxy)

init_state_vec = coherent_spin_state(init_state, N)
def jump_args(hamiltonian):
    return [ N, trajectories, times, init_state_vec, hamiltonian, dec_rates, dec_mat ]

for method in methods:
    if method == OAT: continue
    sqz_path = data_dir + f"sqz_D_exact_logN{log10_N}_{method}.txt"

    if not os.path.isfile(sqz_path):
        correlators = correlators_from_trajectories(*jump_args(h_vec[method]))
        sqz = squeezing_from_correlators(N, correlators)
        write_sqz(sqz, sqz_path, trajectory_header)
        del correlators, sqz

    sqz_D_exact[method] = read_sqz(sqz_path)

del init_state_vec

### correlator expansions

def correlator_args(h_vec):
    return [ N, order_cap, times, init_state, h_vec, dec_rates, dec_mat ]

sqz_D_trunc = {}
for method in methods:
    sqz_path = data_dir + f"sqz_D_trunc_logN{log10_N}_{method}.txt"

    if not os.path.isfile(sqz_path):
        correlators = compute_correlators(*correlator_args(h_vec[method]))
        sqz = squeezing_from_correlators(N, correlators)
        write_sqz(sqz, sqz_path, order_header)
        del correlators, sqz

    sqz_D_trunc[method] = read_sqz(sqz_path)


##########################################################################################
# make squeezing plots
##########################################################################################

time_pad = 1/3 # fractional time to add past TAT squeezing minimum
sqz_pad = 1/10
trajectory_marker_size = 2

def positive(vals):
    idx = np.argmax(vals < 0)
    if idx == 0: return vals.size
    else: return idx

max_plot_time = min(max_time, times[np.argmin(sqz_C_exact[TAT])]*(1+time_pad))
def ylims(sqz_exact, sqz_trunc):
    idx_TAT = min(positive(sqz_trunc[TAT]), np.argmax(times > max_plot_time))
    idx_TNT = min(positive(sqz_trunc[TNT]), np.argmax(times > max_plot_time))
    ymin = min(sqz_exact[TAT].min(), sqz_trunc[TAT][:idx_TAT].min())
    ymax = min(sqz_exact[TNT].max(), sqz_trunc[TNT][:idx_TNT].max())
    yscale = (ymax/ymin)**sqz_pad
    return ymin/yscale, ymax*yscale

### coherent evolution

plt.figure(figsize = figsize)
line = {}
for method in methods:
    line[method], = plt.semilogy(times, sqz_C_exact[method], label = method)
    positive_vals = positive(sqz_C_trunc[method])
    plt.semilogy(times[:positive_vals], sqz_C_trunc[method][:positive_vals],
                 "--", color = line[method].get_color())
    if positive_vals < len(times):
        plt.semilogy(times[positive_vals-1],[sqz_C_trunc[method][positive_vals-1]],
                     "o", color = line[method].get_color())

plt.xlabel(r"$\chi t$")
plt.ylabel(r"$\xi^2$")

plt.xlim(0, max_plot_time)
plt.ylim(*ylims(sqz_C_exact,sqz_C_trunc))
plt.gca().ticklabel_format(axis = "x", style = "scientific", scilimits = (0,0))

plt.legend(loc = "best")
plt.tight_layout()
if save: plt.savefig(fig_dir + "coherent.pdf")

### evolution with weak decoherence

plt.figure(figsize = figsize)
line = {}
for method in methods:
    linestyle = "-" if method == OAT else "."
    line[method], = plt.semilogy(times, sqz_D_exact[method], linestyle, label = method,
                                 markersize = trajectory_marker_size)
    positive_vals = positive(sqz_D_trunc[method])
    plt.semilogy(times[:positive_vals], sqz_D_trunc[method][:positive_vals],
                 "--", color = line[method].get_color())
    if positive_vals < len(times):
        plt.semilogy(times[positive_vals-1],[sqz_D_trunc[method][positive_vals-1]],
                     "o", color = line[method].get_color())

plt.xlabel(r"$\chi t$")
plt.ylabel(r"$\xi^2$")

plt.xlim(0, max_plot_time)
plt.ylim(*ylims(sqz_D_exact,sqz_D_trunc))
plt.gca().ticklabel_format(axis = "x", style = "scientific", scilimits = (0,0))

plt.legend(loc = "best")
plt.tight_layout()
if save: plt.savefig(fig_dir + "decoherence_weak.pdf")

if show: plt.show()
