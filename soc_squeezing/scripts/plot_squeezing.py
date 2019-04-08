#!/usr/bin/env python3

import os, sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from dicke_methods import spin_op_vec_mat_dicke, coherent_spin_state
from squeezing_methods import spin_squeezing, squeezing_from_correlators, squeezing_OAT
from correlator_methods import compute_correlators, mat_zxy_to_pzm, vec_zxy_to_pzm
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
N = int(10**log10_N) # number of spins
order_cap = 35 # order limit for short-time correlator expansion
trajectories = 100 # number of trajectories to use in quantum jump simulations
recompute_exact = False
recompute_trunc = False

time_steps = 100 # time steps in plot
ivp_tolerance = 1e-10 # relative error tolerance in numerical integrator
max_tau = 1 # for simulation: chi * max_time = max_tau * N **(-2/3)

# determine simulation times in units of the OAT strength \chi
max_time = max_tau * N**(-2/3)
times = np.linspace(0, max_time, time_steps)

# (excitation, dephasing, decay) rates for single- and collective-spin operators
# in units of the OAT strength (i.e. \chi in \chi S_\z^2)
dec_rates = [ (1,)*3, (0,)*3 ]
dec_rates_strong = [ (100,)*3, (0,)*3 ]

OAT, TAT, TNT = "OAT", "TAT", "TNT"
methods = [ OAT, TAT, TNT ]

# construct Hamiltonians in (z,x,y) format, assuming an initial state in +X
h_vec = { OAT : { (2,0,0) : 1 },
          TAT : { (2,0,0) : +1/3,
                  (0,0,2) : -1/3 },
          TNT : { (2,0,0) : 1,
                  (0,1,0) : N/2 } }
for method, vec in h_vec.items():
    h_vec[method] = vec_zxy_to_pzm(vec, "+X")
init_state = "-Z"

# construct transformation matrix to rotate jump operators: rotate +X into -Z
basis_change_zxy = np.array([ [ 0, -1, 0 ],
                              [ 1,  0, 0 ],
                              [ 0,  0, 1 ]])
dec_mat = mat_zxy_to_pzm(basis_change_zxy)

sqz_header = ""
sqz_header += r"# first column: time in units of the OAT strength \chi" + "\n"
sqz_header += r"# second column: squeezing as \xi^2" + "\n"
order_header = f"# order_cap: {order_cap}\n"
trajectory_header = f"# trajectories: {trajectories}\n"

def write_sqz(times, sqz, sqz_path, extra_header = ""):
    with open(sqz_path, "w") as f:
        f.write(sqz_header)
        f.write(extra_header)
        for tt in range(times.size):
            f.write(f"{times[tt]},{sqz[tt]}\n")

def read_sqz(fname):
    return np.loadtxt(sqz_path, delimiter = ",", unpack = True)


##########################################################################################
# compute squeezing parameters without decoherence
##########################################################################################

### exact calculations

time_sqz_C_exact = { OAT : [ times.copy(), squeezing_OAT(N, times) ] }

# compute Hamiltonian, spin vector, spin-spin matrix, and initial states for TAT and TNT,
#   exploiting a parity symmetry in both cases to reduce the size of the Hilbert space
S_op_vec, SS_op_mat = spin_op_vec_mat_dicke(N)
S_op_vec = [ X.todok()[::2,::2] for X in S_op_vec ]
SS_op_mat = [ [ X.todok()[::2,::2] for X in XS ] for XS in SS_op_mat ]

H = { OAT : SS_op_mat[1][1],
      TAT : 1/3 * ( SS_op_mat[1][2] + SS_op_mat[2][1] ),
      TNT : SS_op_mat[1][1] - N/2 * S_op_vec[0] }

init_nZ = np.zeros(S_op_vec[0].shape[0], dtype = complex)
init_nZ[0] = 1

for method in methods:
    if method == OAT: continue
    sqz_path = data_dir + f"sqz_C_exact_logN{log10_N}_{method}.txt"

    if not os.path.isfile(sqz_path) or recompute_exact:
        states = solve_ivp(lambda time, state : -1j * H[method].dot(state),
                           (0,times[-1]), init_nZ, t_eval = times,
                           rtol = ivp_tolerance, atol = ivp_tolerance).y
        sqz = np.array([ spin_squeezing(N, states[:,tt], S_op_vec, SS_op_mat)
                         for tt in range(times.size) ])
        write_sqz(times, sqz, sqz_path)
        del states, sqz

    time_sqz_C_exact[method] = read_sqz(sqz_path)

del S_op_vec, SS_op_mat, H, init_nZ

sqz_max = max([ max(time_sqz_C_exact[method][1]) for method in methods ])
sqz_min = min([ min(time_sqz_C_exact[method][1]) for method in methods ])

### correlator expansions

time_sqz_C_trunc = {}
for method in methods:
    sqz_path = data_dir + f"sqz_C_trunc_logN{log10_N}_{method}.txt"

    if not os.path.isfile(sqz_path) or recompute_trunc:
        correlators = compute_correlators(times, order_cap, N, init_state, h_vec[method])
        sqz = squeezing_from_correlators(N, correlators)
        write_sqz(times, sqz, sqz_path, order_header)
        del correlators, sqz

    time_sqz_C_trunc[method] = read_sqz(sqz_path)


##########################################################################################
# compute squeezing parameters with decoherence -- in a rotated basis (z,x) --> (x,-z)
##########################################################################################

### exact and quantum trajectory calculations

time_sqz_D_exact = { OAT : [ times.copy(), squeezing_OAT(N, times, dec_rates[0]) ] }
if dec_rates[1] != (0,0,0):
    print("WARNING: 'exact' simulations do not account for collective decoherence!")

init_state_vec = coherent_spin_state(init_state, N)
def jump_args(hamiltonian):
    return [ N, trajectories, times, init_state_vec, hamiltonian, dec_rates, dec_mat ]

for method in methods:
    if method == OAT: continue
    sqz_path = data_dir + f"sqz_D_exact_logN{log10_N}_{method}.txt"
    if method == TNT:
        sqz_path = data_dir + f"test_sqz_D_exact_logN{log10_N}_{method}.txt"

    if not os.path.isfile(sqz_path) or recompute_exact:
        correlators = correlators_from_trajectories(*jump_args(h_vec[method]))
        sqz = squeezing_from_correlators(N, correlators)
        write_sqz(times, sqz, sqz_path, trajectory_header)
        del correlators, sqz

    time_sqz_D_exact[method] = read_sqz(sqz_path)

del init_state_vec

### correlator expansions

def correlator_args(h_vec):
    return [ times, order_cap, N, init_state, h_vec, dec_rates, dec_mat ]

time_sqz_D_trunc = {}
for method in methods:
    sqz_path = data_dir + f"sqz_D_trunc_logN{log10_N}_{method}.txt"

    if not os.path.isfile(sqz_path) or recompute_trunc:
        correlators = compute_correlators(*correlator_args(h_vec[method]))
        sqz = squeezing_from_correlators(N, correlators)
        write_sqz(times, sqz, sqz_path, order_header)
        del correlators, sqz

    time_sqz_D_trunc[method] = read_sqz(sqz_path)


##########################################################################################
# compute squeezing parameters with strong decoherence
##########################################################################################

def correlator_args_strong(h_vec):
    return [ times, order_cap, N, init_state, h_vec, dec_rates_strong, dec_mat ]

time_sqz_D_trunc_strong = {}
for method in methods:
    sqz_path = data_dir + f"sqz_D_trunc_logN{log10_N}_{method}_strong.txt"

    if not os.path.isfile(sqz_path) or recompute_trunc:
        correlators = compute_correlators(*correlator_args_strong(h_vec[method]))
        sqz = squeezing_from_correlators(N, correlators)
        write_sqz(times, sqz, sqz_path, order_header)
        del correlators, sqz

    time_sqz_D_trunc_strong[method] = read_sqz(sqz_path)


##########################################################################################
# make squeezing plots
##########################################################################################

# rescale time (or, equivalently, \chi)
max_time *= N
times *= N
for time_sqz in [ time_sqz_C_exact, time_sqz_C_trunc,
                  time_sqz_D_exact, time_sqz_D_trunc,
                  time_sqz_D_trunc_strong ]:
    for method in methods:
        time_sqz[method][0] *= N

time_pad = 1/3 # fractional time to add past TAT squeezing minimum
sqz_pad = 1/10
trajectory_marker_size = 2
trunc_width = 2

def positive(vals):
    idx = np.argmax(vals < 0)
    if idx == 0: return vals.size
    else: return idx

max_plot_time = min(max_time, times[np.argmin(time_sqz_C_exact[TAT][1])]*(1+time_pad))
def ylims(time_sqz_exact, time_sqz_trunc):
    idx_TAT = positive(time_sqz_trunc[TAT][1])
    idx_TNT = positive(time_sqz_trunc[TNT][1])
    ymin = min(time_sqz_exact[TAT][1].min(), time_sqz_trunc[TAT][1][:idx_TAT].min())
    ymax = min(time_sqz_exact[TNT][1].max(), time_sqz_trunc[TNT][1][:idx_TNT].max())
    yscale = (ymax/ymin)**sqz_pad
    return ymin/yscale, ymax*yscale

# darken a color provided in hexidecimal format
import colorsys
def darken_color(color, darken_val = 0.08):
    # convert hexidecimal to [0,1)^3 RGB format
    color = tuple([ int(color[jj:jj+2], 16)/255 - darken_val
                           for jj in (1, 3 ,5) ])
    return tuple( min(max(cc-darken_val,0), 1) for cc in color )

### coherent evolution

plt.figure(figsize = figsize)
color = {}
for method in methods:
    line, = plt.semilogy(time_sqz_C_exact[method][0],
                         time_sqz_C_exact[method][1], label = method)
    color[method] = line.get_color()
    positive_vals = positive(time_sqz_C_trunc[method][1])
    plt.semilogy(time_sqz_C_trunc[method][0][:positive_vals],
                 time_sqz_C_trunc[method][1][:positive_vals],
                 "--", linewidth = trunc_width, color = darken_color(color[method]))
    if positive_vals < len(times):
        plt.semilogy([time_sqz_C_trunc[method][0][positive_vals-1]],
                     [time_sqz_C_trunc[method][1][positive_vals-1]],
                     "o", color = darken_color(color[method]))

plt.xlabel(r"$N\chi t$")
plt.ylabel(r"$\xi^2$")

plt.xlim(0, max_plot_time)
plt.ylim(*ylims(time_sqz_C_exact,time_sqz_C_trunc))

plt.legend(loc = "best")
plt.tight_layout()
if save: plt.savefig(fig_dir + "coherent.pdf")


### evolution with weak decoherence

plt.figure(figsize = figsize)
for method in methods:
    linestyle = "-" if method == OAT else "."
    plt.semilogy(time_sqz_D_exact[method][0],
                 time_sqz_D_exact[method][1], linestyle, label = method,
                 color = color[method], markersize = trajectory_marker_size)
    positive_vals = positive(time_sqz_D_trunc[method][1])
    plt.semilogy(time_sqz_D_trunc[method][0][:positive_vals],
                 time_sqz_D_trunc[method][1][:positive_vals],
                 "--", linewidth = trunc_width, color = darken_color(color[method]))
    if positive_vals < len(times):
        plt.semilogy([time_sqz_D_trunc[method][0][positive_vals-1]],
                     [time_sqz_D_trunc[method][1][positive_vals-1]],
                     "o", color = darken_color(color[method]))

plt.xlabel(r"$N\chi t$")
plt.ylabel(r"$\xi^2$")

plt.xlim(0, max_plot_time)
plt.ylim(*ylims(time_sqz_D_exact,time_sqz_D_trunc))

plt.legend(loc = "best", numpoints = 3)
plt.tight_layout()
if save: plt.savefig(fig_dir + "decoherence_weak.pdf")


### evolution with strong decoherence

plt.figure(figsize = figsize)
for method in methods:
    positive_vals = positive(time_sqz_D_trunc_strong[method][1])
    plt.semilogy(time_sqz_D_trunc_strong[method][0][:positive_vals],
                 time_sqz_D_trunc_strong[method][1][:positive_vals],
                 color = color[method], label = method)
    if positive_vals < len(times):
        plt.semilogy([time_sqz_D_trunc_strong[method][0][positive_vals-1]],
                     [time_sqz_D_trunc_strong[method][1][positive_vals-1]],
                     "o", color = color[method])

plt.xlabel(r"$N\chi t$")
plt.ylabel(r"$\xi^2$")

strong_time_lim_idx = positive(time_sqz_D_trunc_strong[TAT][1])
plt.xlim(0, time_sqz_D_trunc_strong[TAT][0][strong_time_lim_idx]*(1+time_pad))

plt.legend(loc = "best")
plt.tight_layout()
if save: plt.savefig(fig_dir + "decoherence_strong.pdf")


if show: plt.show()
