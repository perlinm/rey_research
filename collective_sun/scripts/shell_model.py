#!/usr/bin/env python3

import numpy as np
import os, sys, itertools, functools
import matplotlib.pyplot as plt

from squeezing_methods import spin_squeezing
from dicke_methods import coherent_spin_state as coherent_state_PS
from multibody_methods import dist_method, get_multibody_states
from operator_product_methods import build_shell_operator

np.set_printoptions(linewidth = 200)
cutoff = 1e-10

lattice_shape = (3,4)
manifolds = [ 0, 2, 4 ]
alpha = 3 # power-law couplings ~ 1 / r^\alpha

# values of the ZZ coupling to simulate in an XXZ model
sweep_coupling_zz = np.linspace(-1,3,41)

# values of the ZZ coupling to inspect more closely
inspect_coupling_zz = [ -1 ]
inspect_sim_time = 2

max_time = 10 # in units of J_\perp

plot_all_shells = False # plot the population for each shell?

ivp_tolerance = 1e-10 # error tolerance in the numerical integrator

data_dir = "../data/projectors/"
fig_dir = "../figures/shells/"

figsize = (5,4)
params = { "font.size" : 16,
           "text.usetex" : True,
           "text.latex.preamble" : [ r"\usepackage{amsmath}",
                                     r"\usepackage{braket}" ]}
plt.rcParams.update(params)

lattice_dim = len(lattice_shape)
spin_num = np.product(lattice_shape)

print("lattice shape:",lattice_shape)
##########################################################################################
# build SU(n) interaction matrix, and build generators of interaction eigenstates
##########################################################################################
print("builing generators of interaction eigenstates")

dist = dist_method(lattice_shape)

sunc = {}
sunc["mat"] = np.zeros((spin_num,spin_num))
for pp, qq in itertools.combinations(range(spin_num),2):
    sunc["mat"][pp,qq] = sunc["mat"][qq,pp] = -1/dist(pp,qq)**alpha
sunc["TI"] = True

# build generators of interaction eigenstates, compute energies, etc.
sunc["shells"], sunc["energies"], sunc_tensors \
    = get_multibody_states(lattice_shape, sunc["mat"], manifolds, sunc["TI"])
sunc.update(sunc_tensors)
shell_num = len(sunc["energies"])

##########################################################################################
# compute states and operators in the shell / Z-projection basis
##########################################################################################
print("building operators in the shell / Z-projection basis")
sys.stdout.flush()

# 1-local Z, X, Y operators
local_ops = { "Z" : np.array([[ 1,   0 ], [  0, -1 ]]),
              "X" : np.array([[ 0,   1 ], [  1,  0 ]]),
              "Y" : np.array([[ 0, -1j ], [ 1j,  0 ]]) }

# 2-local products of Z, X, Y operators
for op_lft, op_rht in itertools.product(local_ops.keys(), repeat = 2):
    mat_lft = local_ops[op_lft]
    mat_rht = local_ops[op_rht]
    local_ops[op_lft + op_rht] = np.kron(mat_lft,mat_rht).reshape((2,)*4)

# build the ZZ perturbation operator in the shell / Z-projection basis
print("building perturbation operator")
shell_coupling_mat \
    = build_shell_operator([sunc["mat"]], [local_ops["ZZ"]], sunc, sunc["TI"])

# collective spin vector, and its outer product with itself
def _pauli_mat(pauli):
    tensors = [np.ones(spin_num)]
    operators = [local_ops[pauli]]
    full_pauli_op = build_shell_operator(tensors, operators, sunc,
                                         sunc["TI"], shell_diagonal = True)
    full_pauli_op.shape = ( shell_num*(spin_num+1), )*2
    return full_pauli_op
print("building collective spin operators")
sys.stdout.flush()
S_op_vec = [ _pauli_mat(pauli)/2 for pauli in [ "Z", "X", "Y" ] ]
SS_op_mat = [ [ AA @ BB for BB in S_op_vec ] for AA in S_op_vec ]

# energies and energy eigenstates within each sector of fixed spin projection
def energies_states(zz_sun_ratio):
    energies = np.zeros( ( shell_num, spin_num+1 ) )
    eig_states = np.zeros( ( shell_num, shell_num, spin_num+1 ), dtype = complex )

    for spins_up in range(spin_num+1):
        # construct the Hamiltonian at this Z projection, from SU(n) + ZZ couplings
        _proj_hamiltonian = np.diag(sunc["energies"]) \
                          + zz_sun_ratio * shell_coupling_mat[:,spins_up,:,spins_up]

        # diagonalize the net Hamiltonian at this Z projection
        energies[:,spins_up], eig_states[:,:,spins_up] = np.linalg.eigh(_proj_hamiltonian)

        spins_dn = spin_num - spins_up
        if spins_up == spins_dn: break
        energies[:,spins_dn], eig_states[:,:,spins_dn] \
            = energies[:,spins_up], eig_states[:,:,spins_up]

    return energies, eig_states

# coherent spin state
def coherent_spin_state(vec):
    zero_shell = np.zeros(shell_num)
    zero_shell[0] = 1
    return np.outer(zero_shell, coherent_state_PS(vec,spin_num))

##########################################################################################
# simulate!
##########################################################################################

# note: extra factor of 1/2 in che_eff_bare for compatibility with Chunlei's work
chi_eff_bare = 1/4 * sunc["mat"].sum() / np.math.comb(spin_num,2)
state_X = coherent_spin_state([0,1,0])

def _states(initial_state, zz_sun_ratio, times):
    energies, eig_states = energies_states(zz_sun_ratio)
    init_state_eig = np.einsum("Ssz,Sz->sz", eig_states, initial_state)
    phases = np.exp(-1j * np.tensordot(times, energies, axes = 0))
    evolved_eig_states = phases * init_state_eig[None,:,:]
    return np.einsum("sSz,tSz->tsz", eig_states, evolved_eig_states)

def simulate(coupling_zz, sim_time = None, max_tau = 2,
             overshoot_ratio = 1.5, points = 500):
    print("coupling_zz:", coupling_zz)
    sys.stdout.flush()
    zz_sun_ratio = coupling_zz - 1

    # determine how long to simulate
    if sim_time is None:
        max_tt = None
        if zz_sun_ratio != 0:
            chi_eff = abs(zz_sun_ratio * chi_eff_bare)
            sim_time = min(max_time, max_tau * spin_num**(-2/3) / chi_eff)
        else:
            sim_time = max_time
    else:
        max_tt = points

    times = np.linspace(0, sim_time, points)

    # compute states at all times of interest
    # note: factor of 1/2 included for compatibility with Chunlei's work
    states = _states(state_X, zz_sun_ratio/2, times)

    # compute squeezing
    sqz = np.array([ spin_squeezing(spin_num, state.flatten(), S_op_vec, SS_op_mat)
                     for state in states ])

    # don't look too far beyond the maximum squeezing time
    if max_tt is None:
        max_tt = int( np.argmin(sqz) * overshoot_ratio )
        if max_tt == 0:
            max_tt = len(times)
        else:
            max_tt = min(max_tt, len(times))

    times = times[:max_tt]
    sqz = sqz[:max_tt]

    # compute populations
    pops = np.einsum("tsz->ts", abs(states[:max_tt])**2)

    return times, sqz, pops

def name_tag(coupling_zz = None):
    lattice_name = "_".join([ str(size) for size in lattice_shape ])
    base_tag = f"L{lattice_name}_a{alpha}"
    if coupling_zz == None: return base_tag
    else: return base_tag + f"_z{coupling_zz}"

def pop_label(manifold, prefix = None):
    label = r"$\braket{\mathcal{P}_{" + str(manifold) + r"}}$"
    if prefix == None:
        return label
    else:
        return prefix + " " + label

def to_dB(sqz):
    return 10*np.log10(np.array(sqz))

if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

##########################################################################################
print("running inspection simulations")
sys.stdout.flush()

lattice_text = r"\times".join([ str(size) for size in lattice_shape ])
common_title = f"L={lattice_text},~\\alpha={alpha}"

for coupling_zz in inspect_coupling_zz:
    title_text = f"${common_title},~J_{{\mathrm{{z}}}}/J_\perp={coupling_zz}$"
    times, sqz, pops = simulate(coupling_zz, sim_time = inspect_sim_time)

    try:
        sqz_end = np.where(sqz[1:] > 1)[0][0] + 2
    except:
        sqz_end = len(times)

    plt.figure(figsize = figsize)
    plt.title(title_text)
    plt.plot(times[:sqz_end], to_dB(sqz)[:sqz_end], "k")
    plt.ylim(plt.gca().get_ylim()[0], 0)
    plt.xlabel(r"time ($J_\perp t$)")
    plt.ylabel(r"$\xi_{\mathrm{min}}^2$ (dB)")
    plt.tight_layout()

    plt.savefig(fig_dir + f"squeezing_{name_tag(coupling_zz)}.pdf")

    plt.figure(figsize = figsize)
    plt.title(title_text)
    for manifold, shells in sunc["shells"].items():
        if plot_all_shells:
            for shell in shells:
                plt.plot(times, pops[:,shell], color = "gray", linestyle = "--")
        plt.plot(times, pops[:,shells].sum(axis = 1), label = pop_label(manifold))
    plt.axvline(times[np.argmin(sqz)], color = "gray", linestyle  = "--")
    plt.xlabel(r"time ($J_\perp t$)")
    plt.ylabel("population")
    plt.legend(loc = "best")
    plt.tight_layout()

    plt.savefig(fig_dir + f"populations_{name_tag(coupling_zz)}.pdf")

##########################################################################################
if len(sweep_coupling_zz) == 0: exit()
print("running sweep simulations")
sys.stdout.flush()

sweep_coupling_zz = sweep_coupling_zz[sweep_coupling_zz != 1]
sweep_results = [ simulate(coupling_zz) for coupling_zz in sweep_coupling_zz ]
sweep_times, sweep_sqz, sweep_pops = zip(*sweep_results)

sweep_min_sqz = [ min(sqz) for sqz in sweep_sqz ]
min_sqz_idx = [ max(1,np.argmin(sqz)) for sqz in sweep_sqz ]

title_text = f"${common_title}$"

plt.figure(figsize = figsize)
plt.title(title_text)
plt.plot(sweep_coupling_zz, to_dB(sweep_min_sqz), "ko")
plt.ylim(plt.gca().get_ylim()[0], 0)
plt.xlabel(r"$J_{\mathrm{z}}/J_\perp$")
plt.ylabel(r"$\xi_{\mathrm{min}}^2$ (dB)")
plt.tight_layout()
plt.savefig(fig_dir + f"squeezing_{name_tag()}.pdf")

sweep_pops = [ np.vstack([ pops[:min_idx,shells].sum(axis = 1)
                           for shells in sunc["shells"].values() ])
               for pops, min_idx in zip(sweep_pops, min_sqz_idx) ]
sweep_min_pops = np.array([ pops.min(axis = 1) for pops in sweep_pops ])
sweep_max_pops = np.array([ pops.max(axis = 1) for pops in sweep_pops ])

plt.figure(figsize = figsize)
plt.title(title_text)
plt.plot(sweep_coupling_zz, sweep_min_pops[:,0], "o",
         label = pop_label(0,"min"))
for idx, manifold in enumerate(sunc["shells"].keys()):
    if manifold == 0: continue
    plt.plot(sweep_coupling_zz, sweep_max_pops[:,idx], "o",
             label = pop_label(manifold,"max"))
plt.xlabel(r"$J_{\mathrm{z}}/J_\perp$")
plt.ylabel("population")
plt.legend(loc = "best")
plt.tight_layout()
plt.savefig(fig_dir + f"populations_{name_tag()}.pdf")

print("completed")
