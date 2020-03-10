#!/usr/bin/env python3

import numpy as np
import os, itertools, functools
import scipy.sparse as sparse
import matplotlib.pyplot as plt

from squeezing_methods import spin_squeezing
from dicke_methods import coherent_spin_state as coherent_state_PS
from multibody_methods import dist_method, multibody_problem
from operator_product_methods import build_shell_operator

np.set_printoptions(linewidth = 200)
cutoff = 1e-10

lattice_shape = (2,4)
alpha = 3 # power-law couplings ~ 1 / r^\alpha

# values of the ZZ coupling to simulate in an XXZ model
sweep_coupling_zz = np.linspace(-1,3,41)

# values of the ZZ coupling to inspect more closely
inspect_coupling_zz = [ -1 ]

max_time = 10 # in units of J_\perp

ivp_tolerance = 1e-10 # error tolerance in the numerical integrator

data_dir = "../data/projectors/"
fig_dir = "../figures/shells/"

figsize = (5,4)
params = { "font.size" : 16,
           "text.usetex" : True,
           "text.latex.preamble" : [ r"\usepackage{amsmath}" ]}
plt.rcParams.update(params)

lattice_dim = len(lattice_shape)
spin_num = np.product(lattice_shape)

##########################################################################################
# build SU(n) interaction matrix,
# decompose the interaction matrix into generators of interaction eigenstates
##########################################################################################

dist = dist_method(lattice_shape)

sunc = {}
sunc["mat"] = np.zeros((spin_num,spin_num))
for pp, qq in np.ndindex(sunc["mat"].shape):
    _dist = dist(pp,qq)
    if _dist == 0: continue
    sunc["mat"][pp,qq] = -1/_dist**alpha

excitation_mat, vector_to_tensor, tensor_to_vector \
    = multibody_problem(sunc["mat"], 2, lattice_shape)
shell_num = excitation_mat.shape[0]

eig_vals, eig_vecs = np.linalg.eig(excitation_mat)
eigs = { shell : { "val" : eig_vals[shell], "vec" : eig_vecs[:,shell] }
         for shell in range(shell_num) }

# decompose couplings in the basis of coefficients that generate eigenstates
for shell in range(shell_num):
    sunc_shell_magnitude = eigs[shell]["vec"] @ tensor_to_vector(sunc["mat"])
    sunc[shell] = sunc_shell_magnitude * vector_to_tensor(eigs[shell]["vec"])

##########################################################################################
# compute states and operators in the Z-projection/shell basis
##########################################################################################

# 1-local Z, X, Y operators
local_ops = { "Z" : np.array([[ 1,   0 ], [  0, -1 ]]),
              "X" : np.array([[ 0,   1 ], [  1,  0 ]]),
              "Y" : np.array([[ 0, -1j ], [ 1j,  0 ]]) }

# 2-local products of Z, X, Y operators
for op_lft, op_rht in itertools.product(local_ops.keys(), repeat = 2):
    mat_lft = local_ops[op_lft]
    mat_rht = local_ops[op_rht]
    local_ops[op_lft + op_rht] = np.kron(mat_lft,mat_rht).reshape((2,)*4)

# collective spin vector, and its outer product with itself
def _pauli_mat(pauli):
    full_pauli_op = build_shell_operator([np.ones(spin_num)], [local_ops[pauli]], sunc)
    return full_pauli_op.reshape( ( (spin_num+1)*shell_num, )*2 )
S_op_vec = [ sparse.csr_matrix(_pauli_mat(pauli))/2 for pauli in [ "Z", "X", "Y" ] ]
SS_op_mat = [ [ AA @ BB for BB in S_op_vec ] for AA in S_op_vec ]

# build the ZZ perturbation operator in the Z-projection/shell basis
shell_coupling_mat = build_shell_operator([sunc["mat"]], [local_ops["ZZ"]], sunc)

# energies and energy eigenstates within each sector of fixed spin projection
def energies_states(zz_sun_ratio):
    energies = np.zeros( ( spin_num+1, shell_num ) )
    eig_states = np.zeros( ( spin_num+1, shell_num, shell_num ) )

    for spins_up in range(spin_num+1):
        # construct the Hamiltonian at this Z projection, from SU(n) + ZZ couplings
        _proj_hamiltonian \
            = np.diag(eig_vals) + zz_sun_ratio * shell_coupling_mat[spins_up,:,spins_up,:]

        # diagonalize the net Hamiltonian at this Z projection
        energies[spins_up,:], eig_states[spins_up,:,:] \
            = np.linalg.eigh(_proj_hamiltonian)

        spins_dn = spin_num - spins_up
        if spins_up == spins_dn: break
        energies[spins_dn,:], eig_states[spins_dn,:,:] \
            = energies[spins_up,:], eig_states[spins_up,:,:]

    return energies, eig_states

# coherent spin state
def coherent_spin_state(vec):
    zero_shell = np.zeros(shell_num)
    zero_shell[0] = 1
    return np.outer(coherent_state_PS(vec,spin_num), zero_shell)

##########################################################################################
# simulate!
##########################################################################################

# note: extra factor of 1/2 in che_eff_bare for compatibility with Chunlei's work
chi_eff_bare = 1/4 * sunc["mat"].sum() / np.math.comb(spin_num,2)
state_X = coherent_spin_state([0,1,0])

def _states(initial_state, zz_sun_ratio, times):
    energies, eig_states = energies_states(zz_sun_ratio)
    init_state_eig = np.einsum("zSs,zS->zs", eig_states, initial_state)
    phases = np.exp(-1j * np.tensordot(times, energies, axes = 0))
    evolved_eig_states = phases * init_state_eig[None,:,:]
    return np.einsum("zsS,tzS->tzs", eig_states, evolved_eig_states)

def simulate(coupling_zz, max_tau = 2, overshoot_ratio = 1.5, points = 500):
    print("coupling_zz:", coupling_zz)
    zz_sun_ratio = coupling_zz - 1

    # determine how long to simulate
    if zz_sun_ratio != 0:
        chi_eff = abs(zz_sun_ratio * chi_eff_bare)
        sim_time = min(max_time, max_tau * spin_num**(-2/3) / chi_eff)
    else:
        sim_time = max_time

    times = np.linspace(0, sim_time, points)

    # compute states at all times of interest
    # note: factor of 1/2 included for compatibility with Chunlei's work
    states = _states(state_X, zz_sun_ratio/2, times)

    # compute squeezing
    sqz = np.array([ spin_squeezing(spin_num, state.flatten(), S_op_vec, SS_op_mat)
                     for state in states ])

    # don't look too far beyond the maximum squeezing time
    max_tt = int( np.argmin(sqz) * overshoot_ratio )
    if max_tt == 0:
        max_tt = len(times)
    else:
        max_tt = min(max_tt, len(times))

    times = times[:max_tt]
    sqz = sqz[:max_tt]

    # compute populations
    pops = np.einsum("tzs->ts", abs(states[:max_tt])**2)

    return times, sqz, pops

def name_tag(coupling_zz = None):
    base_tag = f"N{spin_num}_D{lattice_dim}_a{alpha}"
    if coupling_zz == None: return base_tag
    else: return base_tag + f"_z{coupling_zz}"

def to_dB(sqz):
    return 10*np.log10(np.array(sqz))

if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

##########################################################################################
print("running inspection simulations")

for coupling_zz in inspect_coupling_zz:
    times, sqz, pops = simulate(coupling_zz)
    title_text = f"$N={spin_num},~D={lattice_dim},~\\alpha={alpha}," \
               + f"~J_{{\mathrm{{z}}}}/J_\perp={coupling_zz}$"

    plt.figure(figsize = figsize)
    plt.title(title_text)
    plt.plot(times, to_dB(sqz), "k")
    plt.ylim(plt.gca().get_ylim()[0], 0)
    plt.xlabel(r"time ($J_\perp t$)")
    plt.ylabel(r"$\xi_{\mathrm{min}}^2$ (dB)")
    plt.tight_layout()

    plt.savefig(fig_dir + f"squeezing_{name_tag(coupling_zz)}.pdf")

    plt.figure(figsize = figsize)
    plt.title(title_text)
    plt.plot(times, pops[:,0])
    plt.plot([times[0], times[-1]], [0,0])
    plt.plot(times, pops[:,1:].sum(axis = 1)) ########################################
    for ss in range(1,shell_num):
        plt.plot(times, pops[:,ss], "k--")
    plt.axvline(times[np.argmin(sqz)], color = "gray", linestyle  = "--")
    plt.xlabel(r"time ($J_\perp t$)")
    plt.ylabel("population")
    plt.tight_layout()

    plt.savefig(fig_dir + f"populations_{name_tag(coupling_zz)}.pdf")

##########################################################################################
if len(sweep_coupling_zz) == 0: exit()
print("running sweep simulations")

sweep_results = [ simulate(coupling_zz) for coupling_zz in sweep_coupling_zz ]
sweep_times, sweep_sqz, sweep_pops = zip(*sweep_results)

sweep_min_sqz = [ min(sqz) for sqz in sweep_sqz ]
min_sqz_idx = [ max(1,np.argmin(sqz)) for sqz in sweep_sqz ]

title_text = f"$N={spin_num},~D={lattice_dim},~\\alpha={alpha}$"

plt.figure(figsize = figsize)
plt.title(title_text)
plt.plot(sweep_coupling_zz, to_dB(sweep_min_sqz), "ko")
plt.ylim(plt.gca().get_ylim()[0], 0)
plt.xlabel(r"$J_{\mathrm{z}}/J_\perp$")
plt.ylabel(r"$\xi_{\mathrm{min}}^2$ (dB)")
plt.tight_layout()
plt.savefig(fig_dir + f"squeezing_N{spin_num}_D{lattice_dim}_a{alpha}.pdf")

sweep_pops = [ np.vstack([ pops[:min_idx,0], pops[:min_idx,1:].sum(axis = 1) ])
               for pops, min_idx in zip(sweep_pops, min_sqz_idx) ]
sweep_min_pops = np.array([ pops.min(axis = 1) for pops in sweep_pops ])
sweep_max_pops = np.array([ pops.max(axis = 1) for pops in sweep_pops ])

plt.figure(figsize = figsize)
plt.title(title_text)
plt.plot(sweep_coupling_zz, sweep_min_pops[:,0], "o")
plt.plot(sweep_coupling_zz, sweep_max_pops[:,1], "o")
plt.xlabel(r"$J_{\mathrm{z}}/J_\perp$")
plt.ylabel("population")
plt.tight_layout()
plt.savefig(fig_dir + f"populations_N{spin_num}_D{lattice_dim}_a{alpha}.pdf")

print("completed")
