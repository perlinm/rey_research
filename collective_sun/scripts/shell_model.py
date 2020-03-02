#!/usr/bin/env python3

import numpy as np
import itertools, functools
import matplotlib.pyplot as plt

from dicke_methods import coherent_spin_state as coherent_state_PS
from operator_product_methods import reduced_diagrams, operator_contractions, \
    evaluate_multi_local_op, evaluate_operator_product, build_shell_operator

np.set_printoptions(linewidth = 200)
cutoff = 1e-10

lattice_shape = (3,4)
alpha = 3 # power-law couplings ~ 1 / r^\alpha

# values of the ZZ coupling to simulate in an XXZ model
sweep_coupling_zz = np.linspace(-2,4,25)

# values of the ZZ coupling to inspect more closely
inspect_coupling_zz = [ -1, 0.7 ]

max_time = 10 # in units of J_\perp

ivp_tolerance = 1e-10 # error tolerance in the numerical integrator

data_dir = "../data/projectors/"
fig_dir = "../figures/shells/"

figsize = (5,4)
params = { "font.size" : 16,
           "text.usetex" : True,
           "text.latex.preamble" : [ r"\usepackage{amsmath}",
                                     r"\usepackage{braket}" ]}
plt.rcParams.update(params)

##########################################################################################
# define some basic methods
##########################################################################################

lattice_dim = len(lattice_shape)
spin_num = np.product(lattice_shape)
spin_pairs = [ (pp,qq) for pp, qq in itertools.combinations(range(spin_num), 2) ]
pair_index = { pair : idx for idx, pair in enumerate(spin_pairs) }

# convert between integer and vector indices for a spin
_to_vec = { idx : tuple(vec) for idx, vec in enumerate(np.ndindex(lattice_shape)) }
_to_idx = { vec : idx for idx, vec in _to_vec.items() }
def to_vec(idx):
    if hasattr(idx, "__getitem__"):
        return np.array(idx) % np.array(lattice_shape)
    return np.array(_to_vec[ idx % spin_num ])
def to_idx(vec):
    if type(vec) is int:
        return vec % spin_num
    return _to_idx[ tuple( np.array(vec) % np.array(lattice_shape) ) ]

# get the distance between two spins
def dist_1D(pp, qq, axis):
    diff = ( pp - qq ) % lattice_shape[axis]
    return min(diff, lattice_shape[axis] - diff)
def dist(pp, qq):
    pp = to_vec(pp)
    qq = to_vec(qq)
    return np.sqrt(sum( dist_1D(*pp_qq,aa)**2 for aa, pp_qq in enumerate(zip(pp,qq)) ))

# organize all displacements by their magnitude
dist_to_disps = {}
for dd in range(1,spin_num):
    dd_dist = dist(0, dd)
    try:
        dist_to_disps[dd_dist] += [ dd ]
    except:
        dist_to_disps[dd_dist] = [ dd ]
shell_num = len(dist_to_disps)

# convert "distance" vectors to "displacement" vectors
def unit_vec(dim, idx):
    vec = np.zeros(dim, dtype = int)
    vec[idx] = 1
    return vec
def dist_vec(dist):
    vec = sum( unit_vec(spin_num-1,to_idx(disp)-1) for disp in dist_to_disps[dist] )
    return vec / np.sqrt( len(dist_to_disps[dist]) )
_dist_to_disp = np.vstack([ dist_vec(dist) for dist in dist_to_disps.keys() ]).T
def dist_to_disp_vec(dist_vec):
    return _dist_to_disp @ dist_vec

# define cycle matrices
def cycle_mat(dd, kk = 0):
    dd_vec = to_vec(dd)
    def _shift(pp):
        return to_idx( dd_vec + to_vec(pp) )

    if kk == 0:
        def _val(_): return 1
    else:
        kk_vec = to_vec(kk) * 2*pi/np.array(lattice_shape)
        def _val(pp):
            return np.exp(1j * to_vec(pp) @ kk_vec)

    mat = np.zeros((spin_num,)*2)
    for pp in range(spin_num):
        mat[_shift(pp),pp] = _val(pp)
    return mat

# recover a matrix from a vector of cycle (displacement) matrix coefficients
def disp_vec_to_mat(disp_vec, kk = 0):
    ignore_0 = int( len(disp_vec) == spin_num-1 )
    return sum( disp_vec[dd] * cycle_mat(dd+ignore_0, kk)
                for dd in range(len(disp_vec)) )

# convert a translationally invariant matrix into
#   a vector of displacement matrix coefficients
def mat_to_disp_vec(mat):
    return np.array([ cycle_mat(dd).T.flatten() @ mat.flatten() / spin_num
                      for dd in range(1,spin_num) ])

# methods to convert between matrix and pair-vectors
def mat_to_pair_vec(mat):
    return np.array([ mat[pairs] for pairs in spin_pairs ])
def pair_vec_to_mat(vec):
    mat = np.zeros((spin_num,spin_num))
    for idx, pairs in enumerate(spin_pairs):
        mat[pairs] = vec[idx]
        mat[pairs[::-1]] = vec[idx]
    return mat

##########################################################################################
# collect interaction data
##########################################################################################

_sunc_dist_vec = np.array([ -1/dist**alpha * np.sqrt(len(disps))
                            for dist, disps in dist_to_disps.items() ])
_sunc_disp_vec = dist_to_disp_vec(_sunc_dist_vec)
_sunc_mat = disp_vec_to_mat(_sunc_disp_vec)
_sunc_pair_vec = mat_to_pair_vec(_sunc_mat)
_sunc_col_vec = sum(_sunc_mat)
_sunc_tot = sum(_sunc_pair_vec)
sunc = { "dist" : _sunc_dist_vec,
         "disp" : _sunc_disp_vec,
         "pair" : _sunc_pair_vec,
         "mat" : _sunc_mat,
         "col" : _sunc_col_vec,
         "tot" : _sunc_tot }

##########################################################################################
# decompose interaction matrix into generators of SU(n)-symmetric interaction eigenstates
##########################################################################################

# solve the translationally invariant, isotropic two-body eigenvalue problem
characteristic_matrix = np.zeros((shell_num,)*2)
for aa, aa_disps in enumerate(dist_to_disps.values()):
    characteristic_matrix[aa,aa] += sunc["mat"][0,aa_disps[0]]

    for bb, bb_disps in enumerate(dist_to_disps.values()):
        if bb > aa: break
        aa_bb_val = sum( sunc["mat"][aa_disp,bb_disp]
                         for aa_disp in aa_disps
                         for bb_disp in bb_disps )
        aa_bb_val /= np.sqrt( len(aa_disps) * len(bb_disps) )

        characteristic_matrix[aa,bb] += aa_bb_val
        if bb != aa:
            characteristic_matrix[bb,aa] += aa_bb_val

eig_vals, _eig_dist_vecs = np.linalg.eigh(characteristic_matrix)
eig_vals -= sunc["col"][0]
eig_vals *= 2
eig_vals[abs(eig_vals) < cutoff] = 0
_eig_dist_vecs[abs(_eig_dist_vecs) < cutoff] = 0
assert(eig_vals[0] == 0)

_eig_disp_vecs = _dist_to_disp @ _eig_dist_vecs
_eig_disp_vecs[abs(_eig_disp_vecs) < cutoff] = 0

# collect all two-body problem data in one place
eigs = { shell : { "val" : eig_vals[shell],
                   "disp" : _eig_disp_vecs[:,shell] }
         for shell in range(shell_num) }

# convert vector into projector
def to_proj(vec):
    return np.outer( vec.conj(), vec ) / ( vec.conj() @ vec )

# decompose couplings in the basis of coefficients that generate eigenstates
for shell in range(shell_num):
    sunc_shell_disp = to_proj( eigs[shell]["disp"] ) @ sunc["disp"]
    sunc[shell] = disp_vec_to_mat(sunc_shell_disp)

##########################################################################################
# compute states and operators in the Z-projection/shell basis
##########################################################################################

# 1-local I, Z, X, Y operators
local_ops = { "I" : np.array([[ 1,   0 ], [  0,  1 ]]),
              "Z" : np.array([[ 1,   0 ], [  0, -1 ]]),
              "X" : np.array([[ 0,   1 ], [  1,  0 ]]),
              "Y" : np.array([[ 0, -1j ], [ 1j,  0 ]]) }

# 2-local products of Z, X, Y operators
for op_lft, op_rht in itertools.product(local_ops.keys(), repeat = 2):
    mat_lft = local_ops[op_lft]
    mat_rht = local_ops[op_rht]
    local_ops[op_lft + op_rht] = np.kron(mat_lft,mat_rht).reshape((2,)*4)

# build the ZZ perturbation operator in the Z-projection/shell basis
shell_coupling_mat = build_shell_operator([sunc["mat"]], [local_ops["ZZ"]], sunc)

# energies and energy eigenstates within each sector of fixed spin projection
def energies_states(zz_sun_ratio):
    energies = np.zeros((spin_num+1,shell_num), dtype = complex)
    eig_states = np.zeros((spin_num+1,shell_num,shell_num), dtype = complex)

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
chi_eff_bare = 1/4 * np.mean(sunc["pair"])
state_X = coherent_spin_state([0,1,0])

def _states(initial_state, zz_sun_ratio, times):
    energies, eig_states = energies_states(zz_sun_ratio)
    init_state_eig = np.einsum("zSs,zS->zs", eig_states, initial_state)
    phases = np.exp(-1j * np.tensordot(times, energies, axes = 0))
    evolved_eig_states = phases * init_state_eig[None,:,:]
    return np.einsum("zsS,tzS->tzs", eig_states, evolved_eig_states)

def simulate(coupling_zz, max_tau = 2, overshoot_ratio = 1.5, points = 500):
    zz_sun_ratio = coupling_zz - 1

    if zz_sun_ratio != 0:
        chi_eff = zz_sun_ratio * chi_eff_bare
        sim_time = min(max_time, max_tau * spin_num**(-2/3) / chi_eff)
    else:
        sim_time = max_time

    ##################################################
    sim_time = 2
    ##################################################

    times = np.linspace(0, sim_time, points)

    # note: factor of 1/2 included for compatibility with Chunlei's work
    states = _states(state_X, zz_sun_ratio/2, times)

    pops = np.einsum("tzs->ts", abs(states)**2)

    return times, pops

def name_tag(coupling_zz = None):
    base_tag = f"N{spin_num}_D{lattice_dim}_a{alpha}"
    if coupling_zz == None: return base_tag
    else: return base_tag + f"_z{coupling_zz}"

for coupling_zz in inspect_coupling_zz:
    times, pops = simulate(coupling_zz)
    title_text = f"$N={spin_num},~D={lattice_dim},~\\alpha={alpha}," \
               + f"~J_{{\mathrm{{z}}}}/J_\perp={coupling_zz}$"

    plt.figure(figsize = figsize)
    plt.title(title_text)
    ##################################################
    # plt.plot(times, pops[:,0], "k")
    # for ss in range(1,shell_num):
        # plt.plot(times, pops[:,ss])
    plt.plot(times, pops[:,0])
    plt.plot([times[0], times[-1]], [0,0])
    plt.plot(times, np.sum(pops[:,1:], axis = 1))
    plt.plot([times[0], times[-1]], [0,0])
    for ss in range(1,shell_num):
        plt.plot(times, pops[:,ss], "k--")
    ##################################################
    plt.xlabel(r"time ($J_\perp t$)")
    plt.ylabel("population")
    plt.tight_layout()

    plt.savefig(fig_dir + f"populations_{name_tag(coupling_zz)}.pdf")

print("completed")
