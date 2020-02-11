#!/usr/bin/env python3

import numpy as np
import itertools, functools
import matplotlib.pyplot as plt

from dicke_methods import coherent_spin_state as coherent_state_PS

np.set_printoptions(linewidth = 200)
cutoff = 1e-10

# size and dimension of a periodic lattice
lattice_size = 3
lattice_dim = 2

# default exponent for power-law couplings
default_alpha = 3

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

# identify number of spins and collect a list of all spin pairs
spin_num = lattice_size**lattice_dim
spin_pairs = [ (pp,qq) for qq in range(spin_num) for pp in range(qq) ]
pair_index = { pair : idx for idx, pair in enumerate(spin_pairs) }

# convert between integer and vector indices for a spin
def to_int(spin_vec):
    if type(spin_vec) == int:
        return spin_vec % spin_num
    return sum( lattice_size**pos * ( idx % lattice_size )
                for pos, idx in enumerate(spin_vec[::-1]) )
_idx_vec = { to_int(vec) : np.array(vec)
            for vec in np.ndindex((lattice_size,)*lattice_dim) }
def to_vec(spin_idx):
    if hasattr(spin_idx, "__getitem__"):
        return np.array(spin_idx)
    return _idx_vec[spin_idx % spin_num]

# distance between two spins
def dist_1D(pp,qq):
    diff = ( pp - qq ) % lattice_size
    return min(diff, lattice_size - diff)
def dist(pp,qq):
    if type(pp) is int: pp = to_vec(pp)
    if type(qq) is int: qq = to_vec(qq)
    return np.sqrt(sum( dist_1D(pp_jj,qq_jj)**2 for pp_jj, qq_jj in zip(pp, qq) ))

# define cycle matrices
def cycle_mat(dd, kk = 0):
    dd_vec = to_vec(dd)
    def _shift(pp):
        return to_int( dd_vec + to_vec(pp) )

    if kk == 0:
        def _val(_): return 1
    else:
        kk_vec = to_vec(kk) * 2*pi/lattice_size
        def _val(pp):
            return np.exp(1j * to_vec(pp) @ kk_vec)

    mat = np.zeros((spin_num,)*2)
    for pp in range(spin_num):
        mat[_shift(pp),pp] = _val(pp)
    return mat

# recover a matrix from a vector of cycle (displacement) matrix coefficients
def disp_vec_to_mat(disp_vec, kk = 0):
    ignore_0 = len(disp_vec) == spin_num-1
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
# collect interaction matrix data
##########################################################################################

# define SU(n) interaction couplings between spins
def sunc_val(pp, qq, alpha = default_alpha):
    if to_int(pp) == to_int(qq): return 0
    return -1/dist(pp, qq)**alpha

# define quantities associated with the couplings
_sunc_mat = np.array([ [ sunc_val(pp,qq) for pp in range(spin_num) ]
                       for qq in range(spin_num) ])
_sunc_pair_vec = mat_to_pair_vec(_sunc_mat)
_sunc_disp_vec = mat_to_disp_vec(_sunc_mat)
_sunc_col_vec = sum(_sunc_mat)
_sunc_tot = sum(_sunc_pair_vec)
sunc = { "disp" : _sunc_disp_vec,
         "pair" : _sunc_pair_vec,
         "mat" : _sunc_mat,
         "col" : _sunc_col_vec,
         "tot" : _sunc_tot }

##########################################################################################
# decompose interaction matrix into generators of SU(n)-symmetric interaction eigenstates
##########################################################################################

# organize all displacements by their magnitude
dist_to_disps = {}
for dd in range(1,spin_num):
    dd_dist = dist(0, dd)
    try:
        dist_to_disps[dd_dist] += [ dd ]
    except:
        dist_to_disps[dd_dist] = [ dd ]
shell_num = len(dist_to_disps)

# solve the translationally invariant, isotropic two-body eigenvalue problem
characteristic_matrix = np.zeros((len(dist_to_disps),)*2)
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

# convert "distance" vectors to "displacement" vectors
def unit_vec(dim, idx):
    vec = np.zeros(dim, dtype = int)
    vec[idx] = 1
    return vec
def dist_vec(dist):
    vec = sum( unit_vec(spin_num-1,to_int(disp)-1) for disp in dist_to_disps[dist] )
    return vec / np.sqrt( len(dist_to_disps[dist]) )
_dist_to_disp = np.vstack([ dist_vec(dist) for dist in dist_to_disps.keys() ]).T
def dist_to_disp_vec(dist_vec):
    return _sym_to_disp @ dist_vec

_eig_disp_vecs = _dist_to_disp @ _eig_dist_vecs
_eig_disp_vecs[abs(_eig_disp_vecs) < cutoff] = 0

# collect remaining quantities associated with two-body eigenstates
_eig_mats = np.array([ disp_vec_to_mat(vec) for vec in _eig_disp_vecs.T ])
_eig_pair_vecs = np.array([ mat_to_pair_vec(mat) for mat in _eig_mats ]).T
_eig_col_vecs = np.array([ mat[:,0] for mat in _eig_mats ]).T
_eig_tots = np.array([ sum(vec) for vec in _eig_pair_vecs.T ])

_eig_pair_vecs[abs(_eig_pair_vecs) < cutoff] = 0
_eig_col_vecs[abs(_eig_col_vecs) < cutoff] = 0
_eig_tots[abs(_eig_tots) < cutoff] = 0
assert(np.all(_eig_tots[1:] == 0))

# collect all two-body problem data in one place
eigs = { shell : { "val" : eig_vals[shell],
                   "disp" : _eig_disp_vecs[:,shell],
                   "pair" : _eig_pair_vecs[:,shell],
                   "mat" : _eig_mats[shell],
                   "col" : _eig_col_vecs[:,shell],
                   "tot" : _eig_tots[shell] }
         for shell in range(shell_num) }

# convert vector into projector
def to_proj(vec):
    return np.outer( vec.conj(), vec ) / ( vec.conj() @ vec )

# decompose couplings in the basis of coefficients that generate eigenstates
for shell in range(shell_num):
    sunc[shell] = {}
    sunc[shell]["disp"] = to_proj( eigs[shell]["disp"] ) @ sunc["disp"]
    sunc[shell]["pair"] = to_proj( eigs[shell]["pair"] ) @ sunc["pair"]
    sunc[shell]["mat"] = disp_vec_to_mat(sunc[shell]["disp"])
    sunc[shell]["col"] = sum(sunc[shell]["mat"])
    sunc[shell]["tot"] = sum(sunc[shell]["pair"])

# compute norms of eigenstates generated by the action of ZZ interactions
#   on states with definite spin projection onto the z axis
def _ff(nn, kk): # falling factorial
    return np.prod([ nn-jj for jj in range(kk) ])
def _norm(spin_proj, shell):
    tot_sqr = sunc[shell]["tot"]**2
    col_sqr = sunc[shell]["col"] @ sunc[shell]["col"]
    pair_sqr = sunc[shell]["pair"] @ sunc[shell]["pair"]
    c_4 = ( tot_sqr - col_sqr + pair_sqr ) / _ff(spin_num, 4)
    c_2 = ( col_sqr - 2 * pair_sqr ) / _ff(spin_num, 2) - 2*(3*spin_num-4) * c_4
    c_0 = 3*tot_sqr - spin_num*col_sqr + spin_num*(spin_num-2)*pair_sqr
    c_0 /= (spin_num-1) * (spin_num-3)
    norm_sqr = c_4 * (2*spin_proj)**4 + c_2 * (2*spin_proj)**2 + c_0
    if np.allclose(norm_sqr,0):
        return 0
    else:
        return np.sqrt(norm_sqr)

for shell in range(shell_num):
    sunc[shell]["norm"] = functools.partial(_norm, shell = shell)

##########################################################################################
# compute couplings induced between shells by ZZ interactions
##########################################################################################

# compute diagram coefficients
def diagram_coefs(shell_1, shell_2):
    sunc_1 = sunc[shell_1]
    sunc_2 = sunc[shell_2]
    triplets = [ [ sunc, sunc_1, sunc_2 ],
                 [ sunc_1, sunc_2, sunc ],
                 [ sunc_2, sunc, sunc_1 ] ]

    if shell_1 == 0 and shell_2 == 0:
        D_0 = sunc_1["tot"] * sunc_2["tot"] * sunc["tot"]

        D_1 = sum( uu["tot"] * vv["col"] @ ww["col"] for uu, vv, ww in triplets ) \
            - 2 * sum( sunc_1["col"] * sunc_2["col"] * sunc["col"] )

    else:
        D_0 = 0
        D_1 = 0

    D_2 = sum( ( uu["col"] @ vv["mat"] @ ww["col"]
               + uu["tot"] * vv["pair"] @ ww["pair"]
               - 2 * uu["col"] @ sum( vv["mat"] * ww["mat"] ) )
               for uu, vv, ww in triplets ) \
        + 4 * sum( sunc_1["pair"] * sunc_2["pair"] * sunc["pair"] )

    D_3 = spin_num * sunc_1["mat"][0,:] @ sunc["mat"] @ sunc_2["mat"][:,0]

    return D_0, D_1, D_2, D_3

# compute multi-local operator coefficients
def local_coefs(shell_1, shell_2):
    D_0, D_1, D_2, D_3 = diagram_coefs(shell_1, shell_2)

    A_6 = D_0 - D_1 + D_2 - D_3
    A_4 = D_1 - 2 * D_2 + 3 * D_3
    A_2 = D_2 - 3 * D_3
    A_0 = D_3

    return A_6, A_4, A_2, A_0

# compute collective operator coefficients
def collective_coefs(shell_1, shell_2):
    local_A_6, local_A_4, local_A_2, local_A_0 = local_coefs(shell_1, shell_2)

    def _ff(kk):
        return np.product([ spin_num-jj for jj in range(kk) ])
    sup_A_6 = local_A_6 / _ff(6) if spin_num >= 6 else 0
    sup_A_4 = local_A_4 / _ff(4) if spin_num >= 4 else 0
    sup_A_2 = local_A_2 / _ff(2) if spin_num >= 2 else 0
    sup_A_0 = local_A_0

    A_6 = sup_A_6
    A_4 = sup_A_4 \
        - 5*(3*spin_num-8) * sup_A_6
    A_2 = sup_A_2 \
        - 2*(3*spin_num-4) * sup_A_4 \
        + (45*spin_num**2 - 210*spin_num + 184) * sup_A_6
    A_0 = local_A_0 \
        - local_A_2 / (spin_num-1) \
        + local_A_4 * 3 / ( (spin_num-1) * (spin_num-3) ) \
        - local_A_6 * 15 / ( (spin_num-1) * (spin_num-3) * (spin_num-5) )

    return A_6, A_4, A_2, A_0

# collect all collective operator coefficients
shell_num = len(eig_vals)
shell_coefs_6 = np.zeros((shell_num,shell_num))
shell_coefs_4 = np.zeros((shell_num,shell_num))
shell_coefs_2 = np.zeros((shell_num,shell_num))
shell_coefs_0 = np.zeros((shell_num,shell_num))
for pp in range(shell_num):
    A_6, A_4, A_2, A_0 = collective_coefs(pp, pp)
    shell_coefs_6[pp,pp] = A_6
    shell_coefs_4[pp,pp] = A_4
    shell_coefs_2[pp,pp] = A_2
    shell_coefs_0[pp,pp] = A_0
for pp, qq in itertools.combinations(range(shell_num), 2):
    A_6, A_4, A_2, A_0 = collective_coefs(pp, qq)
    shell_coefs_6[pp,qq] = shell_coefs_6[qq,pp] = A_6
    shell_coefs_4[pp,qq] = shell_coefs_4[qq,pp] = A_4
    shell_coefs_2[pp,qq] = shell_coefs_2[qq,pp] = A_2
    shell_coefs_0[pp,qq] = shell_coefs_0[qq,pp] = A_0

# construct the Hamiltonian induced by ZZ interactions
#   for a fixed spin projection onto the Z axis
def _shell_mat(spin_proj):
    bare_shell_mat = shell_coefs_6 * (2*spin_proj)**6 \
                   + shell_coefs_4 * (2*spin_proj)**4 \
                   + shell_coefs_2 * (2*spin_proj)**2 \
                   + shell_coefs_0
    inv_norm_vec = np.array([ ( lambda val : 1/val if val != 0 else 0 )
                              ( sunc[shell]["norm"](spin_proj) )
                              for shell in range(shell_num) ])
    return bare_shell_mat * np.outer(inv_norm_vec, inv_norm_vec)

# construct the net Hamiltonian induced by SU(n) + ZZ interactions
def _hamiltonian(zz_sun_ratio, spins_up):
    return np.diag(eig_vals) + zz_sun_ratio * _shell_mat(spins_up)

# energies and energy eigenstates within each sector of fixed spin projection
def energies_states(zz_sun_ratio):
    energies = np.zeros((spin_num+1,shell_num))
    eig_states = np.zeros((spin_num+1,shell_num,shell_num))

    for spins_up in range(spin_num+1):
        spin_proj = spins_up - spin_num/2
        energies[spins_up,:], eig_states[spins_up,:,:] \
            = np.linalg.eigh(_hamiltonian(zz_sun_ratio, spin_proj))

        spins_dn = spin_num - spins_up
        if spins_dn == spins_up: break
        energies[spins_dn,:], eig_states[spins_dn,:,:] \
            = energies[spins_up,:], eig_states[spins_up,:,:]

    return energies, eig_states

##########################################################################################
# simulate!
##########################################################################################

# coherent spin state
def coherent_spin_state(vec):
    zero_shell = np.zeros(shell_num)
    zero_shell[0] = 1
    return np.outer(coherent_state_PS(vec,spin_num), zero_shell)

# note: extra factor of 1/2 in che_eff_bare for compatibility with Chunlei's work
chi_eff_bare = 1/4 * np.mean(sunc["pair"])
state_X = coherent_spin_state([0,1,0])

def _states(initial_state, zz_sun_ratio, times):
    energies, eig_states = energies_states(zz_sun_ratio)
    init_state_eig = np.einsum("zsS,zS->zs", eig_states, initial_state)
    phases = np.exp(-1j * np.tensordot(times, energies, axes = 0))
    return np.einsum("zSs,tzS->tzs", eig_states, phases * init_state_eig[None,:,:])

def simulate(coupling_zz, max_tau = 2, overshoot_ratio = 1.5, points = 500):
    zz_sun_ratio = coupling_zz - 1

    if zz_sun_ratio != 0:
        chi_eff = zz_sun_ratio * chi_eff_bare
        sim_time = min(max_time, max_tau * spin_num**(-2/3) / chi_eff)
    else:
        sim_time = max_time

    sim_time = 2

    times = np.linspace(0, sim_time, points)

    # note: factor of 1/2 included for compatibility with Chunlei's work
    states = _states(state_X, zz_sun_ratio/2, times)

    pops = np.einsum("tzs->ts", abs(states)**2)

    return times, pops

for coupling_zz in inspect_coupling_zz:
    plt.plot(*simulate(coupling_zz))
    plt.xlabel(r"time ($J_\perp t$)")
    plt.ylabel("population")
    plt.tight_layout()
    plt.show()
