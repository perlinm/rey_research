#!/usr/bin/env python3

import numpy as np
import scipy, itertools
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

np.set_printoptions(linewidth = 200)
cutoff = 1e-10

# size and dimension of a periodic lattice
lattice_size = 10
lattice_dim = 2

spin_num = lattice_size**lattice_dim

# dimension of each spin, i.e. the `n` in SU(n)
spin_dim = 2
assert(spin_dim == 2) # TODO: generalize code in the future

# default exponent for power-law couplings
default_alpha = 3

# dimension of fully symmetric manifold
dim_PS = scipy.special.comb(spin_num + spin_dim - 1, spin_dim - 1, exact = True)

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

# collect a list of all spin pairs
spin_pairs = [ (pp,qq) for qq in range(spin_num) for pp in range(qq) ]
pair_index = { pair : idx for idx, pair in enumerate(spin_pairs) }

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

# define SU(n) interaction couplings between spins
def sunc_val(pp, qq, alpha = default_alpha):
    if to_int(pp) == to_int(qq): return 0
    return -1/dist(pp, qq)**alpha

# define quantities associated with the couplings
_sunc_mat = np.array([ [ sunc_val(pp,qq) for pp in range(spin_num) ]
                       for qq in range(spin_num) ])
_sunc_pair_vec = mat_to_pair_vec(_sunc_mat)
_sunc_disp_vec = mat_to_disp_vec(_sunc_mat)
_sunc_col_vec = _sunc_mat[0,:]
_sunc_tot = sum(_sunc_pair_vec)
sunc = { "mat" : _sunc_mat,
         "pair" : _sunc_pair_vec,
         "disp" : _sunc_disp_vec,
         "col" : _sunc_col_vec,
         "tot" : _sunc_tot }

# organize all displacements by their magnitude
dist_to_disps = {}
for dd in range(1,spin_num):
    dd_dist = dist(0, dd)
    try:
        dist_to_disps[dd_dist] += [ dd ]
    except:
        dist_to_disps[dd_dist] = [ dd ]

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
_eig_dist_vecs = _eig_dist_vecs.T
eig_vals -= sum(sunc["col"])
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

_eig_disp_vecs = _eig_dist_vecs @ _dist_to_disp.T
_eig_disp_vecs[abs(_eig_disp_vecs) < cutoff] = 0

# collect remaining quantities associated with two-body eigenstates
_eig_mats = np.array([ disp_vec_to_mat(vec) for vec in _eig_disp_vecs ])
_eig_pair_vecs = np.array([ mat_to_pair_vec(mat) for mat in _eig_mats ])
_eig_col_vecs = np.array([ sum(mat) for mat in _eig_mats ])
_eig_tots = np.array([ sum(vec) for vec in _eig_pair_vecs ])

_eig_pair_vecs[abs(_eig_pair_vecs) < cutoff] = 0
_eig_col_vecs[abs(_eig_col_vecs) < cutoff] = 0
_eig_tots[abs(_eig_tots) < cutoff] = 0
assert(np.all(_eig_tots[1:] == 0))

# decompose the SU(n) coupling matrix into matrices that generate two-body eigenstates
sunc_eig_coefs = np.array([ vec.conj() @ sunc["disp"] for vec in _eig_disp_vecs ])
sunc_eig_coefs[abs(sunc_eig_coefs) < cutoff] = 0

# collect all data in one place
eig = { shell : { "val" : eig_vals[shell],
                  "mat" : _eig_mats[shell],
                  "pair" : _eig_pair_vecs[shell],
                  "disp" : _eig_disp_vecs[shell],
                  "col" : _eig_col_vecs[shell],
                  "tot" : _eig_tots[shell] }
        for shell in range(len(eig_vals)) }

##########################################################################################

# compute diagram coefficients
def diagram_coefs(idx_1, idx_2):
    eig_1 = eig[idx_1]
    eig_2 = eig[idx_2]
    triplets = [ [ sunc, eig_1, eig_2 ],
                 [ eig_1, eig_2, sunc ],
                 [ eig_2, sunc, eig_1 ] ]

    if idx_1 == 0 and idx_2 == 0:
        D_0 = sum(eig_1["pair"]) * sum(eig_2["pair"]) * sum(sunc["pair"])

        D_1 = sum( uu["tot"] * vv["col"] @ ww["col"] for uu, vv, ww in triplets ) \
            - 2 * sum( eig_1["col"] * eig_2["col"] * sunc["col"] )

    else:
        D_0 = 0
        D_1 = 0

    D_2 = sum( ( uu["col"] @ vv["mat"] @ ww["col"]
               + uu["tot"] * vv["pair"] @ ww["pair"]
               - 2 * uu["col"] @ sum( vv["mat"] * ww["mat"] ) )
               for uu, vv, ww in triplets ) \
        + 4 * sum( eig_1["pair"] * eig_2["pair"] * sunc["pair"] )

    D_3 = spin_num * eig_1["mat"][0,:] @ eig_2["mat"] @ sunc["mat"][:,0]

    return D_0, D_1, D_2, D_3

# compute collective operator coefficients
def collective_coefs(idx_1, idx_2):
    D_0, D_1, D_2, D_3 = diagram_coefs(idx_1, idx_2)

    local_A_6 = D_0 - D_1 + D_2 - D_3
    local_A_4 = D_1 - 2 * D_2 + 3 * D_3
    local_A_2 = D_2 - 3 * D_3

    def _sup(nn):
        return np.product([ spin_num-jj for jj in range(nn) ])
    sup_A_6 = local_A_6 / _sup(6) if spin_num > 6 else 0
    sup_A_4 = local_A_4 / _sup(4) if spin_num > 4 else 0
    sup_A_2 = local_A_2 / _sup(2) if spin_num > 2 else 0

    A_6 = sup_A_6
    A_4 = sup_A_4 - 5*(3*spin_num-8) * sup_A_6
    A_2 = sup_A_2 - 2*(3*spin_num-4) * sup_A_4 \
        + (45*spin_num**2 - 210*spin_num + 184) * sup_A_6

    return A_6, A_4, A_2

shell_num = len(eig_vals)
shell_mat_6 = np.zeros((shell_num,shell_num))
shell_mat_4 = np.zeros((shell_num,shell_num))
shell_mat_2 = np.zeros((shell_num,shell_num))
for pp in range(shell_num):
    A_6, A_4, A_2 = collective_coefs(pp, pp)
    shell_mat_6[pp,pp] = A_6
    shell_mat_4[pp,pp] = A_4
    shell_mat_2[pp,pp] = A_2
for pp, qq in itertools.combinations(range(shell_num), 2):
    A_6, A_4, A_2 = collective_coefs(pp, qq)
    shell_mat_6[pp,qq] = shell_mat_6[qq,pp] = A_6
    shell_mat_4[pp,qq] = shell_mat_4[qq,pp] = A_4
    shell_mat_2[pp,qq] = shell_mat_2[qq,pp] = A_2

print(shell_mat_6)
print()
print(shell_mat_4)
print()
print(shell_mat_2)
print()
print(eig_vals)
print()

# define quantities associated with the two-body perturbation coefficients
epsilon = 0.05
