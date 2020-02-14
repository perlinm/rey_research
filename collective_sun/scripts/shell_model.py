#!/usr/bin/env python3

import numpy as np
import itertools, functools
import matplotlib.pyplot as plt

from dicke_methods import coherent_spin_state as coherent_state_PS

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
sunc = { "disp" : _sunc_disp_vec,
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

# collect remaining quantities associated with two-body eigenstates
_eig_mats = np.array([ disp_vec_to_mat(vec) for vec in _eig_disp_vecs.T ])
_eig_pair_vecs = np.array([ mat_to_pair_vec(mat) for mat in _eig_mats ]).T

_eig_mats[abs(_eig_mats) < cutoff] = 0
_eig_pair_vecs[abs(_eig_pair_vecs) < cutoff] = 0

# collect all two-body problem data in one place
eigs = { shell : { "val" : eig_vals[shell],
                   "disp" : _eig_disp_vecs[:,shell],
                   "pair" : _eig_pair_vecs[:,shell],
                   "mat" : _eig_mats[shell] }
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

##########################################################################################
# compute operators in the spin projection/shell basis
##########################################################################################

# falling factorial
def _ff(nn, kk):
    return np.prod([ nn-jj for jj in range(kk) ])

# diagram coefficients that appear in the triple-ZZ product
def diagram_coefs(shell_1, shell_2):
    sunc_1 = sunc[shell_1]
    sunc_2 = sunc[shell_2]

    triplets = [ [ sunc_1, sunc, sunc_2 ],
                 [ sunc, sunc_2, sunc_1 ],
                 [ sunc_2, sunc_1, sunc ] ]

    D_0 = sunc_1["tot"] * sunc_2["tot"] * sunc["tot"]

    D_1 = sum( uu["tot"] * vv["col"] @ ww["col"] for uu, vv, ww in triplets ) \
        - 2 * sum( sunc_1["col"] * sunc_2["col"] * sunc["col"] )

    D_2 = sum( ( uu["col"] @ vv["mat"] @ ww["col"]
                 + uu["tot"] * vv["pair"] @ ww["pair"]
                 - 2 * uu["col"] @ sum( vv["mat"] * ww["mat"] ) )
               for uu, vv, ww in triplets ) \
        + 4 * sum( sunc_1["pair"] * sunc_2["pair"] * sunc["pair"] )

    D_3 = spin_num * sunc_1["mat"][0,:] @ sunc["mat"] @ sunc_2["mat"][:,0]

    return D_0, D_1, D_2, D_3

# coefficients of the triple-ZZ product
def triple_product_coefs(shell_1, shell_2):
    D_0, D_1, D_2, D_3 = diagram_coefs(shell_1, shell_2)

    # coefficients in the multi-local operator expansion
    lA_6 = D_0 - D_1 + D_2 - D_3
    lA_4 = D_1 - 2 * D_2 + 3 * D_3
    lA_2 = D_2 - 3 * D_3
    lA_0 = D_3

    A_6 = A_4 = A_2 = 0
    A_0 = lA_0
    if spin_num >= 2:
        sA_2 = lA_2 / _ff(spin_num, 2)
        A_2 += sA_2
        A_0 -= lA_2 / (spin_num-1)
    if spin_num >= 4:
        sA_4 = lA_4 / _ff(spin_num, 4)
        A_4 += sA_4
        A_2 -= sA_4 * 2*(3*spin_num-4)
        A_0 += lA_4 * 3 / ( (spin_num-1) * (spin_num-3) )
    if spin_num >= 6:
        sA_6 = lA_6 / _ff(spin_num, 6)
        A_6 += sA_6
        A_4 -= sA_6 * 5*(3*spin_num-8)
        A_2 += sA_6 * (45*spin_num**2 - 210*spin_num + 184)
        A_0 -= lA_6 * 15 / ( (spin_num-1) * (spin_num-3) * (spin_num-5) )

    return A_6, A_4, A_2, A_0

# coefficients of the double-ZZ product
def double_product_coefs(shell):
    pair_sqr = sum(sunc[shell]["pair"]**2)

    # coefficients in the multi-local operator expansion
    lB_4 = pair_sqr
    lB_2 = -2*pair_sqr
    lB0 = pair_sqr

    if shell == 0:
        tot_sqr = sunc[shell]["tot"]**2
        col_sqr = sum(sunc[shell]["col"]**2)
        lB_4 += tot_sqr - col_sqr
        lB_2 += col_sqr

    B_4 = B_2 = 0
    B_0 = pair_sqr
    if spin_num >= 2:
        B_2 += lB_2 / _ff(spin_num, 2)
        B_0 -= lB_2 / (spin_num-1)
    if spin_num >= 4:
        B_4 += lB_4 / _ff(spin_num, 4)
        B_2 -= 2*(3*spin_num-4) * B_4
        B_0 += lB_4 * 3 / ( (spin_num-1) * (spin_num-3) )

    return B_4, B_2, B_0

# collect all operator product coefficients
prod_coefs = { 3 : {}, 2 : {}, 1 : {} }
for op_num in prod_coefs.keys():
    for power in range(0,2*op_num+1,2):
        prod_coefs[op_num][power] = np.zeros((shell_num,)*(op_num-1))

prod_coefs[1][2] = sum(sunc["pair"]) / ( spin_num*(spin_num-1) )
prod_coefs[1][0] = -spin_num * prod_coefs[1][2]

for ss in range(shell_num):
    B_4, B_2, B_0 = double_product_coefs(ss)
    prod_coefs[2][4][ss] = B_4
    prod_coefs[2][2][ss] = B_2
    prod_coefs[2][0][ss] = B_0

    A_6, A_4, A_2, A_0 = triple_product_coefs(ss, ss)
    prod_coefs[3][6][ss,ss] = A_6
    prod_coefs[3][4][ss,ss] = A_4
    prod_coefs[3][2][ss,ss] = A_2
    prod_coefs[3][0][ss,ss] = A_0

for rr, ss in itertools.combinations(range(shell_num), 2):
    A_6, A_4, A_2, A_0 = triple_product_coefs(rr, ss)
    prod_coefs[3][6][rr,ss] = prod_coefs[3][6][ss,rr] = A_6
    prod_coefs[3][4][rr,ss] = prod_coefs[3][4][ss,rr] = A_4
    prod_coefs[3][2][rr,ss] = prod_coefs[3][2][ss,rr] = A_2
    prod_coefs[3][0][rr,ss] = prod_coefs[3][0][ss,rr] = A_0

# expectation value of the `num`-ZZ product with respect to
#   a permutationally symmetric state with definite spin difference
def prod_val(num, spin_diff):
    return sum( prod_coefs[num][power] * spin_diff**power
                for power in prod_coefs[num].keys() )

##########################################################################################
##########################################################################################
##########################################################################################

from operator_product_methods import reduced_diagrams, operator_contractions, \
    project_product, evaluate_multi_local_op

Z1_op = np.array([[1,0],[0,-1]])
Z2_op = np.kron(Z1_op,Z1_op).reshape((2,)*4)

Z2_diags = { num : reduced_diagrams([2]*num)
             for num in [ 1, 2, 3 ] }
Z2_opers = { num : operator_contractions([Z2_op]*num)
                 for num in [ 1, 2, 3 ] }

def product(num, *args):
    if num == 3:
        shell_lft, shell_rht = args
        couplings = [ sunc[shell_lft]["mat"], sunc["mat"], sunc[shell_rht]["mat"] ]
    if num == 2:
        shell = args[0]
        couplings = [ sunc[shell]["mat"], sunc["mat"] ]
    if num == 1:
        couplings = [ sunc["mat"] ]

    return project_product(couplings, Z2_diags[num], Z2_opers[num])

def couplings(shell_lft, shell_rht):
    return sunc[shell_lft]["mat"], sunc["mat"], sunc[shell_rht]["mat"]

prod_vecs = {}
prod_vecs[3] = { ( shell_lft, shell_rht ) : product(3, shell_lft, shell_rht)
                 for shell_lft in range(shell_num)
                 for shell_rht in range(shell_lft+1) }
prod_vecs[2] = { shell : product(2, shell) for shell in range(shell_num) }
prod_vecs[1] = product(1)

# expectation value of the `num`-ZZ operator product with respect to
#   a permutationally symmetric state with definite spin difference
def prod_val(num, spin_diff):
    spins_up = ( spin_diff + spin_num ) // 2
    spins_dn = spin_num - spins_up
    populations = ( spins_up, spins_dn )
    def _val(ops):
        return sum( evaluate_multi_local_op(op, populations)
                    for op in ops )
    if num == 1:
        return _val(prod_vecs[num])

    if num == 2:
        return np.array([ _val(prod_vecs[num][shell])
                          for shell in range(shell_num) ])

    if num == 3:
        prod_val = np.zeros((shell_num,shell_num))
        for shell_lft in range(shell_num):
            for shell_rht in range(shell_lft+1):
                prod_val[shell_lft,shell_rht] = _val(prod_vecs[num][shell_lft,shell_rht])
                if shell_rht != shell_lft:
                    prod_val[shell_rht,shell_lft] = prod_val[shell_lft,shell_rht]
        return prod_val

##########################################################################################
##########################################################################################
##########################################################################################

# construct the Hamiltonian induced by ZZ interactions
#   for a fixed spin projection onto the Z axis
def _shell_mat(spin_diff):
    mat = np.zeros((shell_num,shell_num))
    if abs(spin_diff) < spin_num-2:
        norms = ( lambda vec : np.outer(vec,vec) )( prod_val(2, spin_diff)[1:] )
        mat[1:,1:] = prod_val(3, spin_diff)[1:,1:] / np.sqrt(norms)
        mat[0,1:] = mat[1:,0] = np.sqrt(abs(prod_val(2, spin_diff)[1:]))
    mat[0,0] = prod_val(1, spin_diff)
    return mat

# construct the net Hamiltonian induced by SU(n) + ZZ interactions
def _hamiltonian(zz_sun_ratio, spin_diff):
    return np.diag(eig_vals) + zz_sun_ratio * _shell_mat(spin_diff)

# energies and energy eigenstates within each sector of fixed spin projection
def energies_states(zz_sun_ratio):
    energies = np.zeros((spin_num+1,shell_num), dtype = complex)
    eig_states = np.zeros((spin_num+1,shell_num,shell_num), dtype = complex)

    for spins_dn in range(spin_num+1):
        spins_up = spin_num - spins_dn
        spin_diff = spins_up - spins_dn

        energies[spins_up,:], eig_states[spins_up,:,:] \
            = np.linalg.eigh(_hamiltonian(zz_sun_ratio, spin_diff))

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
