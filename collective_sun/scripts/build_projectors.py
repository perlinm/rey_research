#!/usr/bin/env python3

import numpy as np
import functools, scipy, sympy

np.set_printoptions(linewidth = 200)
cutoff = 1e-10

spin_num = 16
data_dir = "./projectors/"

##########################################################################################

def _bit_sign(bit):
    return 1-int(bit)*2

def _get_bit(num, pos):
    return ( num >> pos ) & 1

def Z_op(pp):
    pp %= spin_num
    data = [ _bit_sign(_get_bit(idx,pp)) for idx in range(2**spin_num) ]
    return scipy.sparse.dia_matrix(([data], [0]), shape = (2**spin_num,)*2)

def ZZ_op(pp,qq):
    pp %= spin_num
    qq %= spin_num
    data = [ _bit_sign(_get_bit(idx,pp) == _get_bit(idx,qq))
             for idx in range(2**spin_num) ]
    return scipy.sparse.dia_matrix(([data], [0]), shape = (2**spin_num,)*2)

def sym_state(mm):
    assert(type(mm) is int and mm >= 0 and mm <= spin_num)
    states = [ 1 ] * mm + [ 0 ] * (spin_num-mm)
    def _to_vec(state): return np.array( [1,0] if state == 0 else [0,1] )
    vec = sum( functools.reduce(np.kron, map(_to_vec, joint_state))
               for joint_state in sympy.utilities.iterables.multiset_permutations(states) )
    vec = vec / np.sqrt( vec @ vec )
    return scipy.sparse.csr_matrix(vec).T

##########################################################################################
print("solving the two-body eigenvalue problem")

spin_pairs = [ (pp,qq) for qq in range(spin_num) for pp in range(qq) ]
pair_index = { frozenset(pair) : idx for idx, pair in enumerate(spin_pairs) }

# set up the general two-body eigenvalue problem
mat = np.zeros((len(spin_pairs),)*2)
for pp_qq_idx, pp_qq in enumerate(spin_pairs):
    pp, qq = pp_qq
    mat[pp_qq_idx,pp_qq_idx] = 2 * ( spin_num - 2 )
    for kk in range(spin_num):
        if kk == pp or kk == qq: continue
        pp_kk_idx = pair_index[frozenset((pp,kk))]
        qq_kk_idx = pair_index[frozenset((qq,kk))]
        mat[pp_qq_idx,pp_kk_idx] = -1
        mat[pp_qq_idx,qq_kk_idx] = -1

vals, vecs = np.linalg.eigh(mat)
vals = np.round(vals).astype(int)
vecs[abs(vecs) < cutoff] = 0

energy_0 = 0
energy_1 = spin_num
energy_2 = 2*(spin_num-1)
vecs = vecs[:, vals == energy_2 ]
vals = vals[ vals == energy_2 ]

##########################################################################################
print("building projectors")

projs = { 0 : scipy.sparse.csr_matrix((2**spin_num,)*2),
          1 : scipy.sparse.csr_matrix((2**spin_num,)*2, dtype = complex),
          2 : scipy.sparse.csr_matrix((2**spin_num,)*2) }

from time import time

def sparse_chop(op):
    for idx, data in enumerate(op.data):
        if abs(data) < cutoff:
            op.data[idx] = 0
    op.eliminate_zeros()
    return op

for net_up in range(spin_num+1):
    print(f" {net_up}/{spin_num+1}")

    start = time()

    # add to projector onto fully symmetric manifold
    this_sym_state = sym_state(net_up)
    projs[0] += this_sym_state @ this_sym_state.T

    if net_up == 0 or net_up == spin_num:
        continue

    # add to projector onto spin-wave manifold
    for kk in range(1,spin_num):
        op = sum( np.exp(1j * pp * kk * 2*np.pi / spin_num) * Z_op(pp)
                  for pp in range(spin_num) )
        op = sparse_chop(op)

        state = op @ this_sym_state
        projs[1] += ( state @ state.conj().T ) / ( state.conj().T @ state )[0,0]

    # add to projector onto double-spin-wave manifold
    for val, vec in zip(vals, vecs.T):
        op = sum( weight * ZZ_op(*spin_pairs[pp_qq])
                  for pp_qq, weight in enumerate(vec) )
        op = sparse_chop(op)

        state = op @ this_sym_state
        if state.count_nonzero() == 0: continue
        projs[2] += ( state @ state.T ) / ( state.T @ state )[0,0]

    print("    ",time()-start)

projs[1] = projs[1].real

for mm, proj in projs.items():
    print(f"writing projector {mm}")
    with open(data_dir + f"projector_N{spin_num}_M{mm}.txt", "w") as f:
        f.write("# projector onto collective shell number {mm} of {spin_num} spins\n")
        f.write("# represented by a matrix in the standard basis of {spin_num} qubits\n")
        f.write("# row, column, value\n")
        for idx, val in proj.todok().items():
            if abs(val) < cutoff: continue
            f.write(f"{idx[0]}, {idx[1]}, {val}\n")
