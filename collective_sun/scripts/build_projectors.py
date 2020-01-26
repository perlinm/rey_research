#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import functools, scipy, sympy

from tensorflow_extension import tf_outer_product

np.set_printoptions(linewidth = 200)
cutoff = 1e-10

# size and dimension of a periodic lattice
lattice_size = 16
lattice_dim = 1
spin_num = lattice_size**lattice_dim

##########################################################################################

# convert between integer and vector indices for a spin
def to_int(spin_vec):
    if type(spin_vec) == int:
        return spin_vec % spin_num
    return sum( lattice_size**pos * ( idx % lattice_size)
                for pos, idx in enumerate(spin_vec[::-1]) )
_idx_vec = { to_int(vec) : np.array(vec)
            for vec in np.ndindex((lattice_size,)*lattice_dim) }
def to_vec(spin_idx):
    if hasattr(spin_idx, "__getitem__"):
        return np.array(spin_idx)
    return _idx_vec[spin_idx % spin_num]

# qubit states and operators
up = np.array([1,0])
dn = np.array([0,1])
up_z, dn_z = up, dn
up_x = ( up + dn ) / np.sqrt(2.)
dn_x = ( up - dn ) / np.sqrt(2.)
up_y = ( up + 1j * dn ) / np.sqrt(2.)
dn_y = ( up - 1j * dn ) / np.sqrt(2.)

base_ops = {}
base_ops["I"] = np.outer(up_z,up_z) + np.outer(dn_z,dn_z)
base_ops["Z"] = np.outer(up_z,up_z.conj()) - np.outer(dn_z,dn_z.conj())
base_ops["X"] = np.outer(up_x,up_x.conj()) - np.outer(dn_x,dn_x.conj())
base_ops["Y"] = np.outer(up_y,up_y.conj()) - np.outer(dn_y,dn_y.conj())
base_ops["sz"] = base_ops["Z"] / 2
base_ops["sx"] = base_ops["X"] / 2
base_ops["sy"] = base_ops["Y"] / 2

# act with a multi-local operator `op` on the spin specified by `indices`
def spin_op(op, indices = None):
    # make `op` and `indices` correspond to a single spin operator,
    #   make them single-item lists
    if type(op) is str:
        op = [ op ]
    if type(indices) is int:
        indices = [ indices ]

    if type(op) is list:
        dtype = complex if "Y" in op or "sy" in op else float
    else:
        dtype = op.dtype

    # convert an array into a sparse tensor
    def _to_sparse_tensor(array):
        if type(array) is tf.sparse.SparseTensor: return array
        indices = np.array(list(np.ndindex(array.shape)))
        return tf.SparseTensor(indices = indices[array.flatten()!=0],
                               values = array[array!=0].flatten().astype(dtype),
                               dense_shape = array.shape)

    # convert a list of strings into a sparse tensor in which the tensor factors
    #   associated with each qubit are collected together and flattened
    if type(op) is list:
        op_spins = len(op)
        op = functools.reduce(tf_outer_product,
                              [ _to_sparse_tensor(base_ops[tag].flatten()) for tag in op ])
    else:
        op_spins = int(np.log2(np.prod(op.shape)) + 1/2) // 2
        op = tf.sparse.reshape(_to_sparse_tensor(op), (2,2,)*op_spins)

        fst_half = range(op_spins)
        snd_half = range(op_spins,2*op_spins)
        perm = np.array(list(zip(list(fst_half),list(snd_half)))).flatten()
        op = tf.sparse.transpose(op, perm)
        op = tf.sparse.reshape(op, (4,)*op_spins)

    if indices is None:
        # if we were not given indices, just return the multi-local operator
        indices = np.arange(op_spins)
        total_spins = op_spins
    else:
        # otherwise, return an operator that acts on the indexed spins
        indices = [ to_int(idx) for idx in indices ]
        total_spins = spin_num
        for _ in range(total_spins - op_spins):
            op = tf_outer_product(op, _to_sparse_tensor(base_ops["I"].flatten()))

    # rearrange tensor factors according to the desired qubit order
    old_order = indices + [ jj for jj in range(total_spins) if jj not in indices ]
    new_order = np.arange(total_spins)[np.argsort(old_order)]
    op = tf.sparse.transpose(op, new_order)
    op = tf.sparse.reshape(op, (2,2,)*total_spins)

    # un-flatten the tensor factors, and flatten the tensor into
    #   a matrix that acts on the joint Hilbert space of all qubits
    evens = range(0,2*total_spins,2)
    odds = range(1,2*total_spins,2)
    op = tf.sparse.transpose(op, list(evens)+list(odds))
    op = tf.sparse.reshape(op, (2**total_spins,2**total_spins))

    # convert matrix into scipy's sparse matrix format
    return scipy.sparse.csr_matrix((op.values.numpy(), op.indices.numpy().T))

def sym_state(mm):
    assert(type(mm) is int and mm >= 0 and mm <= spin_num)
    states = [ 1 ] * mm + [ 0 ] * (spin_num-mm)
    def to_vec(state): return dn if state == 0 else up
    vec = sum( functools.reduce(np.kron, map(to_vec, joint_state))
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

def chop(op):
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
        kk = to_vec(kk) * 2*np.pi / lattice_size

        op = sum( np.exp(1j * to_vec(pp) @ kk) * spin_op("Z", pp)
                  for pp in range(spin_num) )
        op = chop(op)

        state = op @ this_sym_state
        projs[1] += ( state @ state.conj().T ) / ( state.conj().T @ state )[0,0]

    # add to projector onto double-spin-wave manifold
    for val, vec in zip(vals, vecs.T):
        op = sum( weight * spin_op(["Z","Z"], spin_pairs[pp_qq])
                  for pp_qq, weight in enumerate(vec) )
        op = chop(op)

        state = op @ this_sym_state
        if state.count_nonzero() == 0: continue
        projs[2] += ( state @ state.T ) / ( state.T @ state )[0,0]

    print("    ",time()-start)

projs[1] = projs[1].real

for mm, proj in projs.items():
    print(f"writing projector {mm}")
    with open(f"projector_N{spin_num}_M{mm}.txt", "w") as f:
        f.write("# projector onto collective shell number {mm} of {spin_num} spins\n")
        f.write("# represented by a matrix in the standard basis of {spin_num} qubits\n")
        f.write("# row, column, value\n")
        for idx, val in proj.todok().items():
            f.write(f"{idx[0]}, {idx[1]}, {val}\n")

exit()
##########################################################################################
print("verifying projectors")

def col_op(op):
    return sum( spin_op(op,idx) for idx in range(spin_num) )
Sz = col_op("sz")
Sx = col_op("sx")
Sy = col_op("sy")
SS = ( Sx @ Sx + Sy @ Sy + Sz @ Sz ).real

net_SS = np.linalg.eigvalsh(SS.todense())
net_S = np.sqrt(net_SS + 1/4) - 1/2
for idx, val in enumerate(net_S):
    if abs( 2*val - np.round(2*val) ) < cutoff:
        net_S[idx] = np.round(2*val)/2
print()
print(set(net_S))
print()

iden = scipy.sparse.identity(2**spin_num)

for mm, proj in projs.items():
    net_SS = ( SS @ proj ).diagonal().sum() / proj.diagonal().sum()
    net_S = np.sqrt(net_SS + 1/4) - 1/2
    manifold = int(spin_num/2 - net_S + 1/2)
    print(mm, manifold)

    SS -= net_SS * proj

net_SS = np.linalg.eigvalsh(SS.todense())
net_S = np.sqrt(net_SS + 1/4) - 1/2
for idx, val in enumerate(net_S):
    if abs( 2*val - np.round(2*val) ) < cutoff:
        net_S[idx] = np.round(2*val)/2
print()
print(set(net_S))
