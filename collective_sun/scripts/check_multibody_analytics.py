#!/usr/bin/env python3

import numpy as np
import itertools as it
import functools, operator

from itertools_extension import assignments, set_diagrams
from multibody_methods import unit_tensor, random_tensor, multibody_problem, \
    sym_state, embed_operator

trans_inv = False
isotropic = False

min_spins = 4
max_spins = 5
max_dim = 2**max_spins

total_dim = max_dim+1 # total dimension of Hilbert space
while total_dim > max_dim:
    spin_dim = np.random.randint(2,5) # dimension of each spin
    lattice_shape = ( np.random.randint(1,max_spins),
                      np.random.randint(1,max_spins) )
    spin_num = np.product(lattice_shape)
    if spin_num > max_spins or spin_num < min_spins: continue
    total_dim = spin_dim**spin_num

op_num = np.random.randint(2,5) # total number of multi-body operators

# the dimension of each multi-body operator
# an operator with dimension M is an M-local operator
dimensions = [ np.random.randint(1,spin_num) for _ in range(op_num) ]

print("dimension of each spin:", spin_dim)
print("lattice shape:", lattice_shape)
print("total number of spins:", spin_num)
print("multi-body operator dimensions", dimensions)

def check_equal(xx, yy):
    equal = np.allclose(xx, yy)
    print(equal, end = "")
    if equal: print()
    else: print(" ", abs(xx-yy).max())

##########################################################################################
# construct projector onto the fully symmetric manifold
##########################################################################################

# labels for all fully symmetric states
sym_labels = list(assignments(spin_num,spin_dim))

# projector onto the fully symmetric manifold
def to_proj(state): return np.outer(state, state.conj())
sym_proj = sum( to_proj(sym_state(label,spin_num,spin_dim)) for label in sym_labels )

##########################################################################################
# methods to construct random multi-body operators
##########################################################################################

# return a random symmetric tensor with zeros on all diagonal blocks
_random_tensor = random_tensor
def random_tensor(dimension):
    return _random_tensor(dimension, lattice_shape, trans_inv, isotropic)

# random `dimension`-local operator that obeys permutational symmetry
def random_op(dimension):
    ops = [ np.random.rand(spin_dim,spin_dim) for op in range(dimension) ]
    perms = np.math.factorial(dimension)
    return sum( functools.reduce(np.kron, [ ops[pp] for pp in perm ])
                for perm in it.permutations(range(dimension)) ) / perms

##########################################################################################
# methods to construct the product of multi-body operators from set diagrams
##########################################################################################

def _embed_operator(base_op, indices):
    return embed_operator(base_op, indices, spin_num, spin_dim)

# build random multi-body operators, each defined by a tensor and a "base" operator
tensors = [ random_tensor(dimension) for dimension in dimensions ]
base_ops = [ random_op(dimension) for dimension in dimensions ]

def build_op(tensor, base_op):
    return sum( tensor[idx] * _embed_operator(base_op, idx)
                for idx in it.combinations(range(spin_num), tensor.ndim) )

# construct the full, exact product of multi-body operators
full_ops = [ build_op(tensor, base_op)
             for tensor, base_op in zip(tensors, base_ops) ]
exact_op = sym_proj @ functools.reduce(operator.matmul, full_ops) @ sym_proj

# for shared index groups of given sizes, generate all allowed values of indices
def index_values(group_sizes, _exclusions = None):
    if _exclusions == None: _exclusions = ()
    group_sizes = list(group_sizes)

    fst_range = ( num for num in range(spin_num) if num not in _exclusions )
    fst_combs = ( comb for comb in it.combinations(fst_range, group_sizes[0]) )

    if len(group_sizes) == 1:
        for fst_comb in fst_combs:
            yield ( fst_comb, )

    else:
        for fst_comb in fst_combs:
            other_combs = index_values(group_sizes[1:], _exclusions + fst_comb)
            for other_comb in other_combs:
                yield ( fst_comb, ) + other_comb

# given a choice of values for grouped indices,
# return the indices associated with a single tensor
def restrict_indices(subset_idx_vals, subsets, primary_index):
    primary_idx_vals = ( values for values, subset in zip(subset_idx_vals, subsets)
                         if primary_index in subset )
    return functools.reduce(operator.add, primary_idx_vals)

# build the analytically simplified form of the exact product of multi-body operators
simp_op = np.zeros(exact_op.shape)
for set_diagram in set_diagrams(dimensions):
    def _index_val_set():
        return index_values(set_diagram.values())
    def _indices(subset_idx_vals, primary_index):
        return restrict_indices(subset_idx_vals, set_diagram.keys(), primary_index)

    diagram_coeficient = 0
    for idx_vals in _index_val_set():
        tensor_vals = ( tensor[_indices(idx_vals, tt)]
                        for tt, tensor in enumerate(tensors) )
        diagram_coeficient += functools.reduce(operator.mul, tensor_vals)

    try:
        idx_vals = next(_index_val_set())
    except StopIteration:
        continue
    evaluated_ops = ( _embed_operator(base_op, _indices(idx_vals, pp))
                      for pp, base_op in enumerate(base_ops) )
    diagram_op = functools.reduce(operator.matmul, evaluated_ops)

    simp_op += diagram_coeficient * ( sym_proj @ diagram_op @ sym_proj )

# verify that the exact and simplified products of multi-body operators are equal
check_equal(simp_op, exact_op)

##########################################################################################
# verify the diagonsis of multi-body excitations
##########################################################################################

# build random SU(n)-symmetric interaction Hamiltonian
sun_coefs = random_tensor(2)
swap = sum( ( lambda op : np.kron(op,op.T) )( unit_tensor((mu,nu), spin_dim) )
            for mu in range(spin_dim) for nu in range(spin_dim) )
sun_interactions = sum( sun_coefs[pp,qq] * _embed_operator(swap, [pp,qq])
                        for pp in range(spin_num) for qq in range(pp) )

# build objects that appear in the multi-body eigenvalue problem
sym_energy = sun_coefs.sum()/2
sun_coef_vec = sum(sun_coefs)

def coef_mod_val(choice):
    return sum( sun_coef_vec[pp] - sum( sun_coefs[pp,qq] for qq in choice )
                for pp in choice )
def coef_mod_mat(shape):
    spin_num = shape[0]
    dimension = len(shape)
    mat = np.zeros(shape)
    for choice in it.combinations(range(spin_num), tensor.ndim):
        for perm in it.permutations(choice):
            mat[perm] = coef_mod_val(perm)
    return mat
def coef_mod(tensor):
    return coef_mod_mat(tensor.shape) * tensor


# remove diagonal elements of a tensor
def _remove_diags(tensor):
    for kk in np.ndindex(tensor.shape):
        if len(set(kk)) != len(kk):
            tensor[kk] = 0
    return tensor

def coef_zip(tensor):
    axes = tensor.ndim
    def _zip_axis(axis):
        prod = np.tensordot(sun_coefs, tensor, axes = [ 1, axis ])
        old_order = [ axis ] + [ jj for jj in range(axes) if jj != axis ]
        new_order = np.arange(axes)[np.argsort(old_order)]
        return np.transpose(prod, axes = new_order)
    zipped_tensor = sum( _zip_axis(axis) for axis in range(axes) )
    return _remove_diags(zipped_tensor)

def coef_act(tensor):
    return coef_zip(tensor) - coef_mod(tensor)

# verify the multi-body eigenvalue problem for every multi-body operator that we built
for tensor, base_op, full_op in zip(tensors, base_ops, full_ops):
    exact = sun_interactions @ full_op @ sym_proj

    coef_act_tensor = coef_act(tensor)
    remaining_op = build_op(coef_act_tensor, base_op)

    simpl = ( sym_energy * full_op + remaining_op ) @ sym_proj

    check_equal(simpl, exact)

##########################################################################################
# verify the multi-body eigenvalue problem
##########################################################################################

# for each allowed tensor dimension
for dimension, base_op, tensor in zip(dimensions, base_ops, tensors):

    excitation_mat, vector_to_tensor, tensor_to_vector \
        = multibody_problem(lattice_shape, sun_coefs, dimension, trans_inv, isotropic = False)

    vector = tensor_to_vector(tensor)
    coef_act_tensor = vector_to_tensor(excitation_mat @ vector)

    check_equal(coef_act_tensor, coef_act(tensor))

    # check that eigenvectors of excitation_mat generate states of definite excitation energy
    energies, vectors = np.linalg.eig(excitation_mat)
    for excitation_energy, vector in zip(energies, vectors.T):

        tensor = vector_to_tensor(vector)
        operator = build_op(tensor, base_op)

        act_tensor = coef_act(tensor)
        check_equal(act_tensor, excitation_energy * tensor)

        exact = sun_interactions @ ( operator @ sym_proj )
        simpl = ( sym_energy + excitation_energy ) * ( operator @ sym_proj )
        check_equal(simpl, exact)
