#!/usr/bin/env python3

import numpy as np
import itertools as it
import functools, operator

from itertools_extension import assignments, set_diagrams

max_spins = 10 # maximum number of spins
max_dim = 2**10 # maximum dimension of Hilbert space that we want to test out

total_dim = max_dim # total dimension of Hilbert space
while total_dim >= max_dim:
    spin_dim = np.random.randint(2,5) # dimension of each spin
    lattice_shape = ( np.random.randint(1,5),
                      np.random.randint(1,5) )
    spin_num = np.product(lattice_shape)
    if spin_num > max_spins or spin_num == 1: continue
    total_dim = spin_dim**spin_num

op_num = np.random.randint(2,5) # total number of multi-body operators

# the dimension of each multi-body operator: an operator with dimension M is an M-local operator
dimensions = [ np.random.randint(1,spin_num) for _ in range(op_num) ]

print("dimension of each spin:", spin_dim)
print("lattice shape:", lattice_shape)
print("total number of spins:", spin_num)
print("multi-body operator dimensions", dimensions)

##########################################################################################
# construct projector onto the fully symmetric manifold
##########################################################################################

# labels for all fully symmetric states
sym_labels = list(assignments(spin_num,spin_dim))

def unit_vector(size, idx):
    vec = np.zeros(size)
    vec[idx] = 1
    return vec

# build a fully symmetric state labelled by occupation number
def sym_state(occupations):
    assert(sum(occupations) == spin_num)
    labels = [ [mm]*pop for mm, pop in enumerate(occupations) ]
    labels = np.concatenate(labels).astype(int)
    def _base_state(label):
        return unit_vector(spin_dim, label)
    vec = sum( functools.reduce(np.kron, map(_base_state, perm))
               for perm in it.permutations(labels) )
    return vec / np.sqrt(vec @ vec)

# construct a projector out of a state
def to_proj(state):
    return np.outer(state, state.conj())

# projector onto the fully symmetric manifold
sym_proj = sum( to_proj(sym_state(label)) for label in sym_labels )

##########################################################################################
# methods to construct random multi-body operators
##########################################################################################

# return a random symmetric tensor with zeros on all diagonal blocks
def random_tensor(dimension):
    tensor = np.zeros((spin_num,)*dimension)
    for comb in it.combinations(range(spin_num), dimension):
        num = np.random.rand()
        for perm in it.permutations(comb):
            tensor[perm] = num
    return tensor

# random `dimension`-local operator that obeys permutational symmetry
def random_op(dimension):
    ops = [ np.random.rand(spin_dim,spin_dim) for op in range(dimension) ]
    return sum( functools.reduce(np.kron, [ ops[pp] for pp in perm ])
                for perm in it.permutations(range(dimension)) ) / np.math.factorial(dimension)

# act with the multi-local operator `op` on the spins indexed by `indices`
def act(op, indices):
    if not hasattr(indices, "__getitem__"):
        indices = list(indices)

    for _ in range(spin_num - len(indices)):
        op = np.kron(op, np.eye(spin_dim))

    # collect and flatten tensor factors associated with each spin
    fst_half = range(spin_num)
    snd_half = range(spin_num,2*spin_num)
    perm = np.array(list(zip(list(fst_half),list(snd_half)))).flatten()
    op = np.reshape(op, (spin_dim,)*2*spin_num)
    op = np.transpose(op, perm)
    op = np.reshape(op, (spin_dim**2,)*spin_num)

    # rearrange tensor factors according to the desired qubit order
    old_order = list(indices) + [ jj for jj in range(spin_num) if jj not in indices ]
    new_order = np.arange(spin_num)[np.argsort(old_order)]
    op = np.transpose(op, new_order)
    op = np.reshape(op, (spin_dim,)*2*spin_num)

    # un-flatten the tensor factors, and flatten the tensor into
    #   a matrix that acts on the joint Hilbert space of all qubits
    evens = range(0,2*spin_num,2)
    odds = range(1,2*spin_num,2)
    op = np.transpose(op, list(evens)+list(odds))
    return np.reshape(op, (spin_dim**spin_num, spin_dim**spin_num))

##########################################################################################
# methods to construct the product of multi-body operators from set diagrams
##########################################################################################

# build random multi-body operators, each defined by a tensor and a "base" operator
tensors = [ random_tensor(dimension) for dimension in dimensions ]
base_ops = [ random_op(dimension) for dimension in dimensions ]

# construct the full, exact product of multi-body operators
full_ops = [ sum( tensor[idx] * act(base_op, idx)
                  for idx in it.combinations(range(spin_num), dimension) )
             for tensor, base_op, dimension in zip(tensors, base_ops, dimensions) ]
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

    diagram_coefficient = 0
    for idx_vals in _index_val_set():
        tensor_vals = ( tensor[_indices(idx_vals, tt)]
                        for tt, tensor in enumerate(tensors) )
        diagram_coefficient += functools.reduce(operator.mul, tensor_vals)

    try:
        idx_vals = next(_index_val_set())
    except StopIteration:
        continue
    evaluated_ops = ( act(base_op, _indices(idx_vals, pp))
                      for pp, base_op in enumerate(base_ops) )
    diagram_op = functools.reduce(operator.matmul, evaluated_ops)

    simp_op += diagram_coefficient * ( sym_proj @ diagram_op @ sym_proj )

# verify that the exact and simplified products of multi-body operators are equal
print(np.allclose(simp_op, exact_op))

##########################################################################################
# verify the multi-body eigenvalue problem
##########################################################################################

# build random SU(n)-symmetric interaction Hamiltonian
sun_coeffs = random_tensor(2)
swap = sum( ( lambda op : np.kron(op.transpose(),op) )
            ( np.outer(unit_vector(spin_dim,nu),
                       unit_vector(spin_dim,mu)) )
            for mu in range(spin_dim) for nu in range(spin_dim) )
sun_interactions = sum( sun_coeffs[pp,qq] * act(swap,[pp,qq])
                        for pp in range(spin_num) for qq in range(pp) )

# build objects that appear in the multi-body eigenvalue problem
sym_energy = sun_coeffs.sum()/2
sun_coeff_vec = sum(sun_coeffs)
def _coeff_mod(spin_choice):
    return sum( sun_coeff_vec[pp] - sum( sun_coeffs[pp,qq] for qq in spin_choice )
                for pp in spin_choice )
def _coeff_zip_tensor(tensor):
    axes = tensor.ndim
    def _zip_axis(axis):
        prod = np.tensordot(sun_coeffs, tensor, axes = [ 1, axis ])
        old_order = [ axis ] + [ jj for jj in range(axes) if jj != axis ]
        new_order = np.arange(axes)[np.argsort(old_order)]
        return np.transpose(prod, axes = new_order)
    return sum( _zip_axis(axis) for axis in range(axes) )

# verify the multi-body eigenvalue problem for every multi-body operator that we built
for tensor, base_op, full_op in zip(tensors, base_ops, full_ops):
    exact = sun_interactions @ full_op @ sym_proj

    _coeff_zip = _coeff_zip_tensor(tensor)
    remaining_op = sum( ( _coeff_zip[idx] - _coeff_mod(idx) * tensor[idx] )
                        * act(base_op, idx)
                        for idx in it.combinations(range(spin_num), tensor.ndim) )

    simpl = ( sym_energy * full_op + remaining_op ) @ sym_proj

    print(np.allclose(simpl, exact))
