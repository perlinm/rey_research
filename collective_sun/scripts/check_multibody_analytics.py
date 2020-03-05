#!/usr/bin/env python3

import numpy as np
import itertools as it
import functools, operator

from itertools_extension import assignments, set_diagrams

translational_invariance = True

min_spins = 3
max_spins = 8
max_dim = 2**max_spins

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
# methods for translations on the lattice
##########################################################################################

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

# shift a choice of spins spins by a displacement
def shift_spins(spins, disp, aa = None):
    if aa is None: # shift all spins
        return tuple( to_idx( to_vec(idx) + to_vec(disp) ) for idx in spins )
    else: # only shift spin aa
        new_spins = list(spins)
        new_spins[aa] = to_idx( to_vec(spins[aa]) + to_vec(disp) )
        return tuple(new_spins)

##########################################################################################
# basic objects for building vectors / tensors
##########################################################################################

def unit_tensor(idx, size = spin_num):
    tensor = np.zeros((size,)*len(idx))
    tensor[tuple( np.array(idx) % size )] = 1
    return tensor

def unit_vector(idx, size = spin_dim):
    return unit_tensor((idx,), size)

def sym_tensor(choice):
    return sum( unit_tensor(kk) for kk in it.permutations(choice) )

def sym_shift_tensor(choice):
    tensor = np.zeros((spin_num,)*len(choice))
    for center in range(spin_num):
        idx = shift_spins(choice,center)
        tensor += sym_tensor(idx)
    tensor /= tensor.max()
    return tensor

def _remove_diags(tensor):
    for kk in np.ndindex(tensor.shape):
        if len(set(kk)) != len(kk):
            tensor[kk] = 0
    return tensor

##########################################################################################
# construct projector onto the fully symmetric manifold
##########################################################################################

# labels for all fully symmetric states
sym_labels = list(assignments(spin_num,spin_dim))

# build a fully symmetric state labelled by occupation number
def sym_state(occupations):
    assert(sum(occupations) == spin_num)
    labels = [ [mm]*pop for mm, pop in enumerate(occupations) ]
    labels = np.concatenate(labels).astype(int)
    def _base_state(label):
        return unit_vector(label)
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
    if not translational_invariance:
        return sum( np.random.rand() * sym_tensor(choice)
                    for choice in it.combinations(range(spin_num), dimension) )
    else:
        return sum( np.random.rand() * sym_shift_tensor((0,)+choice)
                    for choice in it.combinations(range(1,spin_num), dimension-1) )

# random `dimension`-local operator that obeys permutational symmetry
def random_op(dimension):
    ops = [ np.random.rand(spin_dim,spin_dim) for op in range(dimension) ]
    perms = np.math.factorial(dimension)
    return sum( functools.reduce(np.kron, [ ops[pp] for pp in perm ])
                for perm in it.permutations(range(dimension)) ) / perms

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

def build_op(tensor, base_op):
    return sum( tensor[idx] * act(base_op, idx)
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
    evaluated_ops = ( act(base_op, _indices(idx_vals, pp))
                      for pp, base_op in enumerate(base_ops) )
    diagram_op = functools.reduce(operator.matmul, evaluated_ops)

    simp_op += diagram_coeficient * ( sym_proj @ diagram_op @ sym_proj )

# verify that the exact and simplified products of multi-body operators are equal
print(np.allclose(simp_op, exact_op))

##########################################################################################
# verify the diagonsis of multi-body excitations
##########################################################################################

# build random SU(n)-symmetric interaction Hamiltonian
sun_coefs = random_tensor(2)
swap = sum( ( lambda op : np.kron(op,op.T) )( unit_tensor((mu,nu), spin_dim) )
            for mu in range(spin_dim) for nu in range(spin_dim) )
sun_interactions = sum( sun_coefs[pp,qq] * act(swap,[pp,qq])
                        for pp in range(spin_num) for qq in range(pp) )

# build objects that appear in the multi-body eigenvalue problem
sym_energy = sun_coefs.sum()/2
sun_coef_vec = sum(sun_coefs)

def _coef_mod_val(choice):
    return sum( sun_coef_vec[pp] - sum( sun_coefs[pp,qq] for qq in choice )
                for pp in choice )
def coef_mod(tensor):
    coef_mod = np.zeros(tensor.shape)
    for choice in it.combinations(range(spin_num), tensor.ndim):
        for perm in it.permutations(choice):
            coef_mod[perm] = _coef_mod_val(perm)
    return coef_mod * tensor

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

    print(np.allclose(simpl, exact))

##########################################################################################
# verify the multi-body eigenvalue problem
##########################################################################################

if not translational_invariance: exit()

# for each allowed tensor dimension
for dimension, base_op in zip(dimensions, base_ops):

    # identify ( all choices of spins ) modulo ( translations )
    trans_choices = {}
    distinct_choices = 0
    for choice in it.combinations(range(1,spin_num), dimension-1):
        choice = (0,) + choice
        add_to_choices = True
        for center in range(spin_num):
            shifted_choice = tuple(sorted(shift_spins(choice,center)))
            if shifted_choice in trans_choices:
                add_to_choices = False
                break
        if add_to_choices:
            trans_choices[choice] = distinct_choices
            distinct_choices += 1

    # get the index of a spin choice
    def _choice_trans_idx(choice):
        if len(set(choice)) != len(choice): return None
        for shift in range(spin_num):
            shifted_choice = shift_spins(choice,shift)
            shifted_choice = tuple(sorted(shifted_choice))
            idx = trans_choices.get(shifted_choice)
            if idx is not None:
                return idx

    def vector_to_tensor(vector):
        return sum( val * sym_shift_tensor(choice)
                    for val, choice in zip(vector, trans_choices.keys()) )

    # build matrix for many-body eigenvalue problem
    dim = len(trans_choices)
    eig_mat = np.zeros((dim,dim))
    for choice, kk in trans_choices.items():
        eig_mat[kk,kk] += sum( sun_coefs[pp,qq] for pp in choice for qq in choice )
        for dd, aa in it.product(range(spin_num),range(dimension)):
            shifted_choice = shift_spins(choice,dd,aa)
            idx = _choice_trans_idx(shifted_choice)
            if idx is None: continue
            eig_mat[kk,idx] += sun_coefs[dd,0]

    # check that eigenvectors of eig_mat generate states of definite excitation energy
    values, vectors = np.linalg.eig(eig_mat)
    for value, vector in zip(values, vectors.T):
        excitation_energy = value - dimension * sun_coef_vec[0]

        tensor = vector_to_tensor(vector)
        operator = build_op(tensor, base_op)

        act_tensor = coef_act(tensor)
        print(np.allclose(act_tensor, excitation_energy * tensor))

        exact = sun_interactions @ operator @ sym_proj
        simpl = ( sym_energy + excitation_energy ) * operator @ sym_proj
        print(np.allclose(simpl, exact))
