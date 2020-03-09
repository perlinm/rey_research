#!/usr/bin/env python3

import numpy as np
import itertools as it

##########################################################################################
# dealing with lattice geometry
##########################################################################################

# convert between integer and vector indices for a spin
def index_methods(lattice_shape):
    spin_num = np.prod(lattice_shape)
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
    return to_vec, to_idx

# method to compute the distance between two spins
def dist_method(lattice_shape):
    to_vec, to_idx = index_methods(lattice_shape)
    def dist_1D(pp, qq, axis):
        diff = ( pp - qq ) % lattice_shape[axis]
        return min(diff, lattice_shape[axis] - diff)
    def dist(pp, qq):
        pp = to_vec(pp)
        qq = to_vec(qq)
        return np.sqrt(sum( dist_1D(*pp_qq,aa)**2
                            for aa, pp_qq in enumerate(zip(pp,qq)) ))
    return dist

# method to shift spins by a displacement
def spin_shift_method(lattice_shape):
    to_vec, to_idx = index_methods(lattice_shape)
    def spin_shift(spins, disp, aa = None):
        if aa is None: # shift all spins
            return tuple( to_idx( to_vec(idx) + to_vec(disp) ) for idx in spins )
        else: # only shift spin aa
            new_spins = list(spins)
            new_spins[aa] = to_idx( to_vec(spins[aa]) + to_vec(disp) )
            return tuple(new_spins)
    return spin_shift

##########################################################################################
# methods for building tensors
##########################################################################################

# unit tensor
def unit_tensor(idx, size):
    tensor = np.zeros((size,)*len(idx), dtype = int)
    tensor[tuple(idx)] = 1
    return tensor

# "unit" symmetric tensor, possibly symmetrized over all translations on a lattice
def sym_tensor(idx, size, spin_shift = None):
    dimension = len(idx)
    tensor = np.zeros((size,)*dimension, dtype = int)

    if spin_shift is None: # no translational symmetry
        for perm_idx in it.permutations(idx):
            tensor[perm_idx] = 1

    else: # symmetrize over all translations
        for displacement in range(size):
            shifted_idx = spin_shift(idx,displacement)
            for perm_idx in it.permutations(shifted_idx):
                tensor[perm_idx] |= 1 # set tensor[perm_idx] to 1 (from either 0 or 1)

    return tensor

def random_tensor(dimension, lattice_shape, TI, seed = None):
    if seed is not None: np.random.seed(seed)
    spin_num = np.prod(lattice_shape)
    if TI:
        spin_shift = spin_shift_method(lattice_shape)
        return sum( np.random.rand() * sym_tensor((0,)+choice, spin_num, spin_shift)
                for choice in it.combinations(range(1,spin_num), dimension-1) )
    else:
        return sum( np.random.rand() * sym_tensor(choice, spin_num)
                for choice in it.combinations(range(spin_num), dimension) )

# TODO: symmetrize over rotations/reflections for isotropic systems

##########################################################################################
# setting up the multibody eigenvalue problem
##########################################################################################

def multibody_problem(sun_coefs, index_parts, lattice_shape, TI = True):
    if type(index_parts) is int:
        index_parts = (index_parts,)
    index_num = sum( part for part in index_parts )

    # collect basic system info
    spin_num = sun_coefs.shape[0]
    sun_coef_vec = sum(sun_coefs)
    sun_coef_0 = sun_coef_vec[0]
    spin_shift = spin_shift_method(lattice_shape)

    # identify all distinct choices of spins, modding out by translations if appropriate
    if not TI:
        choices = { choice : idx
                    for idx, choice
                    in enumerate(it.combinations(range(spin_num), index_num)) }
        choice_num = len(choices)

        def choice_idx(choice):
            for perm in it.permutations(choice):
                if perm in choices:
                    return choices.get(perm)

    else:
        choices = {}
        choice_num = 0
        for choice in it.combinations(range(1,spin_num), index_num-1):
            choice = (0,) + choice
            add_to_choices = True
            for shift in range(spin_num):
                shifted_choice = tuple(sorted(spin_shift(choice,shift)))
                if shifted_choice in choices:
                    add_to_choices = False
                    break
            if add_to_choices:
                choices[choice] = choice_num
                choice_num += 1

        def choice_idx(choice):
            if len(set(choice)) != len(choice): return None
            for shift in range(spin_num):
                shifted_choice = sorted(spin_shift(choice,shift))
                idx = choices.get(tuple(shifted_choice))
                if idx is not None:
                    return idx

    # build methods to convert between a tensor and a "choice vector"
    shift_method = spin_shift if TI else None
    def vector_to_tensor(vector):
        return sum( val * sym_tensor(choice, spin_num, shift_method)
                    for val, choice in zip(vector, choices.keys()) )
    def tensor_to_vector(tensor):
        return np.array([ tensor[idx] for idx in choices.keys() ])

    def _diag_val(choice):
        return sum( sun_coefs[pp,qq] for pp in choice for qq in choice ) \
             - sum( sun_coef_vec[pp] for pp in choice )

    # build matrix for many-body eigenvalue problem
    excitation_mat = np.zeros((choice_num, choice_num))
    for choice, idx in choices.items():
        excitation_mat[idx,idx] += _diag_val(choice)
        for pp, aa in it.product(range(spin_num), range(index_num)):
            choice_aa = choice[aa]
            choice_aa_pp = list(choice); choice_aa_pp[aa] = pp
            choice_aa_pp_idx = choice_idx(choice_aa_pp)
            if choice_aa_pp_idx is None: continue
            excitation_mat[idx,choice_aa_pp_idx] += sun_coefs[choice_aa,pp]

    return excitation_mat, vector_to_tensor, tensor_to_vector
