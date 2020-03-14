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
    def spin_shift(spins, disp, neg = False, idx = None):
        sign = 1 if not neg else -1
        if idx is None: # shift all spins
            return tuple( to_idx( to_vec(idx) + sign * to_vec(disp) ) for idx in spins )
        else: # only shift spin `idx`
            new_spins = list(spins)
            new_spins[idx] = to_idx( to_vec(spins[idx]) + sign * to_vec(disp) )
            return tuple(new_spins)
    return spin_shift

# method to reflect spins about an axis
def spin_reflect_method(lattice_shape):
    to_vec, to_idx = index_methods(lattice_shape)
    def spin_reflect(spins, reflection):
        reflection = np.array(reflection)
        return tuple( to_idx( reflection * to_vec(idx) ) for idx in spins )
    return spin_reflect

##########################################################################################
# methods for building tensors
##########################################################################################

# unit tensor
def unit_tensor(idx, size):
    tensor = np.zeros((size,)*len(idx), dtype = int)
    tensor[tuple(idx)] = 1
    return tensor

# unit symmetric tensor
def sym_tensor(idx, size):
    dimension = len(idx)
    tensor = np.zeros((size,)*dimension, dtype = int)
    for perm_idx in it.permutations(idx):
        tensor[perm_idx] = 1
    return tensor

# unit symmetric tensor, symmetrized over all translations
#    and possibly some other equivalence relation
def sym_tensor_symmetrized(idx, size, spin_shift, get_equivalent = None):
    if get_equivalent is None:
        get_equivalent = lambda idx : idx
    dimension = len(idx)
    tensor = np.zeros((size,)*dimension, dtype = int)
    for shift, equivalent_idx in it.product(range(size),get_equivalent(idx)):
        shifted_idx = spin_shift(equivalent_idx,shift)
        for perm_idx in it.permutations(shifted_idx):
            tensor[perm_idx] |= 1 # set tensor[perm_idx] to 1 (from either 0 or 1)
    return tensor

def random_tensor(dimension, lattice_shape, TI, seed = None):
    if seed is not None: np.random.seed(seed)
    spin_num = np.prod(lattice_shape)
    if TI:
        spin_shift = spin_shift_method(lattice_shape)
        return sum( np.random.rand() *
                    sym_tensor_symmetrized((0,)+choice, spin_num, spin_shift)
                    for choice in it.combinations(range(1,spin_num), dimension-1) )
    else:
        return sum( np.random.rand() * sym_tensor(choice, spin_num)
                    for choice in it.combinations(range(spin_num), dimension) )

##########################################################################################
# setting up the multibody eigenvalue problem
##########################################################################################

def multibody_problem(lattice_shape, sun_coefs, dimension, TI = True, isotropic = None):
    if dimension == 0:
        excitation_mat = np.zeros((1,1))
        def vector_to_tensor(vector): return vector[0] * np.ones(())
        def tensor_to_vector(tensor): return tensor * np.ones(1)
        return excitation_mat, vector_to_tensor, tensor_to_vector

    # collect basic system info
    spin_num = sun_coefs.shape[0]
    sun_coef_vec = sum(sun_coefs)
    sun_coef_0 = sun_coef_vec[0]
    spin_shift = spin_shift_method(lattice_shape)
    assert( not ( isotropic and not TI ) )
    if isotropic is None: isotropic = TI

    def _diag_val(choice):
        return sum( sun_coefs[pp,qq] for pp in choice for qq in choice ) \
             - sum( sun_coef_vec[pp] for pp in choice )

    # identify all distinct choices of spins,
    #   modding out by translations and rotations/reflections if appropriate
    if not TI: # no translational invariance
        choices = { choice : idx
                    for idx, choice
                    in enumerate(it.combinations(range(spin_num), dimension)) }
        choice_num = len(choices)

        def get_choice_idx(choice):
            for perm in it.permutations(choice):
                idx = choices.get(perm)
                if idx is not None:
                    return idx

        def choice_tensor(choice):
            return sym_tensor(choice, spin_num)

        # build matrix for many-body eigenvalue problem
        excitation_mat = np.zeros((choice_num, choice_num))
        for choice, idx in choices.items():
            excitation_mat[idx,idx] += _diag_val(choice)
            for pp, aa in it.product(range(spin_num), range(dimension)):
                choice_aa = choice[aa]
                choice_aa_pp = list(choice); choice_aa_pp[aa] = pp
                if len(set(choice_aa_pp)) != dimension: continue
                choice_aa_pp_idx = get_choice_idx(choice_aa_pp)
                excitation_mat[idx,choice_aa_pp_idx] += sun_coefs[choice_aa,pp]

    # translationally invariant and maybe isotropic systems
    else:
        # given a choice of spins, return all choices in its equivalence class
        if not isotropic:
            def get_equivalent(spins):
                return ( tuple(sorted(spin_shift(spins,shift,neg=True)))
                         for shift in spins )
        else:
            # get all reflections of a spin choice that are distinct up to a translation
            spin_reflect = spin_reflect_method(lattice_shape)
            def get_reflections(spins):
                fixed_spins = spin_shift(spins,spins[0],neg=True)
                reflections = []
                for reflection in it.product([1,-1], repeat = len(lattice_shape)):
                    ref_spins = spin_reflect(fixed_spins, reflection)
                    add_reflection = True
                    for shift in ref_spins:
                        shifted_spins = spin_shift(ref_spins,shift,neg=True)
                        shifted_spins = tuple(sorted(shifted_spins))
                        if shifted_spins in reflections:
                            add_reflection = False
                            break
                    if add_reflection:
                        reflections += [ ref_spins ]
                return reflections

            # TODO: address remaining symmetries (after TI and reflections)
            #       in isotropic systems
            def get_equivalent(spins):
                return set( tuple(sorted(spin_shift(reflected_choice,shift,neg=True)))
                            for reflected_choice in get_reflections(spins)
                            for shift in reflected_choice )

        # construct all equivalence classes of choices of spins
        choices = {}
        choice_num = 0
        for choice in it.combinations(range(1,spin_num), dimension-1):
            choice = (0,) + choice
            add_to_choices = True
            for equivalent_choice in get_equivalent(choice):
                if equivalent_choice in choices:
                    add_to_choices = False
                    break
            if add_to_choices:
                choices[choice] = choice_num
                choice_num += 1

        # get the equivalence class of a choice of spins
        def get_choice_idx(choice):
            if len(set(choice)) != len(choice): return None
            for equivalent_choice in get_equivalent(choice):
                idx = choices.get(equivalent_choice)
                if idx is not None:
                    return idx

        # construct an appropriately symmetrized tensor corresponding to a choice of spins
        def choice_tensor(choice):
            return sym_tensor_symmetrized(choice, spin_num, spin_shift, get_equivalent)

        # build matrix for many-body eigenvalue problem
        excitation_mat = np.zeros((choice_num, choice_num))
        for choice, idx in choices.items():
            excitation_mat[idx,idx] += _diag_val(choice)
            for pp, aa in it.product(range(spin_num), range(dimension)):
                choice_aa = choice[aa]
                choice_aa_pp = list(choice); choice_aa_pp[aa] = pp
                if len(set(choice_aa_pp)) != dimension: continue
                choice_aa_pp_idx = get_choice_idx(choice_aa_pp)
                excitation_mat[idx,choice_aa_pp_idx] += sun_coefs[choice_aa,pp]

    # build methods to convert between a tensor and a "choice vector"
    def vector_to_tensor(vector):
        return sum( val * choice_tensor(choice)
                    for val, choice in zip(vector, choices) )
    def tensor_to_vector(tensor):
        return np.array([ tensor[choice] for choice in choices ])

    return excitation_mat, vector_to_tensor, tensor_to_vector
