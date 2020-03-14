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
            new_spins = ( to_idx( to_vec(spin) + sign * to_vec(disp) ) for spin in spins )
        else: # only shift spin `idx`
            new_spins = list(spins)
            new_spins[idx] = to_idx( to_vec(spins[idx]) + sign * to_vec(disp) )
        return tuple(sorted(new_spins))
    return spin_shift

# method to reflect spins about an axis
def spin_reflect_method(lattice_shape):
    to_vec, to_idx = index_methods(lattice_shape)
    def spin_reflect(spins, reflection):
        def _reflect(vec):
            return [ pos if not reflect else size - pos
                     for pos, reflect, size in zip(vec, reflection, lattice_shape) ]
        new_spins = ( to_idx(_reflect(to_vec(spin))) for spin in spins )
        return tuple(sorted(new_spins))
    return spin_reflect

##########################################################################################
# methods for building tensors
##########################################################################################

# unit tensor
def unit_tensor(idx, size):
    tensor = np.zeros((size,)*len(idx), dtype = int)
    tensor[idx] = 1
    return tensor

# unit symmetric tensor
def sym_tensor(choice, size):
    dimension = len(choice)
    tensor = np.zeros((size,)*dimension, dtype = int)
    for idx in it.permutations(choice):
        tensor[idx] = 1
    return tensor

# unit symmetric tensor, symmetrized over equivalence classes
def sym_tensor_equivs(choice, size, equivalence_class):
    dimension = len(choice)
    tensor = np.zeros((size,)*dimension, dtype = int)
    for equivalent_choice in equivalence_class(choice):
        for idx in it.permutations(equivalent_choice):
            tensor[idx] = 1
    return tensor

# unit symmetric tensor, symmetrized over translations
def sym_tensor_TI(choice, size, spin_shift):
    def equivalence_class(choice):
        return set( spin_shift(choice,shift) for shift in range(size) )
    return sym_tensor_equivs(choice, size, equivalence_class)

def random_tensor(dimension, lattice_shape, TI, seed = None):
    if seed is not None: np.random.seed(seed)
    spin_num = np.prod(lattice_shape)
    if TI:
        spin_shift = spin_shift_method(lattice_shape)
        return sum( np.random.rand() *
                    sym_tensor_TI((0,)+choice, spin_num, spin_shift)
                    for choice in it.combinations(range(1,spin_num), dimension-1) )
    else:
        return sum( np.random.rand() * sym_tensor(choice, spin_num)
                    for choice in it.combinations(range(spin_num), dimension) )

##########################################################################################
# setting up the multibody eigenvalue problem
##########################################################################################

def multibody_problem(lattice_shape, sun_coefs, dimension, TI = None, isotropic = None):
    if dimension == 0:
        excitation_mat = np.zeros((1,1))
        def vector_to_tensor(vector): return vector[0] * np.ones(())
        def tensor_to_vector(tensor): return tensor * np.ones(1)
        return excitation_mat, vector_to_tensor, tensor_to_vector

    # compute total spin numder and SU(n) coupling vector
    spin_num = np.prod(lattice_shape)
    sun_coef_vec = sum(sun_coefs)

    # identify lattice symmetries
    if TI is None:
        TI = np.allclose(sun_coef_vec/sun_coef_vec[0], np.ones(spin_num))
    if isotropic is None:
        isotropic = TI
    assert( not ( isotropic and not TI ) )
    symmetrize_rotations = ( isotropic and len(set(lattice_shape)) == 1 )

    # identify all equivalence classes of choices of spins
    # TODO: symmetrize properly for isotropic systems
    if TI: # translationally invariant and maybe isotropic systems
        spin_shift = spin_shift_method(lattice_shape)

        # return the equivalence class of a choice of spins
        def equivalence_class(choice):
            return set( spin_shift(choice,shift) for shift in range(spin_num) )

        # return the label for the equivalence class of a choice of spins
        def class_label(choice):
            return sorted( spin_shift(choice,shift,neg=True) for shift in choice )[0]

        # construct all equivalence classes of choices of spins
        classes = {}
        class_num = 0
        for reduced_choice in it.combinations(range(1,spin_num), dimension-1):
            label = (0,) + reduced_choice
            if class_label(label) not in classes:
                classes[label] = class_num
                class_num += 1

        # dictionary mapping the label of an equivalence class
        #   to the corresponding set of spin choices
        label_class = { label : equivalence_class(label)
                        for label in classes }

        # get the index of the equivalence class of a choice of spins
        def get_class_idx(choice):
            return classes.get(class_label(choice))

        # construct a unit tensor symmetrized over an equivalence class
        def class_tensor(label):
            return sym_tensor_equivs(label, spin_num, equivalence_class)

    else: # no translational invariance
        classes = { choice : idx
                    for idx, choice
                    in enumerate(it.combinations(range(spin_num), dimension)) }

        label_class = { label : set({label}) for label in classes }

        def get_class_idx(choice):
            return classes.get(tuple(sorted(choice)))

        def class_tensor(choice):
            return sym_tensor(choice, spin_num)

    # equivalence class sizes (multiplicities)
    mults = np.array([ len(_class) for _class in label_class.values() ])
    sqrt_mults = np.sqrt(mults)

    # build matrix for many-body eigenvalue problem
    def _diag_val(choice):
        return sum( sun_coefs[pp,qq] for pp in choice for qq in choice ) \
             - sum( sun_coef_vec[pp] for pp in choice )
    excitation_mat = np.diag([ _diag_val(choice) for choice in classes ])
    for label, idx in classes.items():
        for choice in label_class[label]:
            for pp, aa in it.product(range(spin_num), range(dimension)):
                choice_aa = choice[aa]
                if pp == choice_aa: continue
                choice_aa_pp = list(choice); choice_aa_pp[aa] = pp
                if len(set(choice_aa_pp)) != dimension: continue
                choice_aa_pp_idx = get_class_idx(choice_aa_pp)
                norm = sqrt_mults[idx] * sqrt_mults[choice_aa_pp_idx]
                excitation_mat[idx,choice_aa_pp_idx] += sun_coefs[choice_aa,pp] / norm

    # convert between a tensor and a vector of equivalence class coefficients
    def vector_to_tensor(vector):
        return sum( val * class_tensor(choice)
                    for val, choice in zip(vector/sqrt_mults, classes) )
    def tensor_to_vector(tensor):
        return sqrt_mults * np.array([ tensor[label] for label in classes ])

    return excitation_mat, vector_to_tensor, tensor_to_vector
