#!/usr/bin/env python3

import numpy as np
import itertools as it
import functools

# identity function
def iden(xx): return xx

# function composition
def compose(*functions):
    return functools.reduce(lambda ff, gg: lambda xx: ff(gg(xx)), functions, iden)

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
def dist_method(lattice_shape, _index_methods = None):
    if _index_methods is None:
        _index_methods = index_methods(lattice_shape)
    to_vec, to_idx = _index_methods

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
def spin_shift_method(lattice_shape, _index_methods = None):
    if _index_methods is None:
        _index_methods = index_methods(lattice_shape)
    to_vec, to_idx = _index_methods

    def spin_shift(spins, disp, neg = False, idx = None):
        sign = 1 if not neg else -1
        if idx is None: # shift all spins
            new_spins = ( to_idx( to_vec(spin) + sign * to_vec(disp) ) for spin in spins )
        else: # only shift spin `idx`
            new_spins = list(spins)
            new_spins[idx] = to_idx( to_vec(spins[idx]) + sign * to_vec(disp) )
        return tuple(sorted(new_spins))
    return spin_shift

# get generators of all lattice symmetries
# each generator acts on a choice of lattice sites
# TODO: allow singling out a single axis for separate treatment
def lattice_symmetries(lattice_shape, _index_methods = None):
    if _index_methods is None:
        _index_methods = index_methods(lattice_shape)

    dimension = len(lattice_shape)

    # the lattice has a different size along each principal axis
    if len(set(lattice_shape)) == dimension:
        axis_symmetries = [ line_symmetries(_index_methods, axis)
                            for axis in range(dimension) ]
        return [ compose(*symmetries)
                 for symmetries in it.product(*axis_symmetries) ]

    if dimension == 2: # the lattice is a square
        return square_symmetries(_index_methods, [ 0, 1 ])

    if dimension == 3:
        if len(set(lattice_shape)) == 1: # the lattice is a cube
            return cube_symmetries(_index_methods)

        if len(set(lattice_shape)) == 2: # the lattice is a ( square x line )
            for axis_lft, axis_rht in it.combinations(range(3), 2):
                if lattice_shape[axis_lft] == lattice_shape[axis_rht]:
                    square_axes = ( axis_lft, axis_rht )
                    break
            line_axis = next( axis for axis in range(3) if axis not in square_axes )
            return [ compose(square_symmetry, line_symmetry)
                     for square_symmetry in square_symmetries(_index_methods, square_axes)
                     for line_symmetry in line_symmetries(_index_methods, line_axis) ]

    if dimension > 3:
        raise NotImplementedError("lattice symmetries in dimension > 3")

def line_symmetries(_index_methods, axis):
    to_vec, to_idx = _index_methods
    lattice_shape = tuple(to_vec(-1))
    def reflect_single(site):
        new_site = to_vec(site)
        new_site[axis] = lattice_shape[axis] - new_site[axis]
        return to_idx(new_site)
    def reflect_all(sites):
        return tuple( reflect_single(site) for site in sites )
    return [ iden, reflect_all ]

def square_symmetries(_index_methods, axes):
    to_vec, to_idx = _index_methods
    lattice_shape = tuple(to_vec(-1))
    lattice_size = lattice_shape[axes[0]]
    assert(lattice_size == lattice_shape[axes[1]])
    def rotate_single(site):
        pos = to_vec(site)
        new_pos = [ pos[axes[1]], lattice_size - pos[axes[0]] ]
        return to_idx(new_pos)
    def rotate_all(sites):
        return tuple( rotate_single(site) for site in sites )
    rotations = [ compose(*[rotate_all]*num) for num in range(4) ]
    reflections = line_symmetries(_index_methods, axes[0])
    return [ compose(rotation, reflection)
             for rotation in rotations
             for reflection in reflections ]

def cube_symmetries(_index_methods):
    to_vec, to_idx = _index_methods
    lattice_shape = tuple(to_vec(-1))
    raise NotImplementedError("construction of cube symmetries")

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
    if TI: # translationally invariant and maybe isotropic systems
        spin_shift = spin_shift_method(lattice_shape)

        # return the equivalence class of a choice of spins, as either a set or a label
        if isotropic:
            _lattice_symmetries = lattice_symmetries(lattice_shape)
        else:
            _lattice_symmetries = [ iden ]

        def equivalence_class(choice, classes = {}):
            choice = tuple(choice)
            if choice not in classes:
                classes[choice] = set( spin_shift(symmetry(choice),shift)
                                       for symmetry in _lattice_symmetries
                                       for shift in range(spin_num) )
            return classes[choice]
        def class_label(choice, labels = {}):
            choice = tuple(choice)
            if not choice in labels:
                labels[choice] = min( spin_shift(symmetry_choice,shift,neg=True)
                                      for symmetry in _lattice_symmetries
                                      if ( symmetry_choice := symmetry(choice) )
                                      for shift in symmetry_choice )
            return labels[choice]

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
