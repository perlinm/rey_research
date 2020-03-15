#!/usr/bin/env python3

import sys
import numpy as np
import itertools as it
import functools

from itertools_extension import unique_permutations
from operator_product_methods import compute_overlap

# identity function
def iden(xx): return xx

# function composition
def compose(*functions):
    return functools.reduce(lambda ff, gg: lambda xx: ff(gg(xx)), functions, iden)

##########################################################################################
# methods to deal with lattice geometry and symmetries
##########################################################################################

# convert between integer and vector indices for a lattice site
def index_methods(lattice_shape):
    site_num = np.prod(lattice_shape)
    _to_vec = { idx : tuple(vec) for idx, vec in enumerate(np.ndindex(lattice_shape)) }
    _to_idx = { vec : idx for idx, vec in _to_vec.items() }
    def to_vec(idx):
        if hasattr(idx, "__getitem__"):
            return np.array(idx) % np.array(lattice_shape)
        return np.array(_to_vec[ idx % site_num ])
    def to_idx(vec):
        if type(vec) is int:
            return vec % site_num
        return _to_idx[ tuple( np.array(vec) % np.array(lattice_shape) ) ]
    return to_vec, to_idx

# method to compute the distance between two lattice sites
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

# method to shift sites by a displacement
def site_shift_method(lattice_shape, _index_methods = None):
    if _index_methods is None:
        _index_methods = index_methods(lattice_shape)
    to_vec, to_idx = _index_methods

    def shift_sites(sites, disp, neg = False, idx = None):
        sign = 1 if not neg else -1
        if idx is None: # shift all sites
            new_sites = ( to_idx( to_vec(site) + sign * to_vec(disp) ) for site in sites )
        else: # only shift site `idx`
            new_sites = list(sites)
            new_sites[idx] = to_idx( to_vec(sites[idx]) + sign * to_vec(disp) )
        return tuple(sorted(new_sites))
    return shift_sites

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

# methods to deal with equivalence classes (ECs) of choices of lattice sites
# returns: (1) `classes`: dictionary mapping an EC label to an EC index
#          (2) `equivalence_class`: function mapping a choice of sites to an EC (as a set)
#          (3) `class_label`: function mapping a choice of sites to an EC label
#          (4) `class_tensor`: function mapping a choice of sites to a unit EC tensor
def equivalence_class_methods(dimension, lattice_shape, TI, isotropic = None):
    site_num = np.prod(lattice_shape)
    if isotropic is None: isotropic = TI
    assert( not ( isotropic and not TI ) )

    # identify all equivalence classes of choices of lattice sites
    if TI: # translationally invariant and maybe isotropic systems
        shift_sites = site_shift_method(lattice_shape)

        if isotropic:
            _lattice_symmetries = lattice_symmetries(lattice_shape)
        else:
            _lattice_symmetries = [ iden ]

        # return the equivalence class of a choice of sites, as either a set or a label
        def equivalence_class(choice, classes = {}):
            try: return classes[choice]
            except:
                new_class = set( shift_sites(symmetry(choice),shift)
                                 for symmetry in _lattice_symmetries
                                 for shift in range(site_num) )
                classes[choice] = new_class
                return new_class
        def class_label(choice, labels = {}):
            try: return labels[choice]
            except:
                new_label = min( shift_sites(symmetry_choice,shift,neg=True)
                                 for symmetry in _lattice_symmetries
                                 if ( symmetry_choice := symmetry(choice) )
                                 for shift in symmetry_choice )
                labels[choice] = new_label
                return new_label

        # construct a unit tensor symmetrized over an equivalence class
        def class_tensor(label):
            return sym_tensor_equivs(label, site_num, equivalence_class)

        # construct all equivalence classes of choices of sites
        classes = {}
        class_num = 0
        for reduced_choice in it.combinations(range(1,site_num), dimension-1):
            label = (0,) + reduced_choice
            if class_label(label) not in classes:
                classes[label] = class_num
                class_num += 1

    else: # no translational invariance
        classes = { choice : idx
                    for idx, choice
                    in enumerate(it.combinations(range(site_num), dimension)) }
        def equivalence_class(choice): return set({choice})
        def class_label(choice): return choice
        def class_tensor(choice): return sym_tensor(choice, site_num)

    return classes, equivalence_class, class_label, class_tensor

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
def sym_tensor_TI(choice, size, shift_sites):
    def equivalence_class(choice):
        return set( shift_sites(choice,shift) for shift in range(size) )
    return sym_tensor_equivs(choice, size, equivalence_class)

def random_tensor(dimension, lattice_shape, TI, isotropic = None, seed = None):
    if seed is not None: np.random.seed(seed)
    if isotropic is None: isotropic = TI
    if isotropic and not TI:
        print("WARNING: a tensor cannot be (1) isotropic but not (2) translationally invariant.")
        print("         builing a tensor that is neither (1) nor (2).")
        isotropic = False
    classes, equivalence_class, class_label, class_tensor \
        = equivalence_class_methods(dimension, lattice_shape, TI, isotropic)
    return sum( np.random.rand() * class_tensor(label) for label in classes )

##########################################################################################
# setting up the multibody eigenvalue problem
##########################################################################################

def multibody_problem(lattice_shape, sun_coefs, dimension, TI = None, isotropic = None):
    if dimension == 0:
        excitation_mat = np.zeros((1,1))
        def vector_to_tensor(vector): return vector[0] * np.ones(())
        def tensor_to_vector(tensor): return tensor * np.ones(1)
        return excitation_mat, vector_to_tensor, tensor_to_vector

    # compute total site numder and SU(n) coupling vector
    site_num = np.prod(lattice_shape)
    sun_coef_vec = sum(sun_coefs)

    # determine whether the SU(n) couplings are translationally invariant (TI)
    #   by checking whether sun_coef_vec is constant
    # note: strictly speaking, we are checking a necessary but not sufficient condition
    #       for TI, but sun_coefs must be very carefully tuned to satisfy
    #       this condition without actually being TI
    if TI is None:
        TI = np.allclose(sun_coef_vec/sun_coef_vec[0], np.ones(site_num))

    classes, equivalence_class, class_label, class_tensor \
        = equivalence_class_methods(dimension, lattice_shape, TI, isotropic)

    # get the index of the equivalence class of a choice of sites
    def get_class_idx(choice):
        return classes.get(class_label(tuple(sorted(choice))))

    # equivalence class sizes (multiplicities)
    mults = np.array([ len(equivalence_class(label)) for label in classes ])
    sqrt_mults = np.sqrt(mults)

    # build matrix for many-body eigenvalue problem
    def _diag_val(choice):
        return sum( sun_coefs[pp,qq] for pp in choice for qq in choice ) \
             - sum( sun_coef_vec[pp] for pp in choice )
    excitation_mat = np.diag([ _diag_val(choice) * mult
                               for choice, mult in zip(classes, mults) ])
    for label, idx in classes.items():
        for choice in equivalence_class(label):
            for pp in range(site_num):
                if pp in choice: continue
                for aa in range(dimension):
                    choice_aa = choice[aa]
                    choice_aa_pp = list(choice); choice_aa_pp[aa] = pp
                    choice_aa_pp_idx = get_class_idx(choice_aa_pp)
                    excitation_mat[idx,choice_aa_pp_idx] += sun_coefs[choice_aa,pp]
    excitation_mat /= np.outer(sqrt_mults, sqrt_mults)

    # convert between a tensor and a vector of equivalence class coefficients
    def vector_to_tensor(vector):
        return sum( val * class_tensor(label)
                    for val, label in zip(vector/sqrt_mults, classes) )
    def tensor_to_vector(tensor):
        return sqrt_mults * np.array([ tensor[label] for label in classes ])

    return excitation_mat, vector_to_tensor, tensor_to_vector

# get (1) shell indices with each multi-body excitation manifold,
#     (2) energies of each shell, and
#     (4) tensors that generate states in that shell
def get_multibody_states(lattice_shape, sun_coefs, manifolds, TI, isotropic = None,
                         updates = True):
    if type(manifolds) is int:
        _manifolds = range(manifolds+1)
    else:
        _manifolds = manifolds

    site_num = sun_coefs.shape[0]

    shell_num = 0
    manifold_shells = {}
    energies = {}
    tensors = {}
    for manifold in _manifolds:
        if updates:
            print(f"manifold, size: {manifold}, ", end = "")
        old_shell_num = shell_num
        excitation_mat, vector_to_tensor, tensor_to_vector \
            = multibody_problem(lattice_shape, sun_coefs, manifold, TI)
        if updates:
            print(excitation_mat.shape[0])
            sys.stdout.flush()

        eig_vals, eig_vecs = np.linalg.eigh(excitation_mat)
        for idx in np.argsort(eig_vals):
            eig_val = eig_vals[idx]
            tensor = vector_to_tensor(eig_vecs[:,idx])

            # if this eigenvalue matches an excitation energy of a fewer-body state,
            #   then check that the corresponding states are orthogonal
            # if these states are not orthogonal, then ignore this (redundant) state
            if any( np.allclose(eig_val, energies[shell]) and
                    not np.allclose(compute_overlap(tensors[shell], tensor, TI),
                                    np.zeros((site_num+1,)*2))
                    for shell in range(old_shell_num) ):
                continue

            energies[shell_num] = eig_val
            tensors[shell_num] = tensor
            shell_num += 1
            if updates:
                print("  shells:", shell_num)
                sys.stdout.flush()

        manifold_shells[manifold] = np.array(range(old_shell_num,shell_num), dtype = int)

    energies = np.array(list(energies.values()))
    return manifold_shells, energies, tensors

##########################################################################################
# methods for building "full" multibody states and operators
##########################################################################################

# build a fully symmetric state labelled by occupation number
def sym_state(occupations, site_num, site_dim = 2, normalize = True, sparse = False):
    if type(occupations) is int:
        occupations = (occupations,site_num-occupations)
    labels = [ [mm]*pop for mm, pop in enumerate(occupations) ]
    labels = np.concatenate(labels).astype(int)
    def _base_state(label):
        return unit_tensor((label,), site_dim)
    vec = sum( functools.reduce(np.kron, map(_base_state, perm))
               for perm in unique_permutations(labels) )
    if normalize:
        vec = vec / np.sqrt(vec @ vec)
    if not sparse:
        return vec
    else:
        from scipy import sparse
        return sparse.csr_matrix(vec).T

# act with the multi-local operator `op` on the spins indexed by `indices`
def embed_operator(op, indices, site_num, site_dim = 2):
    if not hasattr(indices, "__getitem__"):
        indices = list(indices)

    op = np.array(op)
    diag = ( np.prod(op.shape) == site_dim**len(indices) )

    # embed operator into a larger hilbert space
    aux_sites = site_num - len(indices)
    aux_identity = [1]*site_dim**aux_sites if diag else np.eye(site_dim**aux_sites)
    op = functools.reduce(np.kron, [ op ] + [ aux_identity ])
    op = np.reshape(op, (site_dim**( 1 if diag else 2 ),)*site_num)

    if not diag:
        # collect and flatten tensor factors associated with each spin
        fst_half = range(site_num)
        snd_half = range(site_num,2*site_num)
        perm = np.array(list(zip(list(fst_half),list(snd_half)))).flatten()
        op = np.reshape(op, (site_dim,)*2*site_num)
        op = np.transpose(op, perm)

    # rearrange tensor factors according to the desired qubit order
    old_order = list(indices) + [ jj for jj in range(site_num) if jj not in indices ]
    new_order = np.arange(site_num)[np.argsort(old_order)]
    op = np.transpose(op, new_order)

    if not diag:
        # un-flatten the tensor factors, and flatten the tensor into
        #   a matrix that acts on the joint Hilbert space of all qubits
        op_shape = (site_dim,)*site_num*( 1 if diag else 2 )
        op = np.reshape(op, op_shape)
        evens = range(0,2*site_num,2)
        odds = range(1,2*site_num,2)
        op = np.transpose(op, list(evens)+list(odds))

    mat_shape = (site_dim**site_num,)*( 1 if diag else 2 )
    return np.reshape(op, mat_shape)
