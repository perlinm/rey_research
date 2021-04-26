#!/usr/bin/env python3

import numpy as np
import itertools as it
import tensorflow as tf
import networkx as nx
import functools, operator

from itertools_extension import multinomial, unique_permutations, \
    assignments, set_diagrams

_chars = "abcdefghijklmnopqrstuvwxyz"
_chars += _chars.upper()
def unique_char_lists(lengths):
    assert( sum(lengths) <= len(_chars) )
    chars = []
    chars_used = 0
    for length in lengths:
        chars += [ list(_chars[chars_used:chars_used+length]) ]
        chars_used += length
    return chars

##########################################################################################
# methods to manipulate diagrams
##########################################################################################

def _hash_dict(dict): return frozenset(sorted(dict.items()))

class diagram_vec:
    def __init__(self, diagrams = None, coefficients = None):
        if diagrams is None:
            self.diags = []
        elif type(diagrams) is list:
            self.diags = diagrams
        elif type(diagrams) is dict:
            self.diags = [ diagrams ]
        else:
            print("diagram cannot be initialized with type:", type(diagrams))

        if coefficients is None:
            self.coefs = np.ones(len(self.diags), dtype = int)
        elif hasattr(coefficients, "__getitem__"):
            assert(len(coefficients) == len(self.diags))
            self.coefs = np.array(coefficients)
        else:
            self.coefs = np.array([ coefficients ])

    def __repr__(self):
        return "\n".join([ f"{coef} : {diag}"
                           for coef, diag in zip(self.coefs, self.diags) ])

    def __str__(self): return self.__repr__()

    def __add__(self, other):
        if other == 0 or other == None: return self
        new_diags = self.diags
        new_coefs = self.coefs
        for other_diag, other_coef in zip(other.diags, other.coefs):
            try:
                diag_idx = new_diags.index(other_diag)
                new_coefs[diag_idx] += other_coef
            except:
                new_diags += [ other_diag ]
                new_coefs = np.append(new_coefs, other_coef)
        new_diags = [ diag for dd, diag in enumerate(new_diags) if new_coefs[dd] != 0 ]
        return diagram_vec(new_diags, new_coefs[new_coefs != 0])
    def __radd__(self, other):
        return self + other
    def __sub__(self, other):
        if other == 0: return self
        return self + diagram_vec(other.diags, -other.coefs)

    def __rmul__(self, scalar):
        return diagram_vec(self.diags, scalar * self.coefs)
    def __mul__(self, scalar):
        return scalar * self
    def __truediv__(self, scalar):
        return 1/scalar * self

    def __pos__(self): return self
    def __neg__(self): return -1 * self

    def reduce(self):
        return sum( coef * reduce_diagram(diag)
                    for coef, diag in zip(self.coefs, self.diags) )

    def join_permutations(self):
        new_diags = []
        new_coefs = []
        for diag, coef in zip(self.diags, self.coefs):
            labels = set.union(*( set(region) for region in diag.keys() ))
            eliminated_diag = False
            for perm in it.permutations(labels):
                perm_diag = permute_diagram(diag, perm)
                for new_diag, new_coef in zip(new_diags, new_coefs):
                    if new_diag == perm_diag:
                        assert( new_coef == coef )
                        eliminated_diag = True
                        break
                if eliminated_diag: break
            if not eliminated_diag:
                new_diags += [ diag ]
                new_coefs += [ coef ]
        new_diags = [ diag for dd, diag in enumerate(new_diags) if new_coefs[dd] != 0 ]
        new_coefs = [ coef for coef in new_coefs if coef != 0 ]
        return diagram_vec(new_diags, new_coefs)

# permute the labels on a diagram
def permute_diagram(diagram, permutation):
    return { tuple(sorted( permutation[idx] for idx in region )) : markers
             for region, markers in diagram.items() }

# reduce a diagram by eliminating filled dots and crosses
def reduce_diagram(diagram):
    # enforce three types of markers in each region: ( filled dots, crosses, empty dots )
    diagram = { region : markers if type(markers) is tuple else (markers,0,0)
                for region, markers in diagram.items()
                if markers != 0 and markers != (0,0,0) }

    # are there any markers of type `mm` in the diagram?
    def _any_markers(mm):
        return any( markers[mm] for region, markers in diagram.items() )

    # first, cover the simpler case of a diagram without any crosses,
    #        in which we eliminate any filled dots that we find
    if not _any_markers(1):
        # if there are no filled dots, then we are done simplifying
        if not _any_markers(0):
            empty_dot_diagram = { region : markers[2]
                                  for region, markers in diagram.items() }
            return diagram_vec(empty_dot_diagram)

        # otherwise, eliminate a filled dot (from anywhere)
        for region, markers in diagram.items():
            if markers[0] == 0: continue

            # convert a filled dot into an empty dot
            empty_copy = diagram.copy()
            empty_copy[region] = ( markers[0]-1, markers[1], markers[2]+1 )
            empty_diag = reduce_diagram(empty_copy)

            # convert a filled dot into a cross
            cross_copy = diagram.copy()
            cross_copy[region] = ( markers[0]-1, markers[1]+1, markers[2] )
            cross_diag = reduce_diagram(cross_copy)

            return empty_diag - cross_diag

    # otherwise, eliminate any crosses in a diagram
    else:
        for region, markers in diagram.items():
            if markers[1] == 0: continue

            def _take_from_region(other_region):
                other_markers = diagram[other_region]
                if other_markers[0] == 0: return 0

                # eliminate a cross from the "original" region,
                #   and a filled dot from the "other" region
                new_diagram = diagram.copy()
                new_diagram[region] = ( markers[0], markers[1]-1, markers[2] )
                new_diagram[other_region] = ( other_markers[0]-1, ) + other_markers[-2:]

                # add a filled dot to the "joint" region
                joint_region = tuple(sorted( region + other_region ))
                joint_markers = new_diagram.get(joint_region)
                if joint_markers == None: joint_markers = (0,0,0)
                new_diagram[joint_region] = ( joint_markers[0]+1, ) + joint_markers[-2:]

                multiplicity = diagram[other_region][0]
                return multiplicity * reduce_diagram(new_diagram)

            return sum([ _take_from_region(other_region)
                         for other_region, other_markers in diagram.items()
                         if all( primary_set not in other_region
                                 for primary_set in region )
                         and other_markers[0] > 0 ])

##########################################################################################
# methods to multiply multi-body operators
##########################################################################################

# get appropriate data type to use, given a list of operators
def _get_dtype(operators):
    return type( np.product([ operator.ravel()[0] for operator in operators ]) )

# contract operators according to a set diagram
def contract_ops(base_ops, diagram):
    num_ops = len(base_ops)
    targets = [ op.ndim//2 for op in base_ops ]
    targets_used = [ 0 ] * num_ops

    indices = unique_char_lists([ op.ndim for op in base_ops ])
    final_out_indices = ""
    final_inp_indices = ""
    for choice, shared_targets in diagram.items():
        for _ in range(shared_targets):
            for fst_op, snd_op in zip(choice[:-1], choice[1:]):
                indices[snd_op][targets_used[snd_op]] \
                    = indices[fst_op][targets[fst_op]+targets_used[fst_op]]

            fst_op, lst_op = choice[0], choice[-1]
            final_out_indices += indices[fst_op][targets_used[fst_op]]
            final_inp_indices += indices[lst_op][targets[lst_op]+targets_used[lst_op]]

            for op in choice:
                targets_used[op] += 1

    start_indices = ",".join([ "".join(idx) for idx in indices ])
    final_indices = final_out_indices + final_inp_indices

    # cast all base operators into the appropriate data type
    dtype = _get_dtype(base_ops)
    base_ops = [ base_op.astype(dtype, copy = False) for base_op in base_ops ]

    contraction = start_indices + "->" + final_indices
    return np.array( tf.einsum(contraction, *base_ops).numpy() )

# contract tensors according to a set/index diagram
# note: assumes that all tensors are permutationally symmetric
def contract_tensors(tensors, diagram, TI):
    if len(diagram) == 0: return 1

    # cast all tensors into the appropriate data type
    dtype = _get_dtype(tensors)
    tensors = [ tensor.astype(dtype, copy = False) for tensor in tensors ]

    # assign contraction indices to each tensor
    indices_used = [ 0 ] * len(tensors)
    num_indices = [ tensor.ndim for tensor in tensors ]
    full_contraction = unique_char_lists(num_indices)
    for choice, shared_indices in diagram.items():
        fst_ten = choice[0]
        for _ in range(shared_indices):
            _this_idx = full_contraction[fst_ten][indices_used[fst_ten]]
            for snd_ten in choice[1:]:
                full_contraction[snd_ten][indices_used[snd_ten]] = _this_idx
            for tensor in choice:
                indices_used[tensor] += 1

    # convert tensor diagram into an index graph
    index_graph = nx.Graph()
    index_graph.add_nodes_from(range(len(tensors)))
    for region, shared_indices in diagram.items():
        if shared_indices == 0: continue
        for ten_1, ten_2 in it.combinations(region,2):
            index_graph.add_edge(ten_1, ten_2)

    # group tensors and contractions
    group_tensors = []
    group_contractions = []
    for tensor_group in nx.connected_components(index_graph):
        _tensors = [ tensors[idx] for idx in tensor_group ]
        if len(_tensors) == 1 and _tensors[0].ndim == 0: continue
        _contractions = [ full_contraction[idx] for idx in tensor_group ]
        group_tensors += [ _tensors ]
        group_contractions += [ _contractions ]

    def _contract(tensors, contraction, eliminate_index = TI):
        if not eliminate_index:
            contraction = ",".join([ "".join(idx) for idx in contraction ]) + "->"
            return tf.einsum(contraction, *tensors).numpy()
        else:
            spin_num = tensors[0].shape[0]
            fixed_index = contraction[0][0]
            new_tensors = []
            new_contraction = []
            for indices, tensor in zip(contraction, tensors):
                if fixed_index in indices:
                    fixed_axis = indices.index(fixed_index)
                    other_axes = list( idx for idx in range(tensor.ndim)
                                       if idx != fixed_axis )
                    tensor = tensor.transpose( [fixed_axis] + other_axes )[0]
                    indices.remove(fixed_index)
                new_tensors += [ tensor ]
                new_contraction += [ indices ]
            return spin_num * _contract(new_tensors, new_contraction, False)

    contraction_factors = ( _contract(tensors, contraction)
                            for tensors, contraction
                            in zip(group_tensors, group_contractions) )
    return functools.reduce(operator.mul, contraction_factors)

# check whether a multi-local operator is diagonal
def _is_diagonal(operator):
    op_spins = operator.ndim//2 # number of spins operator acts on
    spin_dim = operator.shape[0] # dimension of each spin
    op_dim = spin_dim**op_spins # dimension of Hilbert space the operator acts on
    # reshape operator into a matrix of size (op_dim-1)x(op_dim+1),
    #   placing all (previously) diagonal elements (except for the last one)
    #   into the first column of the new matrix
    diagonal_test = np.array(operator).ravel().flatten()[:-1].reshape((op_dim-1,op_dim+1))
    # check whether the matrix is zero in all columns after the first
    return not np.any(diagonal_test[:,1:])

# build a multi-local operator in the permutationally symmetric manifold
def build_multi_local_op(spin_num, spin_dim, local_op,
                         diagonal = None, diagonal_values = False,
                         collective_states = {}):

    dim_PS = np.math.comb(spin_num+spin_dim-1,spin_dim-1)

    # determine dimension of each spin, and total number of spins
    if op_spins := round( np.math.log(np.sqrt(local_op.size), spin_dim) ):
        local_op = local_op.reshape((spin_dim,)*2*op_spins)

    else: # local_op is a scalar
        if diagonal_values:
            return local_op * np.ones(dim_PS)
        else:
            return local_op * np.eye(dim_PS)

    if op_spins > spin_num:
        if diagonal_values:
            return np.zeros(dim_PS)
        else:
            return np.zeros((dim_PS,)*2)

    # determine whether this operator is diagonal
    if diagonal is None:
        diagonal = _is_diagonal(local_op)

    # determine the population differences (i.e. transitions) induced by this operator
    if diagonal:
        transitions = set([ (0,)*spin_dim ])

    else:
        nonzero_indices = local_op.nonzero()
        states_lft = np.array(nonzero_indices[:op_spins]).T
        states_rht = np.array(nonzero_indices[op_spins:]).T

        def counts(state):
            return np.array([ np.count_nonzero(state == mu) for mu in range(spin_dim) ])
        transitions = set( tuple( counts(state_lft) - counts(state_rht) )
                           for state_lft, state_rht in zip(states_lft, states_rht) )

    # shift a permutationally symmetric state by a given transition
    def shift_state(state, transition):
        new_state = np.array(state) + np.array(transition)
        if any( new_state < 0 ): return None
        return tuple(new_state)

    # identify bases for the relevant permutationally symmetric manifolds
    for num in [ spin_num, op_spins ]:
        if ( num, spin_dim ) not in collective_states:
            collective_states[num,spin_dim] \
                = { state : dim_PS - 1 - idx
                    for idx, state in enumerate(assignments(num, spin_dim)) }

    def state_pairs(num):
        return ( (state_lft, state_rht)
                 for state_rht in collective_states[num,spin_dim]
                 for transition in transitions
                 if (state_lft := shift_state(state_rht, transition)) )

    def base_state(assignment):
        states = [ (jj,)
                   for jj in range(len(assignment))
                   for _ in range(assignment[jj]) ]
        return functools.reduce(operator.add, states)

    # compute matrix elements of the permutation-symmetrized local operator
    #   multiplied by a multinomial factor
    local_op_sym_mult = {}
    for pops_lft, pops_rht in state_pairs(op_spins):
        base_lft = base_state(pops_lft)
        base_rht = base_state(pops_rht)

        if diagonal:
            local_op_sym_mult[pops_lft,pops_rht] \
                = sum( local_op[ state + state ]
                       for state in unique_permutations(base_rht) )
        else:
            local_op_sym_mult[pops_lft,pops_rht] \
                = sum( local_op[ state_lft + state_rht ]
                       for state_lft, state_rht
                       in it.product(unique_permutations(base_lft),
                                     unique_permutations(base_rht)) )

    # get a matrix element of the full operator
    def get_matrix_element(state_full_lft, state_full_rht, elem = {}):
        state_full_lft = np.array(state_full_lft)
        state_full_rht = np.array(state_full_rht)
        val = 0
        log_coef_full = sum( np.log(multinomial(state))
                             for state in [ state_full_lft, state_full_rht ] )
        state_full_diff = state_full_rht - state_full_lft
        for state_locl_lft in map(np.array, assignments(op_spins, spin_dim)):
            state_diff_lft = state_full_lft - state_locl_lft
            if any( state_diff_lft < 0 ): continue
            state_locl_rht = state_full_diff + state_locl_lft
            if any( state_locl_rht < 0 ): continue
            state_diff_rht = state_full_rht - state_locl_rht
            if any( state_diff_rht < 0 ): continue

            log_coef_locl = sum( np.log(multinomial(state))
                                 for state in [ state_diff_lft, state_diff_rht ] )
            coef = np.exp( ( log_coef_locl - log_coef_full ) / 2 )
            val += coef * local_op_sym_mult[tuple(state_locl_lft),tuple(state_locl_rht)]

        return val

    # compute and return the full operator
    if diagonal_values:
        assert(diagonal)
        return np.array([ get_matrix_element(state, state)
                          for state in collective_states[spin_num,spin_dim] ])
    else:
        dtype = complex if np.iscomplexobj(local_op) else float
        full_op = np.zeros((dim_PS,)*2, dtype = dtype)
        for state_lft, state_rht in state_pairs(spin_num):
            idx_lft = collective_states[spin_num,spin_dim][state_lft]
            idx_rht = collective_states[spin_num,spin_dim][state_rht]
            full_op[idx_lft,idx_rht] = get_matrix_element(state_lft, state_rht)
        return full_op

# return two lists characterizing the operator content of a multi-local operator product.
# both lists are organized by "bare" diagrams containing filled dots,
#   i.e. the `dd`-th element of a list is associated with the `dd`-th bare diagram.
# elements of the first list are *reduced* diagram vectors
#   (associated with the appropriate bare diagram).
# elements of the second list are essentially maps
#   from a reduced diagram to a multi-local operator.
def diagram_operators(operators):
    dimensions = [ operator.ndim//2 for operator in operators ]

    # number of distinct index assignments consistent with a given diagram
    def index_assignments(diagram):
        num = np.prod([ np.math.factorial(dimension) for dimension in dimensions ])
        den = np.prod([ np.math.factorial(points) for points in diagram.values() ])
        return num / den

    return zip(*[ ( reduce_diagram(diagram),
                    index_assignments(diagram) * contract_ops(operators, diagram) )
                  for diagram in set_diagrams(dimensions) ])

# project the product of multi-body operators onto the permutationally symmetric manifold
# tensors: list of tensors (one for each multi-body operator)
# diagrams:  list of (reduced!) diagrams that define tensor contractions
# diagram_ops: matrix mapping diagram indices to operators on the many-body hilbert space
def evaluate_operator_product(tensors, diagrams, diagram_ops, TI):
    assert(len(diagrams) == diagram_ops.shape[0])

    # evaluate all distinct diagrams
    diagram_vals = {}
    for diagram in diagrams:
        for diag in diagram.diags:
            _hash = _hash_dict(diag)
            if _hash not in diagram_vals:
                diagram_vals[_hash] = contract_tensors(tensors, diag, TI)
    def _evaluate(diag):
        return diagram_vals[_hash_dict(diag)]

    # collect scalar coefficients for each diagram
    coefficients = np.array([ np.dot(diagram.coefs, list(map(_evaluate, diagram.diags)))
                              for diagram in diagrams ])

    return np.tensordot(coefficients, diagram_ops, axes = 1)

# Z-type multi-local operator
def multi_Z_op(tensor_power, diag = False):
    Z_op = np.array([ -1, +1 ])
    if not diag: Z_op = np.diag(Z_op)
    empty_op = np.ones(())
    operator = functools.reduce(np.kron, [ Z_op ]*tensor_power + [ empty_op ])
    operator.shape = (2,)*tensor_power*Z_op.ndim
    return operator

def _conj_op(operator):
    dim = operator.ndim//2
    return operator.transpose(list(range(dim,2*dim)) + list(range(dim))).conj()

# get the diagrams and operators (for each diagram) necessary
#   to evaluate products of multi-local operators
def get_diags_opers(shell_dims, spin_num, overlap_operators = None, dtype = None,
                    spin_dim = 2):
    # build operators that generate excited states
    excitation_op = { shell_dim : multi_Z_op(shell_dim)
                      for shell_dim in shell_dims }

    diags, opers = {}, {}
    if overlap_operators is None:
        for shell_dim in shell_dims:
            op = excitation_op[shell_dim]
            diags[shell_dim], diag_ops = diagram_operators([ _conj_op(op), op ])

            opers[shell_dim] \
                = np.array([ build_multi_local_op(spin_num, spin_dim, local_op,
                                                  diagonal_values = True)
                             for local_op in diag_ops ])

    else:
        if dtype is None:
            dtype = _get_dtype(overlap_operators)

        shell_dims = sorted(shell_dims)
        dim_pairs = ( ( dim_min, dim_max )
                      for min_dim_idx, dim_min in enumerate(shell_dims)
                      for dim_max in shell_dims[min_dim_idx:] )
        for dim_pair in dim_pairs:
            dim_min, dim_max = dim_pair
            op_lft = excitation_op[dim_min]
            op_rht = excitation_op[dim_max]
            op_list = [ _conj_op(op_lft) ] + overlap_operators + [ op_rht ]
            diags[dim_pair], diag_ops = diagram_operators(op_list)

            opers[dim_pair] \
                = np.array([ build_multi_local_op(spin_num, spin_dim, local_op)
                             for local_op in diag_ops ])

    return diags, opers

# compute overlap between two states generated by Z-type operators
def compute_overlap(tensor_lft, tensor_rht, TI, normalize = True):
    tensors = ( tensor_lft.conj(), tensor_rht )
    shell_dims = ( tensor_lft.ndim, tensor_rht.ndim )
    if shell_dims == ( 0, 0 ): return np.array([[1]])

    try:
        spin_num = tensor_lft.shape[0]
    except:
        spin_num = tensor_rht.shape[0]

    diags, opers = get_diags_opers(shell_dims, spin_num, [])

    if shell_dims != tuple(sorted(shell_dims)):
        tensors = tensors[::-1]
        shell_dims = shell_dims[::-1]
    overlap = evaluate_operator_product(tensors, diags[shell_dims], opers[shell_dims], TI)

    if normalize:
        def _norms_sqr(tensor):
            shell_dim = tensor.ndim
            _tensors = ( tensor.conj(), tensor )
            _diags = diags[shell_dim,shell_dim]
            _opers = opers[shell_dim,shell_dim]
            return evaluate_operator_product(_tensors, _diags, _opers, TI)
        norms_sqr = np.product([ _norms_sqr(tensor) for tensor in tensors ], axis = 0)
        norms_sqr[np.isclose(overlap,np.zeros(spin_num+1))] = 1
        overlap /= np.sqrt(norms_sqr)

    return np.array(overlap, ndmin = 2)

# compute norms of generated states in the shell / Z-projection basis
def compute_norms(sunc, TI):
    # determine number of spins, shells, and dimensions of excitation tensors
    spin_num = sunc["mat"].shape[0]
    shell_num = max( shell for shell in sunc.keys() if type(shell) is int ) + 1
    shell_dims = set( sunc[shell].ndim for shell in range(shell_num) )

    # get the diagrams and operators (for each diagram) necessary
    #   to evaluate the relevant products of ZZ operators
    diags, opers = get_diags_opers(shell_dims, spin_num)

    # compute all norms
    norms = np.zeros((shell_num, spin_num+1))
    for shell in range(shell_num):
        shell_tensor = sunc[shell]
        shell_dim = shell_tensor.ndim
        _tensors = [ shell_tensor.conj(), shell_tensor ]
        _diags = diags[shell_dim]
        _opers = opers[shell_dim]
        _norms_sqr = evaluate_operator_product(_tensors, _diags, _opers, TI)
        assert(np.allclose(_norms_sqr.imag, np.zeros(_norms_sqr.shape)))
        _norms_sqr = _norms_sqr.real

        # set the norms of non-existant states to 1
        _norms_sqr[np.isclose(_norms_sqr, np.zeros(_norms_sqr.shape))] = 1
        norms[shell,:] = np.sqrt(_norms_sqr)

    return norms

# construct a product of operators in the shell basis
# todo: simplify for operators diagonal in the population index
def build_shell_operator(tensors, operators, sunc, TI, sunc_norms = {},
                         collective = False, shell_diagonal = False):
    # determine number of spins, shells, and dimensions of excitation tensors
    spin_num = sunc["mat"].shape[0]
    shell_num = max( shell for shell in sunc.keys() if type(shell) is int ) + 1
    shell_dims = set( sunc[shell].ndim for shell in range(shell_num) )

    # compute norms of generated states in the shell / Z-projection basis
    sunc_hash = sunc["mat"].tobytes()
    norms = sunc_norms.get(sunc_hash)
    if norms is None:
        norms = compute_norms(sunc, TI)
        sunc_norms[sunc_hash] = norms

    # get the diagrams and operators (for each diagram) necessary
    #   to evaluate the relevant products of ZZ operators
    shell_tensors = [ sunc[shell] for shell in range(shell_num) ]
    dtype = _get_dtype(tensors + operators + shell_tensors)
    diags, opers = get_diags_opers(shell_dims, spin_num, operators, dtype)

    # determine the shell matrix elements that we need to compute
    if shell_diagonal:
        shell_pairs = ( ( shell, shell ) for shell in range(shell_num) )

    else:
        if collective:
            max_manifold_change = 0
        else:
            max_manifold_change = sum( tensor.ndim for tensor in tensors )

        manifolds = list(sunc["shells"].keys())
        def _manifolds_rht(manifold_lft_idx):
            manifold_lft = manifolds[manifold_lft_idx]
            return ( manifold for manifold in manifolds[manifold_lft_idx:]
                     if manifold - manifold_lft <= max_manifold_change )
        shell_pairs = it.chain.from_iterable(
            ( shell_pair for shell_pair in it.product(sunc["shells"][manifold_lft],
                                                      sunc["shells"][manifold_rht])
              if shell_pair[0] <= shell_pair[1] )
            for manifold_lft_idx, manifold_lft in enumerate(manifolds)
            for manifold_rht in _manifolds_rht(manifold_lft_idx) )

    # start building the full operator in the shell / Z-projection basis
    shape = ( shell_num, spin_num+1, shell_num, spin_num+1 )
    full_operator = np.zeros(shape, dtype = dtype)
    for shell_pair in shell_pairs:
        shell_lft, shell_rht = shell_pair
        dim_lft = sunc[shell_lft].ndim
        dim_rht = sunc[shell_rht].ndim
        dim_pair = tuple(sorted([dim_lft,dim_rht]))
        if dim_pair != (dim_lft,dim_rht):
            shell_lft, shell_rht = shell_rht, shell_lft

        _tensors = [ sunc[shell_lft].conj() ] + tensors + [ sunc[shell_rht] ]

        product_args = ( _tensors, diags[dim_pair], opers[dim_pair], TI )
        product = evaluate_operator_product(*product_args)
        shell_norms = np.outer(norms[shell_lft,:], norms[shell_rht,:])

        product[np.isclose(product,np.zeros(product.shape))] = 0
        full_operator[shell_lft,:,shell_rht,:] \
            = full_operator[shell_rht,:,shell_lft,:] \
            = product / shell_norms

    return full_operator
