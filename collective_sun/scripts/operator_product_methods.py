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
            self.coefs = np.ones(len(self.diags)).astype(int)
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
            empty_sym = ( markers[2]+1 ) / markers[0] # symmetry factor

            # convert a filled dot into a cross
            cross_copy = diagram.copy()
            cross_copy[region] = ( markers[0]-1, markers[1]+1, markers[2] )
            cross_diag = reduce_diagram(cross_copy)
            cross_sym = 1 / markers[0] # symmetry factor

            return empty_sym * empty_diag - cross_sym * cross_diag

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

                symmetry_factor = new_diagram[joint_region][0]
                return symmetry_factor * reduce_diagram(new_diagram)

            return sum([ _take_from_region(other_region)
                         for other_region, other_markers in diagram.items()
                         if all( primary_set not in other_region
                                 for primary_set in region )
                         and other_markers[0] > 0 ])

##########################################################################################
# methods to multiply multi-body operators
##########################################################################################

# symmetrize a tensor operator under all permutations of its target spaces
def symmetrize_operator(oper):
    subsystems = oper.ndim//2
    def _permuted_oper(perm):
        return np.transpose(oper, perm + tuple( np.array(perm) + subsystems ) )
    perm_sum = sum( _permuted_oper(perm) for perm in it.permutations(range(subsystems)) )
    num_perms = np.math.factorial(subsystems)
    return perm_sum / num_perms

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
    dtype = type(np.product([ op.flatten()[0] for op in base_ops ]))
    base_ops = [ op.astype(dtype) for op in base_ops ]

    contraction = start_indices + "->" + final_indices
    return np.array( tf.einsum(contraction, *base_ops).numpy() )

# contract tensors according to a set/index diagram
# note: assumes translational invariance by default
def contract_tensors(tensors, diagram, TI = True):
    if len(diagram) == 0: return 1

    # cast all tensors into the appropriate data type
    dtype = type(np.product([ tensor[(0,)*tensor.ndim] for tensor in tensors ]))
    tensors = [ tensor.astype(dtype) for tensor in tensors ]

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
    symmetry_factor = np.prod([ np.math.factorial(points)
                                for points in diagram.values() ])
    return functools.reduce(operator.mul, contraction_factors) / symmetry_factor

# check whether a multi-local operator is diagonal
def _is_diagonal(operator):
    op_spins = operator.ndim//2 # number of spins operator acts on
    spin_dim = operator.shape[0] # dimension of each spin
    op_dim = spin_dim**op_spins # dimension of Hilbert space the operator acts on
    # reshape operator into a matrix of size (op_dim-1)x(op_dim+1),
    #   placing all (previously) diagonal elements (except for the last one)
    #   into the first column of the new matrix
    diagonal_test = operator.flatten()[:-1].reshape((op_dim-1,op_dim+1))
    # check whether the matrix is zero in all columns after the first
    return not np.any(diagonal_test[:,1:])

# find a matrix element of a multi-local operator
#   in the permutationally symmetric manifold
# TODO: make faster?
def evaluate_multi_local_op(local_op, pops_lft, pops_rht = None, diagonal = None):
    if pops_rht == None:
        pops_rht = pops_lft

    assert(len(pops_lft) == len(pops_rht))
    assert(sum(pops_lft) == sum(pops_rht))
    if local_op.ndim == 0: return pops_lft == pops_rht

    spin_dim = local_op.shape[0]
    op_spins = local_op.ndim//2

    if diagonal is None:
        diagonal = _is_diagonal(local_op)

    def _remainder(pops,assignment):
        return tuple( np.array(pops) - np.array(assignment) )

    def _state_idx(assignment):
        states = [ (jj,)
                   for jj in range(len(assignment))
                   for _ in range(assignment[jj]) ]
        return functools.reduce(operator.add, states)

    # determine overall normalization factor of the permutationally symmetric states
    if pops_lft == pops_rht:
        log_norm = np.log( float(multinomial(pops_lft)) )
    else:
        log_norm = 1/2 * ( np.log(float(multinomial(pops_lft))) +
                           np.log(float(multinomial(pops_rht))) )

    op_val = 0
    for assignment_lft in assignments(op_spins, spin_dim):

        remainder = _remainder(pops_lft, assignment_lft)
        if any( pop < 0 for pop in remainder ): continue

        assignment_rht = _remainder(pops_rht, remainder)
        if any( assignment < 0 for assignment in assignment_rht ): continue

        # TODO: compute remainder_perms in a better way?
        remainder_perms = np.exp( np.log(float(multinomial(remainder))) - log_norm )
        base_state_lft = _state_idx(assignment_lft)

        if diagonal:
            for state in unique_permutations(base_state_lft):
                op_val += remainder_perms * local_op[ state + state ]
        else:
            base_state_rht = _state_idx(assignment_rht)
            for state_lft, state_rht in it.product(unique_permutations(base_state_lft),
                                                   unique_permutations(base_state_rht)):
                op_val += remainder_perms * local_op[ state_lft + state_rht ]

    return op_val

# return two lists characterizing the operator content of a multi-local operator product.
# both lists are organized by "bare" diagrams containing filled dots,
#   i.e. the `dd`-th element of a list is associated with the `dd`-th bare diagram.
# elements of the first list are *reduced* diagram vectors
#   (associated with the appropriate bare diagram).
# elements of the second list are essentially maps
#   from a reduced diagram to a multi-local operator.
def diagram_operators(operators):
    dimensions = [ operator.ndim//2 for operator in operators ]
    return zip(*[ ( reduce_diagram(diagram), contract_ops(operators, diagram) )
                  for diagram in set_diagrams(dimensions) ])

# project the product of multi-body operators onto the permutationally symmetric manifold
# tensors: list of tensors (one for each multi-body operator)
# diagrams:  list of (reduced!) diagrams that define tensor contractions
# diagram_ops: matrix mapping diagram indices to operators on the many-body hilbert space
def evaluate_operator_product(tensors, diagrams, diagram_ops, TI = True):
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
def multi_Z_op(tensor_power):
    empty_op = np.ones(())
    Z = np.array([[1,0],[0,-1]])
    mat = functools.reduce(np.kron, [ Z ]*tensor_power + [ empty_op ])
    return mat.reshape((2,)*2*tensor_power)

# get appropriate data type to use, given a list of operators
def _get_dtype(operators):
    return type( np.product([ operator[(0,)*operator.ndim] for operator in operators ]) )

# get the diagrams and operators (for each diagram) necessary
#   to evaluate products of ZZ operators
def get_diags_opers(shell_dims, spin_num, overlap_operators = None, dtype = None):
    # build operators that generate excited states
    excitation_op = { shell_dim : multi_Z_op(shell_dim)
                      for shell_dim in shell_dims }

    # build diagram vectors and associated operators
    diags, opers = {}, {}
    if overlap_operators is None:
        for shell_dim in shell_dims:
            diags[shell_dim], diag_ops = diagram_operators([excitation_op[shell_dim]]*2)

            opers[shell_dim] = np.zeros((len(diags[shell_dim]), spin_num+1))
            for spins_up in range(spin_num+1):
                spins_dn = spin_num - spins_up
                populations = ( spins_up, spins_dn )
                opers[shell_dim][:,spins_up] \
                    = [ evaluate_multi_local_op(local_op, populations, diagonal = True)
                        for local_op in diag_ops ]
    else:
        # determine the appropriate data type to use
        if dtype is None:
            dtype = _get_dtype(overlap_operators)

        # given a number of spins "up" in the state to the left of an operator,
        #   determine the range for spins "up" in the state to the right of an operator
        if any( not _is_diagonal(op) for op in overlap_operators ):
            def _spins_up_rht_range(spins_up_lft): return range(spins_up_lft, spin_num+1)
        else:
            def _spins_up_rht_range(spins_up_lft): return [ spins_up_lft ]

        shell_dims = sorted(shell_dims)
        for min_dim_idx, dim_min in enumerate(shell_dims):
            for dim_max in shell_dims[min_dim_idx:]:
                dim_pair = dim_min, dim_max
                op_lft = excitation_op[dim_min]
                op_rht = excitation_op[dim_max]
                diags[dim_pair], diag_ops \
                    = diagram_operators([ op_lft ] + overlap_operators + [ op_rht ])

                shape = ( len(diag_ops), spin_num+1, spin_num+1 )
                opers[dim_pair] = np.zeros(shape, dtype = dtype)

                for spins_up_lft in range(spin_num+1):
                    spins_dn_lft = spin_num - spins_up_lft
                    pops_lft = ( spins_up_lft, spins_dn_lft )
                    for spins_up_rht in _spins_up_rht_range(spins_up_lft):
                        spins_dn_rht = spin_num - spins_up_rht
                        pops_rht = ( spins_up_rht, spins_dn_rht )

                        opers[dim_pair][:,spins_up_lft,spins_up_rht] \
                            = [ evaluate_multi_local_op(local_op, pops_lft, pops_rht)
                                for local_op in diag_ops ]

                        if spins_up_rht != spins_up_lft:
                            opers[dim_pair][:,spins_up_rht,spins_up_lft] \
                                = np.conj(opers[dim_pair][:,spins_up_lft,spins_up_rht])
    return diags, opers

# compute overlap between two states generated by Z-type operators
def compute_overlap(tensor_lft, tensor_rht):
    tensors = ( tensor_lft, tensor_rht )
    shell_dims = ( tensor_lft.ndim, tensor_rht.ndim )
    if shell_dims == ( 0, 0 ): return np.array(1)

    try:
        spin_num = tensor_lft.shape[0]
    except:
        spin_num = tensor_rht.shape[0]

    diags, opers = get_diags_opers(shell_dims, spin_num, [])

    if shell_dims != tuple(sorted(shell_dims)):
        tensors = tensors[::-1]
        shell_dims = shell_dims[::-1]
    overlap = evaluate_operator_product(tensors, diags[shell_dims], opers[shell_dims])

    def _norms_sqr(tensor):
        shell_dim = tensor.ndim
        _tensors = (tensor,) * 2
        _diags = diags[shell_dim,shell_dim]
        _opers = opers[shell_dim,shell_dim]
        return evaluate_operator_product(_tensors, _diags, _opers)
    norms_sqr = np.product([ _norms_sqr(tensor) for tensor in tensors ], axis = 0)
    norms_sqr[np.isclose(norms_sqr,np.zeros(spin_num+1))] = 1

    return overlap / np.sqrt(norms_sqr)

# compute norms of generated states in the Z-projection / shell basis
def compute_norms(sunc):
    # determine number of spins, shells, and dimensions of excitation tensors
    spin_num = sunc["mat"].shape[0]
    shell_num = max( shell for shell in sunc.keys() if type(shell) is int ) + 1
    shell_dims = set( sunc[shell].ndim for shell in range(shell_num) )

    # get the diagrams and operators (for each diagram) necessary
    #   to evaluate the relevant products of ZZ operators
    diags, opers = get_diags_opers(shell_dims, spin_num)

    # compute norms, setting the norms of non-existant states to 1
    norms = np.ones((spin_num+1, shell_num))
    for shell in range(shell_num):
        shell_tensor = sunc[shell]
        shell_dim = shell_tensor.ndim
        _tensors = [ shell_tensor, shell_tensor ]
        _diags = diags[shell_dim]
        _opers = opers[shell_dim]
        _norms_sqr = evaluate_operator_product(_tensors, _diags, _opers)

        # the `shell_dim` top/bottom-most projections do not have any shells,
        #   so ignore the corresponding norms
        norms[shell_dim:-shell_dim,shell] = np.sqrt(_norms_sqr[shell_dim:-shell_dim])

    return norms

# construct a product of operators in the shell basis
def build_shell_operator(tensors, operators, sunc, sunc_norms = {}, TI = True):
    # determine number of spins, shells, and dimensions of excitation tensors
    spin_num = sunc["mat"].shape[0]
    shell_num = max( shell for shell in sunc.keys() if type(shell) is int ) + 1
    shell_dims = set( sunc[shell].ndim for shell in range(shell_num) )

    # compute norms of generated states in the Z-projection / shell basis
    sunc_hash = sunc["mat"].tobytes()
    norms = sunc_norms.get(sunc_hash)
    if norms is None:
        norms = compute_norms(sunc)
        sunc_norms[sunc_hash] = norms

    # get the diagrams and operators (for each diagram) necessary
    #   to evaluate the relevant products of ZZ operators
    dtype = _get_dtype(tensors + operators)
    diags, opers = get_diags_opers(shell_dims, spin_num, operators, dtype)

    # start building the full operator in the Z-projection / shell basis
    shape = ( spin_num+1, shell_num, spin_num+1, shell_num )
    shell_operator = np.zeros(shape, dtype = dtype)
    for shell_lft in range(shell_num):
        for shell_rht in range(shell_lft,shell_num):
            dim_lft = sunc[shell_lft].ndim
            dim_rht = sunc[shell_rht].ndim
            dim_pair = tuple(sorted([dim_lft,dim_rht]))
            if dim_pair != (dim_lft,dim_rht):
                shell_lft, shell_rht = shell_rht, shell_lft

            _tensors = [ sunc[shell_lft] ] + tensors + [ sunc[shell_rht] ]

            product_args = ( _tensors, diags[dim_pair], opers[dim_pair] )
            product = evaluate_operator_product(*product_args, TI)
            shell_norms = np.outer(norms[:,shell_lft], norms[:,shell_rht])

            shell_operator[:,shell_lft,:,shell_rht] \
                = shell_operator[:,shell_rht,:,shell_lft] \
                = product / shell_norms

    return shell_operator
