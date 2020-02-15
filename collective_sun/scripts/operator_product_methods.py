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
    diagram = { region : markers if type(markers) is tuple else (markers,0,0)
                for region, markers in diagram.items()
                if markers != 0 and markers != (0,0,0) }

    def _net_markers(mm):
        return sum([ markers[mm] for region, markers in diagram.items() ])

    # first, cover the simpler case of a diagram without any crosses,
    #        in which we eliminate any filled dots that we find
    if _net_markers(2) == 0:
        # if there are no crosses and no filled dots, then we are done simplifying
        if _net_markers(0) == 0:
            empty_dot_diagram = { region : markers[1]
                                  for region, markers in diagram.items() }
            return diagram_vec(empty_dot_diagram)

        for region, markers in diagram.items():
            if markers[0] == 0: continue

            empty_copy = diagram.copy()
            empty_copy[region] = ( markers[0]-1, markers[1]+1, markers[2] )
            empty_diag = reduce_diagram(empty_copy)
            empty_sym = ( markers[1]+1 ) / markers[0]

            cross_copy = diagram.copy()
            cross_copy[region] = ( markers[0]-1, markers[1], markers[2]+1 )
            cross_diag = reduce_diagram(cross_copy)
            cross_sym = 1 / markers[0]

            return empty_sym * empty_diag - cross_sym * cross_diag

    # otherwise, eliminate any crosses in a diagram
    else:
        for region, markers in diagram.items():
            if markers[2] == 0: continue

            def _take_from_region(other_region):
                other_markers = diagram[other_region]
                if other_markers[0] == 0: return 0

                new_diagram = diagram.copy()
                new_diagram[region] = markers[:2] + ( markers[2]-1, )
                new_diagram[other_region] = ( other_markers[0]-1, ) + other_markers[-2:]

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
    num_perms = np.math.factorial(subsystems)
    return sum( _permuted_oper(perm)
                for perm in it.permutations(range(subsystems)) ) / num_perms

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
            for ff, snd_op in enumerate(choice[1:]):
                fst_op = choice[ff]
                indices[snd_op][targets_used[snd_op]] \
                    = indices[fst_op][targets[fst_op]+targets_used[fst_op]]

            fst_op, lst_idx = choice[0], choice[-1]
            final_out_indices += indices[fst_op][targets_used[fst_op]]
            final_inp_indices += indices[lst_idx][targets[lst_idx]+targets_used[lst_idx]]

            for op in choice:
                targets_used[op] += 1

    start_indices = ",".join([ "".join(idx) for idx in indices ])
    final_indices = final_out_indices + final_inp_indices

    contraction = start_indices + "->" + final_indices
    return tf.einsum(contraction, *base_ops).numpy()

# contract tensors according to a set/index diagram
# note: assumes translational invariance by default
def contract_tensors(tensors, diagram, TI = True):
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
        if shared_indices == 0:
            del diagram[region]
        else:
            for ten_1, ten_2 in it.combinations(region,2):
                index_graph.add_edge(ten_1, ten_2)

    # group tensors and contractions
    group_tensors = []
    group_contractions = []
    for tensor_group in nx.connected_components(index_graph):
        group_tensors += [ [ tensors[idx] for idx in tensor_group ] ]
        group_contractions += [ [ full_contraction[idx] for idx in tensor_group ] ]

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

# find a matrix element of a multi-local operator
#   in the permutationally symmetric manifold
# TODO: make faster
def evaluate_multi_local_op(local_op, pops_lft, pops_rht = None, diagonal = None):
    if pops_rht == None:
        pops_rht = pops_lft

    assert(len(pops_lft) == len(pops_rht))
    assert(sum(pops_lft) == sum(pops_rht))
    spin_dim = len(pops_lft)
    spin_num = sum(pops_lft)
    op_spins = local_op.ndim//2

    if diagonal is None:
        op_dim = spin_dim**op_spins
        # check whether the operator is diagonal in the symmetry index:
        # (1) reshape operator into a matrix of size (op_dim-1)x(op_dim+1),
        #       placing all (previously) diagonal elements (except for the last one)
        #       into the first column of the new matrix
        # (2) check whether the matrix is zero in all columns after the first
        diagonal_test = local_op.flatten()[:-1].reshape((op_dim-1,op_dim+1))
        diagonal = not np.any(diagonal_test[:,1:])

    def _remainder(pops,assignment):
        return tuple( np.array(pops) - np.array(assignment) )

    def _state_idx(assignment):
        states = [ (jj,)
                   for jj in range(len(assignment))
                   for _ in range(assignment[jj]) ]
        return functools.reduce(operator.add, states)

    # determine overall normalization factor of the permutationally symmetric states
    if pops_lft == pops_rht:
        norm = multinomial(pops_lft)
    else:
        log_norm = 1/2 * ( np.log(float(multinomial(pops_lft))) +
                           np.log(float(multinomial(pops_rht))) )
        norm = np.exp(log_norm)

    op_val = 0
    for assignment_lft in assignments(op_spins, spin_dim):

        remainder = _remainder(pops_lft, assignment_lft)
        if any( pop < 0 for pop in remainder ): continue

        assignment_rht = _remainder(pops_rht, remainder)
        if any(np.array(assignment_rht) < 0): continue

        # TODO: compute remainder_perms in a better way
        remainder_perms = multinomial(remainder) / norm
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

def reduced_diagrams(set_sizes):
    return [ reduce_diagram(diagram) for diagram in set_diagrams(set_sizes) ]

def operator_contractions(operators):
    dimensions = [ operator.ndim//2 for operator in operators ]
    return [ contract_ops(operators, diagram) for diagram in set_diagrams(dimensions) ]

# project the product of multi-body operators onto the permutationally symmetric manifold
# couplings: list of tensors (one for each multi-body operator)
# diagrams:  list of (reduced!) diagrams that define tensor contractions
# operators: matrix mapping diagram indices to operators on the many-body hilbert space
def evaluate_operator_product(couplings, diagrams, operators, TI = True):
    assert(len(diagrams) == operators.shape[0])

    # evaluate all distinct diagrams
    diagram_vals = {}
    for diagram in diagrams:
        for diag in diagram.diags:
            _hash = _hash_dict(diag)
            if _hash not in diagram_vals:
                diagram_vals[_hash] = contract_tensors(couplings, diag, TI)
    def _evaluate(diag):
        return diagram_vals[_hash_dict(diag)]

    # collect scalar coefficients for each diagram
    coefficients = np.array([ np.dot(diagram.coefs, list(map(_evaluate, diagram.diags)))
                              for diagram in diagrams ])

    return coefficients @ operators
