#!/usr/bin/env python3

import functools, scipy, itertools

import scipy.special as special
import scipy.sparse as sparse


##########################################################################################
# general methods
##########################################################################################

# determine whether an object can be indexed (i.e. is something like a list)
def is_indexed(object):
    return hasattr(object, "__getitem__")

# sum or multiply all operators in a list
def sum_ops(op_list):
    return functools.reduce(lambda x, y : x + y, op_list)
def mul_ops(op_list):
    return functools.reduce(lambda x, y : x * y, op_list)

# exact binomial coefficient
def binom(nn, kk, vals = {}):
    return special.comb(nn, kk, exact = True)

# find the rank of a k-combination
# k-combination a list of k integers (chosen from, say, {0, 1, ..., N-1})
# for Fock space, "combination" is a list of occupied single-particle states,
# where N single-particle states are indexed 0, 1, ..., N-1
# the "rank" of a Fock state is the index of that Fock state
#   within the subspace of fixed particle number
def rank(k_combination):
    return sum([ binom(index, jj+1)
                 for jj, index in enumerate(sorted(k_combination)) ])

# determine the k-combination { c_1, c_2, ..., c_k } corresponding to a rank,
# with c_1 < c_2 < ... < c_k
# in other words: given the index of and number of particles in a Fock state,
# determine which single-particle states are occupied
def unrank(rank, k):
    if k == 0: return []
    c_k = 0
    while binom(c_k+1,k) <= rank: c_k += 1
    return unrank(rank - binom(c_k,k), k-1) + [ c_k ]


##########################################################################################
# "derived" methods for all operator classes
##########################################################################################

class op_object:
    # assumes implementation of str method
    def __repr__(self): return self.str()

    # assumes implementation of __mul__ method with scalar argument
    def __pos__(self): return +1 * self
    def __neg__(self): return -1 * self
    def __truediv__(self, other): return self * (1/other)
    def __rmul__(self, other): return self * other

    # assumes implementation of __add__ method
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-1) * other
    def __rsub__(self, other): return -self + other

    # assumes implementation of __eq__ method
    def __ne__(self, other): return not self == other

    # assumes implementation of __eq__ and __lt__ methods
    def __gt__(self, other): return not self == other and not self < other


##########################################################################################
# individual fermionic operator
##########################################################################################

class fermion_op_individual(op_object):
    def __init__(self, *index, creation = False):
        self.index = tuple(index)
        self.creation = creation

    def dag(self):
        return fermion_op_individual(*self.index, creation = not self.creation)

    # string representation of an individual fermion operator

    def str(self, str_funs = None):
        if str_funs is None:
            text = ", ".join([ str(idx) for idx in self.index ])
        else:
            text = ", ".join([ str(str_funs[jj](idx)) for jj, idx in enumerate(self.index) ])
        text = "(" + text + ")"
        if self.creation: text += u'\u2020'
        return text

    # multiplication and addition

    def __mul__(self, other):
        return fermion_op(self) * other

    def __add__(self, other):
        return fermion_op(self) + other

    # comparisons of individual fermion operators

    def __eq__(self, other):
        if type(other) is not fermion_op_individual: return False
        return self.creation == other.creation and self.index == other.index

    def __lt__(self, other): # a < b <==> the ordered product of a and b is a * b
        if type(other) is not fermion_op_individual: return False
        if self == other: return False
        if self.creation:
            if not other.creation: return True
            return self.index < other.index
        else: # if not self.creation
            if other.creation: return False
            return not self.index < other.index


##########################################################################################
# sequence (ordered product) of individual fermionic operators
##########################################################################################

class fermion_op_seq(op_object):
    def __init__(self, *sequence):
        self.seq = tuple(sequence)

    def dag(self):
        return fermion_op_seq(*[ op.dag() for op in self.seq[::-1] ])

    # string representation of a fermion operator sequence

    def str(self, str_funs = None):
        if self.seq == []: return "1"
        return " ".join([ op.str(str_funs) for op in self.seq ])

    # multiplication and addition

    def __mul__(self, other):
        if type(other) is fermion_op_seq:
            return fermion_op_seq(*self.seq, *other.seq)
        return fermion_op(self) * other

    def __add__(self, other):
        return fermion_op(self) + other

    # literal comparison of (unsorted!) fermion operator sequences

    def __eq__(self, other):
        if type(other) is not fermion_op_seq: return False
        return self.seq == other.seq

    def __lt__(self, other):
        if type(other) is not fermion_op_seq: return False
        return self.seq < other.seq

    # sort operators and return a fermion_op object of the result
    # WARNING: current implementation is extremely inefficient
    def sorted(self, fixed_op_number = None):
        if fixed_op_number is not None and len(self.seq) > fixed_op_number:
            return 0

        for jj in range(len(self.seq)-1):
            op_fst, op_snd = self.seq[jj], self.seq[jj+1]
            if op_fst < op_snd: continue # this pair is already sorted
            if op_fst == op_snd: return 0 # identical fermionic operators

            # if op_fst > op_snd
            head, tail = self.seq[:jj], self.seq[jj+2:] # operators before/after this pair
            comm_list = head + ( op_snd, op_fst ) + tail # commute this pair
            if op_fst.index != op_snd.index: # operators address different states
                return - fermion_op_seq(*comm_list).sorted(fixed_op_number)
            else: # these operators are adjoints of each other
                unit_list = head + tail
                return ( + fermion_op_seq(*unit_list).sorted(fixed_op_number)
                         - fermion_op_seq(*comm_list).sorted(fixed_op_number) )

        if fixed_op_number is None:
            return fermion_op(self)
        else:
            restricted_op = fermion_op(self).restricted(fixed_op_number)
            if fixed_op_number == 0:
                return restricted_op.coeffs[0]
            else:
                return restricted_op

    # return matrix reresentation of this operator
    #   when acting on a fixed-particle-number Fock state
    # index_map associates a unique integer to the indices of an individual operator
    # single_states is the number of single-particle states
    # input_number is the number of particles in the Fock state this operator will act on
    def matrix(self, single_states, input_number, index_map = None):
        # enforce that
        # (i) we have some operators in this product
        # (ii) the operators in this product are sorted
        assert(len(self.seq) != 0)
        assert(len(self.seq) == 1 or
               all( self.seq[jj] < self.seq[jj+1] for jj in range(len(self.seq)-1) ))

        # make sure that if the index_map is trivial (None),
        #   then the individual operators in this product only have one index
        assert(index_map is not None or len(self.seq[0].index) == 1)
        # the default index map is just the value of *the* index on an individual operator
        if index_map is None: index_map = lambda index : index[0]

        # identify (by index) states addressed by creation/annihilation operators
        dest_states = [ index_map(op.index) for op in self.seq if not op.creation ]
        crtn_states = [ index_map(op.index) for op in self.seq if op.creation ]

        # identify states addressed by both creation and destruction operators,
        # as well as states not addressed by any operator
        comm_states = list(set( dest_states + crtn_states ))
        remaining_states = list(range(single_states))
        for idx in comm_states:
            remaining_states.remove(idx)

        # number of particles in output state
        output_number = input_number + len(crtn_states) - len(dest_states)

        # build an empty matrix
        input_dimension = binom(single_states, input_number)
        output_dimension = binom(single_states, output_number)
        matrix = sparse.dok_matrix((output_dimension,input_dimension), dtype = int)

        # number of auxiliary particles not addressed by creation/annihilation operators
        aux_number = output_number - len(comm_states)
        if aux_number < 0: return matrix

        # loop over all combinations of auxiliary single-particle states
        for aux_states in itertools.combinations(remaining_states, aux_number):
            # determine single-particle states occupied in input/output Fock state
            input_states = sorted(aux_states + tuple(dest_states))
            output_states = sorted(aux_states + tuple(crtn_states))

            # count number of sign flips from permuting operators
            sign_flips = 0
            for jj, op_idx in enumerate(dest_states[::-1]):
                sign_flips += sum( state < op_idx for state in input_states ) - jj
            for op_idx in crtn_states:
                sign_flips += sum( state < op_idx for state in aux_states )

            matrix[rank(output_states), rank(input_states)] = (-1)**sign_flips

        return matrix

    # assuming this product of operators is sorted
    # return the vector corresponding to the Fock state that this product creates
    # within the appropriate subspace of fixed particle number
    def vector(self, single_states, index_map = None):
        if not all( op.creation for op in self.seq ): return 0
        return self.matrix(single_states, 0, index_map)


##########################################################################################
# general fermionic operator: sum of sequences of fermionic operators
##########################################################################################

class fermion_op(op_object):
    def __init__(self, seqs = None, coeffs = None, from_index = None):
        # set sequences of individual fermionic operators
        if seqs is None or seqs == [] or seqs == ():
            self.seqs = ()
        elif type(seqs) is fermion_op_individual:
            self.seqs = ( fermion_op_seq(seqs), )
        elif type(seqs) is fermion_op_seq:
            self.seqs = ( seqs, )
        else:
            # assert that all items in seqs are either
            # (i) individual fermion operators, or
            # (ii) sequences of fermion operators
            assert(all( type(item) is fermion_op_individual for item in seqs ) or
                   all( type(item) is fermion_op_seq for item in seqs ))
            if type(seqs[0]) is fermion_op_individual:
                self.seqs = ( fermion_op_seq(item) for item in seqs )
            else:
                self.seqs = tuple(seqs)

        # set coefficients for each sequence
        if coeffs is None or coeffs == [] or coeffs == ():
            self.coeffs = (1,) * len(self.seqs)
        elif not is_indexed(coeffs):
            self.coeffs = ( coeffs )
        else:
            assert(len(coeffs) == len(self.seqs))
            self.coeffs = tuple(coeffs)

    def dag(self):
        return fermion_op([ seq.dag() for seq in self.seqs ],
                          [ coeff.conjugate() for coeff in self.coeffs ])

    def objs(self, sort = False):
        if not sort: return list(zip(self.seqs, self.coeffs))
        else: return sorted(self.objs(), key = lambda x : x[0])

    # string representation of fermion operators

    def str(self, str_funs = None):
        if len(self.seqs) == 0: return "[0]"
        def coeff_seq_str(coeff, seq):
            if len(seq.seq) == 0: return f"[{coeff}]"
            else: return f"[{coeff}] " + seq.str(str_funs)
        return " + ".join([ coeff_seq_str(coeff, seq)
                            for seq, coeff in zip(self.seqs, self.coeffs) ])

    # multiplication and addition of fermion operators

    def __mul__(self, other):
        if type(other) is fermion_op_individual:
            new_seqs = [ fermion_op_seq(*seq.seq, other) for seq in self.seqs ]
            return fermion_op(new_seqs, self.coeffs)
        if type(other) is fermion_op_seq:
            new_seqs = [ fermion_op_seq(*seq.seq, *other.seq) for seq in self.seqs ]
            return fermion_op(new_seqs, self.coeffs)
        if type(other) is fermion_op:
            obj_list = [ ( self_seq * other_seq, self_coeff * other_coeff )
                         for self_seq, self_coeff in self.objs()
                         for other_seq, other_coeff in other.objs() ]
            return fermion_op(*zip(*obj_list))
        else: # "other" is a scalar
            return fermion_op(self.seqs, [ other * coeff for coeff in self.coeffs ])

    def __add__(self, other):
        if type(other) in ( fermion_op_individual, fermion_op_seq ):
            return self + fermion_op(other)
        if type(other) is fermion_op:
            return fermion_op(self.seqs + other.seqs, self.coeffs + other.coeffs)
        else: # "other" is a scalar
            return self + other * fermion_op()

    # literal comparison of (unsorted!) fermion operators
    def __eq__(self, other):
        if type(other) is not fermion_op: return False
        return self.objs(True) == other.objs(True)

    # return this fermion operator restricted to a num_ops-operator subpace
    def restricted(self, num_ops, maximum = False):
        if is_indexed(num_ops):
            include = lambda seq : len(seq.seq) in num_ops
        elif not maximum:
            include = lambda seq : len(seq.seq) == num_ops
        else:
            include = lambda seq : len(seq.seq) <= num_ops
        return fermion_op(*zip(*[ (seq, coeff) for seq, coeff in self.objs()
                                  if include(seq) ]))

    # sort and simplify all terms
    def sorted(self, fixed_op_number = None):
        if len(self.seqs) == 0: return 0

        # sort all products
        sorted_sum = sum_ops( coeff * seq.sorted() for seq, coeff in self.objs() )
        sorted_objs = [ list(obj) for obj in sorted_sum.objs(True) ]

        # if we are fixing an operator number, throw out all unwanted terms
        if fixed_op_number is not None:
            sorted_objs = [ objs for objs in sorted_objs
                            if len(objs[0].seq) == fixed_op_number ]

        # combine terms that are proportional
        for jj in range(len(sorted_objs)-1,0,-1):
            jj_seq, jj_coeff = sorted_objs[jj]
            for kk in range(jj):
                kk_seq, kk_coeff = sorted_objs[kk]
                if jj_seq == kk_seq:
                    sorted_objs[kk][1] += jj_coeff
                    del sorted_objs[jj]
                    break

        # remove terms with a zero coefficient
        for jj, obj in reversed(list(enumerate(sorted_objs))):
            if obj[1] == 0: del sorted_objs[jj]

        # if the result is a scalar, return a scalar
        if len(sorted_objs) == 0: return 0
        if len(sorted_objs) == 1 and sorted_objs[0][0].seq == ():
            return sorted_objs[0][1]

        return fermion_op(*zip(*sorted_objs))

    # return matrix/vector reresentations of this operator
    #   when acting on a fixed-particle-number Fock state
    # index_map associates a unique integer to the indices of an individual operator
    # single_states is the number of single-particle states
    # input_number is the number of particles in the Fock state this operator will act on
    def matrix(self, single_states, input_number, index_map = None):
        sorted_op = self.sorted()
        return sum( coeff * op.matrix(single_states, input_number, index_map)
                    for op, coeff in sorted_op.objs() )
    def vector(self, single_states, index_map = None):
        return self.matrix(single_states, 0, index_map)

# define the method that should *actually* be used to construct fermion operators
# f_op accepts indices for an individual fermion operator, and returns a fermion_op object
def f_op(*indices):
    return fermion_op(fermion_op_individual(*indices))

# return matrix/vector object restricted to the span of given Fock states
# index_map is a dictionary: { < Fock state index > : < restricted subspace index > }
def reduce_dimension(matrix, index_map):
    matrix_is_square = matrix.shape[0] == matrix.shape[1]
    matrix_is_vector = 1 in matrix.shape
    assert(matrix_is_square or matrix_is_vector)

    if matrix_is_square:
        reduced_dim = (len(index_map),len(index_map))
        reduced_matrix = sparse.dok_matrix(reduced_dim, dtype = matrix.dtype)

        cm = matrix.tocoo()
        for state_out, state_in, value in zip(cm.row, cm.col, cm.data):
            try:
                idx_out = index_map[state_out]
                idx_in = index_map[state_in]
                reduced_matrix[idx_out,idx_in] = matrix[state_out,state_in]
            except: None
        return reduced_matrix

    if matrix_is_vector:
        cm = matrix.tocoo()

        if matrix.shape[0] == 1:
            cm_indices = cm.col
            reduced_dim = (1,len(index_map))
            def mat_idx(idx): return (0,idx)
        if matrix.shape[1] == 1:
            cm_indices = cm.row
            reduced_dim = (len(index_map),1)
            def mat_idx(idx): return (idx,0)

        reduced_matrix = sparse.dok_matrix(reduced_dim, dtype = matrix.dtype)

        for state, value in zip(cm_indices, cm.data):
            try:
                old_idx = mat_idx(state)
                new_idx = mat_idx(index_map[state])
                reduced_matrix[new_idx] = matrix[old_idx]
            except: None
        return reduced_matrix
