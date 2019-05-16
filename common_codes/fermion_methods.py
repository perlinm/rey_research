#!/usr/bin/env python3

import functools, scipy, itertools

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
def binom(n, k):
    return scipy.special.comb(n, k, exact = True)

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
# in other words: given the index of a Fock state,
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
    def __sub__(self, other): return self + (-1) * other

    # assumes implementation of __eq__ method
    def __ne__(self, other): return not self == other

    # assumes implementation of __eq__ and __lt__ methods
    def __gt__(self, other): return not self == other and not self < other


##########################################################################################
# individual fermionic operator
##########################################################################################

class c_op(op_object):
    def __init__(self, index, creation = False):
        if is_indexed(index):
            self.index = index
        else:
            self.index = [index]
        self.creation = creation

    def dag(self):
        return c_op(self.index, not self.creation)

    # printing this object

    def str(self, str_funs = None):
        if str_funs is None:
            text = ", ".join([ str(idx) for idx in self.index ])
        else:
            text = ", ".join([ str(str_funs[jj](idx)) for jj, idx in enumerate(self.index) ])
        text = "(" + text + ")"
        if self.creation: text += u'\u2020'
        return text

    # object operations

    def __mul__(self, other):
        if type(other) is c_op:
            return c_seq([self, other])
        if type(other) is c_seq:
            return c_seq([self] + other.seq)
        if type(other) is c_sum:
            return c_sum([ self * item for item in other.seq_list ], other.coeff_list)
        else:
            return c_sum([self], [other])

    def __add__(self, other):
        if type(other) is c_op:
            return c_seq(self) + c_seq(other)
        if type(other) in (c_seq, c_sum):
            return c_seq(self) + other
        else:
            raise TypeError(f"adding invalid type to c_op: {type(other)}")

    # object comparisons

    def __eq__(self, other):
        if type(other) is not c_op: return False
        return self.creation == other.creation and self.index == other.index

    def __lt__(self, other): # a < b <==> the ordered product of a and b is a * b
        assert(type(other) is c_op)
        if self == other: return False
        if self.creation:
            if not other.creation: return True
            return self.index < other.index
        else: # if not self.creation
            if other.creation: return False
            return not self.index < other.index


##########################################################################################
# product of fermionic operators
##########################################################################################

class c_seq(op_object):
    def __init__(self, sequence = None):
        if sequence is None:
            self.seq = []
        if type(sequence) is c_op:
            self.seq = [sequence]
        else:
            self.seq = list(sequence)

    def dag(self):
        return c_seq([ seq.dag() for seq in self.seq[::-1] ])

    # printing this object

    def str(self, str_funs = None):
        if self.seq == []: return "1"
        return " ".join([ item.str(str_funs) for item in self.seq ])

    # object operations

    def __mul__(self, other):
        if type(other) is c_op:
            return c_seq(self.seq + [other])
        if type(other) is c_seq:
            return c_seq(self.seq + other.seq)
        if type(other) is c_sum:
            return c_sum([ self * item for item in other.seq_list ], other.coeff_list)
        else:
            return c_sum([self], [other])

    def __add__(self, other):
        if type(other) is c_op:
            return c_sum(self) + c_seq(other)
        if type(other) in (c_seq, c_sum):
            return c_sum(self) + other
        else:
            raise TypeError(f"adding invalid type to c_seq: {type(other)}")

    # object comparisons
    # WARNING: comparisons are "literal", without sorting

    def __eq__(self, other):
        if type(other) is not c_op: return False
        return self.seq == other.seq

    def __lt__(self, other):
        assert(type(other) is c_seq)
        return self.seq < other.seq

    # miscellaneous methods

    # sort operators and return a c_sum object of the result
    # WARNING: current implementation is extremely inefficient
    def sorted(self):
        if self.seq == []: return c_sum(self)
        seq_list = self.seq.copy()
        for jj in range(len(self.seq)-1):
            op_fst, op_snd = seq_list[jj], seq_list[jj+1]
            if op_fst < op_snd: continue
            if op_fst == op_snd: return c_sum()
            else: # op_fst > op_snd
                head, tail = seq_list[:jj], seq_list[jj+2:]
                comm_list = head + [ op_snd, op_fst ] + tail
                if op_fst.index != op_snd.index:
                    return -c_seq(comm_list).sorted()
                else: # these operators are conjugates of each other
                    unit_list = head + tail
                    return c_seq(unit_list).sorted() - c_seq(comm_list).sorted()

        return c_sum(c_seq(seq_list))

    # return matrix reresentation of this operator
    #   when acting on a fixed-particle-number Fock state
    # index_map associates a unique integer to the indices of an individual operator
    # single_states is the number of sinple-particle states
    # input_number is the number of particles in the Fock state this operator will act on
    def matrix(self, single_states, input_number, index_map = None):
        # enforce that
        # (i) we have some operators in this product
        # (ii) the operators in this product are sorted
        assert(len(self.seq) != 0)
        if len(self.seq) > 1:
            assert(all( self.seq[jj] < self.seq[jj+1] for jj in range(len(self.seq)-1) ))

        # make sure that if the index_map is trivial (None),
        #   then the individual operators in this product only have one index
        assert(index_map is not None or len(self.seq[0].index) == 1)
        # the default index map is just the value of *the* index on an individual operator
        if index_map is None: index_map = lambda index : index[0]

        # identify (by index) states addressed by creation/annihilation operators
        dest_states = [ index_map(op.index) for op in self.seq if not op.creation ]
        crtn_states = [ index_map(op.index) for op in self.seq if op.creation ]

        # list of single-particle states *not* addressed by creation/annihilation operators
        remaining_states = list(range(single_states))
        for idx in set( dest_states + crtn_states ):
            remaining_states.remove(idx)

        # number of particles in output state
        output_number = input_number + len(crtn_states) - len(dest_states)

        # build an empty matrix
        input_dimension = binom(single_states, input_number)
        output_dimension = binom(single_states, output_number)
        matrix = sparse.dok_matrix((output_dimension,input_dimension), dtype = int)

        # number of auxiliary particles not addressed by creation/annihilation operators
        aux_number = output_number - len(self.seq)
        if aux_number < 0: return matrix

        # loop over all combinations of auxiliary single-particle states
        for aux_states in itertools.combinations(remaining_states, aux_number):
            # determine single-particle states occupied in input/output Fock state
            input_states = aux_states + tuple(dest_states)
            output_states = aux_states + tuple(crtn_states)

            # count number of sign flips from permuting operators
            sign_flips = sum( state < op_idx
                              for state in aux_states
                              for op_idx in crtn_states + dest_states )
            sign_flips -= binom(len(dest_states),2)

            matrix[rank(output_states), rank(input_states)] = (-1)**sign_flips

        return matrix

    # assuming this product of operators is sorted
    # return the vector corresponding to the Fock state that this product creates
    # within the appropriate subspace of fixed particle number
    def vector(self, single_states, index_map = None):
        if not all( op.creation for op in self.seq ): return 0
        return self.matrix(single_states, 0, index_map)


##########################################################################################
# sum of products of fermionic operators
##########################################################################################

class c_sum(op_object):
    def __init__(self, seq_list = None, coeff_list = None):
        if seq_list is None or seq_list == []:
            self.seq_list = []
        elif type(seq_list) is c_op:
            self.seq_list = [ c_seq(seq_list) ]
        elif type(seq_list) is c_seq:
            self.seq_list = [ seq_list ]
        else:
            # assert that either all items are operators (c_op)
            #   or all items are sequences of operators (c_seq)
            assert(all( type(item) is c_op for item in seq_list ) or
                   all( type(item) is c_seq for item in seq_list ))
            if type(seq_list[0]) is c_op:
                self.seq_list = [ c_seq(item) for item in seq_list ]
            else: # if items are of type c_seq
                self.seq_list = seq_list
        if coeff_list is None:
            self.coeff_list = [1] * len(self.seq_list)
        elif not is_indexed(coeff_list):
            self.coeff_list = [ coeff_list ]
        else:
            assert(len(coeff_list) == len(self.seq_list))
            self.coeff_list = coeff_list

    def dag(self):
        return c_sum([ seq.dag() for seq in self.seq_list ],
                     [ coeff.conjugate() for coeff in self.coeff_list ])

    def objs(self, sort = False):
        if not sort: return list(zip(self.seq_list, self.coeff_list))
        else: return sorted(self.objs(), key = lambda x : x[0])

    # printing this object

    def str(self, str_funs = None):
        if len(self.seq_list) == 0: return "[0]"
        def coeff_seq_str(coeff, seq):
            if len(seq.seq) == 0: return f"[{coeff}]"
            else: return f"[{coeff}] " + seq.str(str_funs)
        return " + ".join([ coeff_seq_str(coeff, seq)
                            for seq, coeff in zip(self.seq_list, self.coeff_list) ])

    # object operations

    def __mul__(self, other):
        if type(other) is c_op or type(other) is c_seq:
            return c_sum([ item * other for item in self.seq_list ], self.coeff_list)
        if type(other) is c_sum:
            obj_list = [ ( self_seq * other_seq, self_coeff * other_coeff )
                         for self_seq, self_coeff in self.objs()
                         for other_seq, other_coeff in other.objs() ]
            return c_sum(*zip(*obj_list))
        else:
            return c_sum(self.seq_list, [ other * coeff for coeff in self.coeff_list ])

    def __add__(self, other):
        if type(other) is c_op:
            return self + c_seq(other)
        if type(other) is c_seq:
            return c_sum(self.seq_list + [other], self.coeff_list + [1])
        if type(other) is c_sum:
            return c_sum(self.seq_list + other.seq_list,
                         self.coeff_list + other.coeff_list)
        else:
            raise TypeError(f"adding invalid type to c_sum: {type(other)}")

    # object comparisons

    def __eq__(self, other):
        if type(other) is not c_op: return False
        return self.objs(True) == other.objs(True)

    # miscellaneous methods

    # sort and simplify all terms
    def sorted(self):
        if len(self.seq_list) == 0: return c_sum()

        # sort all products
        sorted_sum = sum_ops([ coeff * seq.sorted() for seq, coeff in self.objs() ])
        sorted_objs = [ list(obj) for obj in sorted_sum.objs(True) ]

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

        return c_sum(*zip(*sorted_objs))

    # return this c_sum restricted to a num_ops-operator subpace
    def restricted(self, num_ops, maximum = False):
        if is_indexed(num_ops):
            include = lambda c_seq : len(c_seq.seq) in num_ops
        elif not maximum:
            include = lambda c_seq : len(c_seq.seq) == num_ops
        else:
            include = lambda c_seq : len(c_seq.seq) <= num_ops
        return c_sum(*zip(*[ (c_seq, coeff) for c_seq, coeff in self.objs()
                             if include(c_seq) ]))

    # return matrix/vector reresentations of this operator
    #   when acting on a fixed-particle-number Fock state
    # index_map associates a unique integer to the indices of an individual operator
    # single_states is the number of sinple-particle states
    # input_number is the number of particles in the Fock state this operator will act on
    def matrix(self, single_states, input_number, index_map = None):
        sorted_op = self.sorted()
        return sum( coeff * op.matrix(single_states, input_number, index_map)
                    for op, coeff in sorted_op.objs() )
    def vector(self, single_states, index_map = None):
        return self.matrix(single_states, 0, index_map)
