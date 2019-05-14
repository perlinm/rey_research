#!/usr/bin/env python3

from functools import reduce

def sum_ops(op_list):
    return reduce(lambda x, y : x + y, op_list)

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
        if hasattr(index, "__getitem__"):
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
            return c_seq([self]) + c_seq([other])
        if type(other) is (c_seq, c_sum):
            return c_seq([self]) + other
        else:
            raise TypeError(f"adding invalid type to c_op: {type(other)}")

    # object comparisons

    def __eq__(self, other):
        assert(type(other) is c_op)
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
            return c_sum([ self * item for item in other.seq_list ], other.cof_list)
        else:
            return c_sum([self], [other])

    def __add__(self, other):
        if type(other) is c_op:
            return c_sum([self]) + c_seq([other])
        if type(other) in (c_seq, c_sum):
            return c_sum([self]) + other
        else:
            raise TypeError(f"adding invalid type to c_seq: {type(other)}")

    # object comparisons
    # WARNING: comparisons are "literal", without sorting

    def __eq__(self, other):
        assert(type(other) is c_seq)
        return self.seq == other.seq

    def __lt__(self, other):
        assert(type(other) is c_seq)
        return self.seq < other.seq

    # miscellaneous methods

    # sort operators and return a c_sum object of the result
    # WARNING: current implementation is extremely inefficient
    def sorted(self):
        if self.seq == []: return c_sum([self])
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
        return c_sum([c_seq(seq_list)])



##########################################################################################
# sum of products of fermionic operators
##########################################################################################

class c_sum(op_object):
    def __init__(self, seq_list = None, coeff_list = None):
        if seq_list is None or seq_list == []:
            self.seq_list = []
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
            return self + c_seq([other])
        if type(other) is c_seq:
            return c_sum(self.seq_list + [other], self.coeff_list + [1])
        if type(other) is c_sum:
            return c_sum(self.seq_list + other.seq_list,
                         self.coeff_list + other.coeff_list)
        else:
            raise TypeError(f"adding invalid type to c_sum: {type(other)}")

    # object comparisons

    def __eq__(self, other):
        assert(type(other) is c_sum)
        return self.objs(True) == other.objs(True)

    # miscellaneous methods

    # sort and simplify all terms
    def sorted(self):
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
    def restricted(self, num_ops):
        return c_sum(*zip(*[ (seq, coeff) for seq, coeff in self.objs()
                             if len(seq.seq) == num_ops ]))
