#!/usr/bin/env python3

# FILE CONTENTS: (symbolic) methods relating to fermion squeezing

import sys
import numpy as np
import scipy.linalg as linalg
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import itertools, functools, operator

from scipy.special import binom as binomial
from sympy.combinatorics.permutations import Permutation as sympy_permutation

# return product of items in list
def sum(list):
    try:
        return functools.reduce(operator.add, list)
    except TypeError: # for error: reduce() of empty sequence with no initial value
        return 0
def product(list): return functools.reduce(operator.mul, list)

# multiply a matrix M by a diagonal matrix with entries in the vector D
def diag_mult(D, M, left):
    if left:
        # multiply by D on the left: np.diag(D) @ M
        return ( D * M.T ).T
    else:
        # multiply by D on the right: M @ np.diag(D)
        return D * M


##########################################################################################
# objects and methods for constructing operators
##########################################################################################

# iterable for basis of fock states; returns index of fock state
#   and a tuple specifying the occupied single-particle states (by index)
def fock_state_basis(L_x, L_y, N):
    fock_states = int(binomial(2*L_x*L_y,N))
    return zip(range(fock_states), itertools.combinations(range(2*L_x*L_y),N))

# index of a single-particle state
def single_state_index(q_x, q_y, spin_up, L_x):
    return (q_x + q_y * L_x) * 2 + spin_up

# return annihilation operator corresponding to an index of a single-particle state
def c_op_by_index(index, L_x):
    spin_up = index % 2
    q_x = (index // 2 % L_x)
    q_y = (index // 2 - q_x) // L_x
    return c_op(q_x, q_y, spin_up)

# creation / annihilation operators
class c_op:
    def __init__(self, q_x, q_y, spin_up, creation = False):
        self.q_x = int(q_x)
        self.q_y = int(q_y)
        self.spin_up = bool(spin_up)
        self.creation = bool(creation)

    def __repr__(self):
        spin_text = "up" if self.spin_up else "dn"
        if not self.creation:
            return str((self.q_x, self.q_y, spin_text))
        else:
            return str((self.q_x, self.q_y, spin_text, "dag"))

    def __eq__(self, other):
        assert(type(other) is c_op)
        return self.info() == other.info()

    def __ne__(self, other): return not self == other

    def __lt__(self, other):
        assert(type(other) is c_op)
        return self.info() < other.info()

    def __gt__(self, other):
        assert(type(other) is c_op)
        return self.info() > other.info()

    def __mul__(self, other):
        if np.isscalar(other):
            return c_seq(self, other)
        if type(other) is c_op:
            return c_seq([self, other])
        if type(other) is c_seq:
            return c_seq([self] + other.seq, other.prefactor)
        if type(other) is c_sum:
            return c_sum([ self * item for item in other.seq_list ])
        else:
            sys.exit("error: multiplying c_op by invalid type:", type(other))

    def __rmul__(self, other):
        if np.isscalar(other):
            return self * other
        else:
            sys.exit("error: multiplying c_op by invalid type:", type(other))

    def __truediv__(self, other): return self * (1/other)

    def __add__(self, other):
        if type(other) is c_op:
            return c_sum([self, other])
        if type(other) is c_seq:
            return c_sum([c_seq(self), other])
        else:
            sys.exit("error: adding invalid type to c_seq:", type(other))
    def __sub__(self, other): return self + (-1) * other

    def info(self):
        sign = 1 if self.creation else -1
        return (not self.creation, sign * self.q_y, sign * self.q_x, sign * self.spin_up)

    # return hermitian conjugate
    def dag(self):
        return c_op(self.q_x, self.q_y, self.spin_up, not self.creation)

    # index of the state addressed by this operator
    def index(self, L_x):
        return single_state_index(self.q_x, self.q_y, self.spin_up, L_x)

    def vector(self, L_x, L_y, N):
        return c_seq(self).vector(L_x, L_y, N)

# product of creation / annihilation operators
class c_seq:
    def __init__(self, sequence = [], prefactor = 1):
        if type(sequence) is c_op:
            self.seq = [sequence]
        else:
            self.seq = sequence
        self.prefactor = prefactor
        self.sort()

    def __repr__(self):
        op_text = " * ".join([ str(item) for item in self.seq ])
        return "+ {} * {}".format(self.prefactor, op_text)

    def __eq__(self, other):
        assert(type(other) is c_seq)
        if len(self.seq) != len(other.seq):
            return False
        if self.prefactor != other.prefactor:
            return False
        return all([ self.seq[ii] == other.seq[ii] for ii in range(len(self.seq)) ])

    def __ne__(self, other): return not self == other

    def __mul__(self, other):
        if np.isscalar(other):
            return c_seq(self.seq, self.prefactor * other)
        if type(other) is c_op:
            return c_seq(self.seq + [other], self.prefactor)
        if type(other) is c_seq:
            return c_seq(self.seq + other.seq, self.prefactor * other.prefactor)
        if type(other) is c_sum:
            return c_sum([ self * item for item in other.seq_list ])
        else:
            sys.exit("error: multiplying c_seq by invalid type:", type(other))

    def __rmul__(self, other):
        if np.isscalar(other):
            return self * other
        else:
            sys.exit("error: multiplying c_seq by invalid type:", type(other))

    def __truediv__(self, other): return self * (1/other)

    def __add__(self, other):
        if type(other) is c_op:
            return c_sum([self, c_seq(other)])
        if type(other) is c_seq:
            return c_sum([self, other])
        if type(other) is c_sum:
            return c_sum([self] + other.seq_list)
        else:
            sys.exit("error: adding invalid type to c_seq:", type(other))

    def __sub__(self, other): return self + (-1) * other

    # sort operators change sign of prefactor if appropriate
    def sort(self):
        if len(self.seq) == 0: return None
        sorted_ops = sorted([ [ self.seq[ii], ii ] for ii in range(len(self.seq)) ])
        sorted_list, sorting_permutation = zip(*sorted_ops)
        self.seq = list(sorted_list)
        self.prefactor *= sympy_permutation(sorting_permutation).signature()

    # return Hermitian conjudate of this sequence
    def dag(self):
        return c_seq([ item.dag() for item in self.seq[::-1] ], self.prefactor)

    # return fock state created or destroyed by the given sequence of fermionic operators
    def vector(self, L_x, L_y, N):
        # assert we have the same nubmer of operators as particles
        assert(N == len(self.seq))
        # assert that all operators are of the same type (i.e. creation / annihilation)
        assert( len(set( op.creation for op in self.seq )) == 1 )
        # initialize zero state
        state = np.zeros(int(binomial(2*L_x*L_y,N)), dtype = complex)
        # determine which sinle-particle states will be occupied
        occupied_states = tuple( op.index(L_x) for op in self.seq[::-1] )
        # if any operators are repeats, return the zero state
        if len(occupied_states) != len(set(occupied_states)): return state

        for index, states in fock_state_basis(L_x, L_y, N):
            if occupied_states == states:
                state[index] = self.prefactor
                return state
        sys.exit("state not found in fock basis...")

    # return vector or matrix corresponding to this product of operators
    def matrix(self, L_x, L_y, N):
        assert(N <= 2*L_x*L_y) # we cannot have more particles than states

        num_ops = len(self.seq)
        assert(num_ops % 2 == 0) # we must have an even number of operators

        # make sure all operators are within our bounds
        assert(all( op.q_x < L_x for op in self.seq ))
        assert(all( op.q_y < L_y for op in self.seq ))

        # if it is not a vector, this is a matrix element (e.g. of a Hamiltonian)
        # to enforce conservation of particle number, we only allow sequences in which
        #   all annihilation operators precede all creation operators
        #   when read read right to left
        assert(all( op.creation for op in self.seq[:num_ops//2] ) and
               all( not op.creation for op in self.seq[num_ops//2:] ))

        # dimension of hilbert space and initial zero matrix
        hilbert_dim = int(binomial(2*L_x*L_y, N))
        matrix = np.zeros((hilbert_dim,hilbert_dim), dtype = complex)

        # if we have an empty sequence, return the zero matrix
        if num_ops == 0: return matrix

        # creation / destruction operators and their indices
        created_states = [ op.index(L_x) for op in self.seq[:num_ops//2] ]
        destroyed_states = [ op.index(L_x) for op in self.seq[num_ops//2:] ]

        # if we address any states twice, return the zero matrix
        if len(set(created_states)) + len(set(destroyed_states)) != len(self.seq):
            return matrix

        for index_in, states_in in fock_state_basis(L_x, L_y, N):
            # if this combination of single particle states is not addressed
            #   by the destruction operators, continue to next combination
            if any([ state not in states_in for state in destroyed_states ]): continue

            # count the number of times we have to commute operators in order to
            #   eliminate the destruction operators in this c_seq
            comms_in = sum( sum( 1 for state in states_in
                                 if state > destroyed_state )
                            for destroyed_state in destroyed_states )

            remaining_states_in = [ state for state in states_in
                                    if state not in destroyed_states ]

            for index_out, states_out in fock_state_basis(L_x, L_y, N):
                if any([ state not in states_out for state in created_states ]): continue

                remaining_states_out = [ state for state in states_out
                                         if state not in created_states ]

                # any states not addressed by the operators in c_seq should be identical
                if remaining_states_in != remaining_states_out: continue

                comms_out = sum( sum( 1 for state in states_out
                                      if state > created_state )
                                 for created_state in created_states )

                # set the appropriate matrix element to 1 or -1
                #   depening on appropriate sign from commuting operators
                matrix[index_out, index_in] = (-1)**(comms_in + comms_out)

        return self.prefactor * matrix

# sum of products of creation / annihilation operators
class c_sum:
    def __init__(self, item_list):
        # assert that either all items are operators (c_op)
        #   or all items are sequences of operators (c_seq)
        assert(all( type(item) is c_op for item in item_list ) or
               all( type(item) is c_seq for item in item_list ))
        if type(item_list[0]) is c_op:
            self.seq_list = [ c_seq(item) for item in item_list ]
        else: # if items are of type c_seq
            # assert all nonempty sequences are of equal length
            sequence_lengths = set( len(item.seq) for item in item_list )
            assert(len(sequence_lengths - set([0])) in [ 1, 0 ])
            self.seq_list = item_list

    def __repr__(self):
        return " ".join([ str(item) for item in self.seq_list ])

    def __eq__(self, other):
        assert(type(other) is c_sum)
        return all([ item in other.seq_list for item in self.seq_list ])

    def __ne__(self, other): return not self == other

    def __mul__(self, other):
        if np.isscalar(other):
            return c_sum([ other * item for item in self.seq_list ])
        if type(other) is c_op or type(other) is c_seq:
            return c_sum([ item * other for item in self.seq_list ])
        if type(other) is c_sum:
            return c_sum([ self_item * other_item
                           for self_item in self.seq_list
                           for other_item in other.seq_list ])
        else:
            sys.exit("error: multiplying c_sum by invalid type:", type(other))

    def __rmul__(self, other):
        if np.isscalar(other):
            return self * other
        else:
            sys.exit("error: multiplying c_sum by invalid type:", type(other))

    def __truediv__(self, other): return self * (1/other)

    def __add__(self, other):
        if type(other) is c_op:
            return c_sum(self.seq_list + [c_seq(other)])
        if type(other) is c_seq:
            return c_sum(self.seq_list + [other])
        if type(other) is c_sum:
            return c_sum(self.seq_list + other.seq_list)
        else:
            sys.exit("error: adding invalid type to c_seq:", type(other))

    def __sub__(self, other): return self + (-1) * other

    def dag(self):
        return c_sum([ item.dag() for item in self.seq_list ])

    def vector(self, L_x, L_y, N):
        return sum( item.vector(L_x, L_y, N) for item in self.seq_list )

    def matrix(self, L_x, L_y, N):
        return sum( item.matrix(L_x, L_y, N) for item in self.seq_list )


##########################################################################################
# useful operators
##########################################################################################

# collective spin operators for N particles on a 2-D lattice with (L_x,L_y) sites
def Sz_op(L_x, L_y, N):
    return sum( ( c_op(q_x,q_y,1).dag() * c_op(q_x,q_y,1)
                  - c_op(q_x,q_y,0).dag() * c_op(q_x,q_y,0) ) / 2
                for q_y in range(L_y)
                for q_x in range(L_x) ).matrix(L_x, L_y, N)
def Sx_op(L_x, L_y, N):
    return sum( ( c_op(q_x,q_y,0).dag() * c_op(q_x,q_y,1)
                  + c_op(q_x,q_y,1).dag() * c_op(q_x,q_y,0) ) / 2
                for q_y in range(L_y)
                for q_x in range(L_x) ).matrix(L_x, L_y, N)
def Sy_op(L_x, L_y, N):
    return sum( ( c_op(q_x,q_y,0).dag() * c_op(q_x,q_y,1)
                  - c_op(q_x,q_y,1).dag() * c_op(q_x,q_y,0) ) / 2
                for q_y in range(L_y)
                for q_x in range(L_x) ).matrix(L_x, L_y, N) * 1j
def S_ops(L_x, L_y, N):
    return Sz_op(L_x, L_y, N), Sx_op(L_x, L_y, N), Sy_op(L_x, L_y, N)

# determine collective spin operators from ambiguous inputs
def get_spin_ops(Sz_or_Lx, Sx_or_Ly, Sy_or_N, operators):
    if operators: # the inputs are spin operators
        return Sz_or_Lx, Sx_or_Ly, Sy_or_N
    else: # use the inputs to compute spin operators
        return S_ops(Sz_or_Lx, Sx_or_Ly, Sy_or_N)

def Sn_op(vec, Sz_or_Lx, Sx_or_Ly, Sy_or_N, operators = True):
    Sz, Sx, Sy = get_spin_ops(Sz_or_Lx, Sx_or_Ly, Sy_or_N, operators)
    return ( vec[0] * Sz + vec[1] * Sx + vec[2] * Sy ) / np.linalg.norm(vec)

# density operators with completely mixed spatial degees of freedom,
#   but all spins pointing along a given axis
def polarized_states(L_x, L_y, N):
    hilbert_dim = int(binomial(2*L_x*L_y, N))
    state_z = np.zeros((hilbert_dim,hilbert_dim), dtype = complex)
    state_x = np.zeros((hilbert_dim,hilbert_dim), dtype = complex)
    state_y = np.zeros((hilbert_dim,hilbert_dim), dtype = complex)
    for qs_in in itertools.combinations(itertools.product(range(L_x),range(L_y)),N):
        state_z_in = product( c_op(q[0],q[1],1) for q in qs_in ).vector(L_x, L_y, N)

        state_x_in = product( c_op(q[0],q[1],1) + c_op(q[0],q[1],0)
                              for q in qs_in ).vector(L_x, L_y, N)

        state_y_in = product( c_op(q[0],q[1],1) + 1j * c_op(q[0],q[1],0)
                              for q in qs_in ).vector(L_x, L_y, N)

        for qs_out in itertools.combinations(itertools.product(range(L_x),range(L_y)),N):
            state_z_out = product( c_op(q[0],q[1],1) for q in qs_out ).vector(L_x, L_y, N)

            state_x_out = product( c_op(q[0],q[1],1) + c_op(q[0],q[1],0)
                                   for q in qs_out ).vector(L_x, L_y, N)

            state_y_out = product( c_op(q[0],q[1],1) + 1j * c_op(q[0],q[1],0)
                                   for q in qs_out ).vector(L_x, L_y, N)

            state_z += np.outer(state_z_out, state_z_in.conj())
            state_x += np.outer(state_x_out, state_x_in.conj())
            state_y += np.outer(state_y_out, state_y_in.conj())

    state_z /= np.trace(state_z)
    state_x /= np.trace(state_x)
    state_y /= np.trace(state_y)
    return state_z, state_x, state_y


##########################################################################################
# Hamiltonians
##########################################################################################

# lattice Hamiltonian in quasi-momentum basis
def H_lat_q(L_x, L_y, N, J_x, phi_x, J_y = None, phi_y = None):
    if J_y == None: J_y = J_x
    if phi_y == None: phi_y = phi_x
    if L_x == 1: J_x = 0
    if L_y == 1: J_y = 0
    return 2 * sum( ( J_x * np.cos(2*np.pi*q_x/L_x + s * phi_x) +
                      J_y * np.cos(2*np.pi*q_y/L_y + s * phi_y) ) *
                    c_op(q_x,q_y,s).dag() * c_op(q_x,q_y,s)
                    for q_y in range(L_y)
                    for q_x in range(L_x)
                    for s in range(2) ).matrix(L_x, L_y, N)

# lattice Hamiltonian in on-site basis
def H_lat_j(L_x, L_y, N, J_x, phi_x, J_y = None, phi_y = None, periodic = False):
    if J_y == None: J_y = J_x
    if phi_y == None: phi_y = phi_x
    if L_x == 1: J_x = 0
    if L_y == 1: J_y = 0
    hilbert_dim = int(binomial(2*L_x*L_y, N))
    H = np.zeros((hilbert_dim,hilbert_dim), dtype = complex)
    for s in range(2):
        phase_x = np.exp(1j * s * phi_x)
        phase_y = np.exp(1j * s * phi_y)
        for j_x in range(L_x):
            for j_y in range(L_y):
                tun = c_seq()
                if j_x + 1 < L_x or periodic:
                    tun += ( J_x * phase_x *
                             c_op((j_x+1)%L_x, j_y, s).dag() * c_op(j_x, j_y, s) )
                if j_y + 1 < L_y or periodic:
                    tun += ( J_y * phase_y *
                             c_op(j_x, (j_y+1)%L_y, s).dag() * c_op(j_x, j_y, s) )
                H += tun.matrix(L_x, L_y, N)
    return H + H.conjugate().T

# interaction Hamiltonian in quasi-momentum basis
def H_int_q(L_x, L_y, N, U):
    hilbert_dim = int(binomial(2*L_x*L_y, N))
    H = np.zeros((hilbert_dim,hilbert_dim), dtype = complex)
    for p_x, q_x, r_x in itertools.product(range(L_x), repeat = 3):
        s_x = (p_x + q_x - r_x ) % L_x
        for p_y, q_y, r_y in itertools.product(range(L_y), repeat = 3):
            s_y = (p_y + q_y - r_y ) % L_y
            H += ( c_op(p_x,p_y,1).dag() * c_op(q_x,q_y,0).dag() *
                   c_op(r_x,r_y,0) * c_op(s_x,s_y,1) ).matrix(L_x, L_y, N)
    return H * U / (L_x*L_y)

# interaction Hamiltonian in on-site basis
def H_int_j(L_x, L_y, N, U):
    return U * sum( c_op(j_x, j_y, 0).dag() * c_op(j_x, j_y, 1).dag() *
                    c_op(j_x, j_y, 1) * c_op(j_x, j_y, 0)
                    for j_y in range(L_y)
                    for j_x in range(L_x) ).matrix(L_x, L_y, N)


##########################################################################################
# methods for computing spin squeezing
##########################################################################################

# expectation value of an operator with respect to a state (i.e. density operator)
def val(op, state): return np.real(np.trace(op @ state))

# rotate a vector about an axis by a given angle
def rotate_vector(vector, axis, angle):
    L_z = np.array([ [  0,  0,  0 ],
                     [  0,  0, -1 ],
                     [  0,  1,  0 ] ])
    L_x = np.array([ [  0,  0,  1 ],
                     [  0,  0,  0 ],
                     [ -1,  0,  0 ] ])
    L_y = np.array([ [  0, -1,  0 ],
                     [  1,  0,  0 ],
                     [  0,  0,  0 ] ])
    L = ( axis[0] * L_z + axis[1] * L_x + axis[2] * L_y)  / np.linalg.norm(axis)
    return linalg.expm(angle * L) @ vector

# get spin vector <\vec S> for a state
def spin_vec(state, Sz_or_Lx, Sx_or_Ly, Sy_or_N, operators = True):
    Sz, Sx, Sy = get_spin_ops(Sz_or_Lx, Sx_or_Ly, Sy_or_N, operators)
    def val(X): return np.real(np.trace(X @ state))
    return np.real(np.array([ val(Sz), val(Sx), val(Sy) ]))

# get normalized spin vector <\hat S> for a state
def spin_axis(state, Sz_or_Lx, Sx_or_Ly, Sy_or_N, operators = True):
    vec = spin_vec(state, Sz_or_Lx, Sx_or_Ly, Sy_or_N, operators)
    return vec / np.linalg.norm(vec)

# variance of spin state about an axis
def spin_variance(state, axis, Sz_or_Lx, Sx_or_Ly, Sy_or_N, operators = True):
    Sn = Sn_op(axis, Sz_or_Lx, Sx_or_Ly, Sy_or_N, operators)
    def val(X): return np.real(np.trace(X @ state))
    return abs(val(Sn @ Sn) - abs(val(Sn))**2)

# return (\xi^2, axis), where:
#   "axis" is the axis of minimal spin variance in the plane orthogonal to <\vec S>
#   \xi^2 = (<S_axis^2> - <S_axis>^2) / (S/2) is the spin squeezing parameter
def spin_squeezing(state, Sz_or_Lx, Sx_or_Ly, Sy_or_N, operators = True):
    Sz, Sx, Sy = get_spin_ops(Sz_or_Lx, Sx_or_Ly, Sy_or_N, operators)
    state_axis = spin_axis(state, Sz, Sx, Sy)
    if state_axis[1] == 0 and state_axis[2] == 0:
        perp_vec = [0,1,0]
    else:
        rot_axis = np.cross([1,0,0], state_axis)
        perp_vec = rotate_vector(state_axis, rot_axis, np.pi/2)

    def squeezing_axis(eta): # eta is the angle in the plane orthogonal to the spin vector
        return rotate_vector(perp_vec, state_axis, eta)
    def variance(eta):
        return spin_variance(state, squeezing_axis(eta), Sz, Sx, Sy)

    optimum = optimize.minimize_scalar(variance, method = "bounded", bounds = (0, np.pi))
    if not optimum.success:
        sys.exit("squeezing optimization failed")
    optimal_phi = optimum.x
    squeezing_parameter = optimum.fun / (N/4)

    return squeezing_parameter, squeezing_axis(optimal_phi)




##########################################################################################
##########################################################################################

L_x = 1
L_y = 5
N = 5

J_0 = 1
U = 31.85
phi = np.pi

max_time = 10


# L_x = 1
# L_y = 5
# N = 5

# J_0 = 296.6 * 2*np.pi
# U = 1357 * 2*np.pi
# phi = np.pi / 50

# max_time = 1

##########################################################################################

hilbert_dim = int(binomial(2*L_x*L_y, N))
print("hilbert_dim:",hilbert_dim)
H = np.zeros((hilbert_dim,hilbert_dim), dtype = complex)

Sz, Sx, Sy = S_ops(L_x, L_y, N)

state_z, state_x, state_y = polarized_states(L_x, L_y, N)


##########################################################################################


H_q = H_int_q(L_x, L_y, N, U) + H_lat_q(L_x, L_y, N, J_0, phi)
H_j = H_int_j(L_x, L_y, N, U) + H_lat_j(L_x, L_y, N, J_0, phi)


# q_vals = np.real(np.linalg.eigvals(H_q))
# j_vals = np.real(np.linalg.eigvals(H_j))
# q_vals -= np.mean(q_vals)
# j_vals -= np.mean(j_vals)

# plt.plot(sorted(q_vals), label = "Q")
# plt.plot(sorted(j_vals), label = "J")
# plt.legend(loc = "best")
# plt.show()

# exit()


vals_q, vecs_q = np.linalg.eig(H_q)
vals_q = np.real(vals_q)
inv_vecs_q = np.linalg.inv(vecs_q)

vals_j, vecs_j = np.linalg.eig(H_j)
vals_j = np.real(vals_j)
inv_vecs_j = np.linalg.inv(vecs_j)

state_q_0 = inv_vecs_q @ state_x @ vecs_q
Sz_q = inv_vecs_q @ Sz @ vecs_q
Sx_q = inv_vecs_q @ Sx @ vecs_q
Sy_q = inv_vecs_q @ Sy @ vecs_q

state_j_0 = inv_vecs_j @ state_x @ vecs_j
Sz_j = inv_vecs_j @ Sz @ vecs_j
Sx_j = inv_vecs_j @ Sx @ vecs_j
Sy_j = inv_vecs_j @ Sy @ vecs_j

times = np.linspace(0, max_time, 100)
squeezing_j = np.zeros(len(times))
squeezing_q = np.zeros(len(times))
for ii in range(len(times)):
    print("{}/{}".format(ii,len(times)))
    U_q = np.exp(-1j * times[ii] * vals_q)
    U_j = np.exp(-1j * times[ii] * vals_j)
    state_q_t = diag_mult( U_q.conj(), diag_mult(U_q, state_q_0, left = True), left = False)
    state_j_t = diag_mult( U_j.conj(), diag_mult(U_j, state_j_0, left = True), left = False)
    # squeezing_q[ii], _ = spin_squeezing(state_q_t, Sz_q, Sx_q, Sy_q)
    # squeezing_j[ii], _ = spin_squeezing(state_j_t, Sz_j, Sx_j, Sy_j)
    squeezing_q[ii] = val(Sx_q, state_q_t)
    squeezing_j[ii] = val(Sx_j, state_j_t)

# convert squeezing to dB
# squeezing_q = -10*np.log(squeezing_q)/np.log(10)
# squeezing_j = -10*np.log(squeezing_j)/np.log(10)

fig_dir = "../figures/"
figsize = (4,3)
params = { "text.usetex" : True }
plt.rcParams.update(params)

plt.figure(figsize = figsize)
plt.plot(times, squeezing_q, label = "periodic")
plt.plot(times, squeezing_j, label = "closed")
plt.xlim(0, times[-1])
plt.xlabel(r"Time (seconds)")
plt.ylabel(r"Squeezing: $-10\log_{10}(\xi^2)$")
plt.legend(loc="best")
plt.tight_layout()
plt.show()
