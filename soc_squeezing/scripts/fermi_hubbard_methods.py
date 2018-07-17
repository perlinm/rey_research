#!/usr/bin/env python3

# FILE CONTENTS: (symbolic) methods relating to fermion squeezing

import sys, time
import numpy as np
import scipy.sparse as sparse
import scipy.linalg as linalg
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import itertools, functools, operator

from scipy.special import binom as binomial
from sympy.combinatorics.permutations import Permutation as sympy_permutation

np.set_printoptions(linewidth = 200)

# return product of items in list
def sum(list):
    try:
        return functools.reduce(operator.add, list)
    except TypeError:
        return 0
def product(list):
    try:
        return functools.reduce(operator.mul, list)
    except TypeError:
        return 1

# convert value to decibels
def to_dB(x): return 10*np.log10(x)


##########################################################################################
# objects and methods for constructing operators
##########################################################################################

# return iterator for all momentum states for given lattice dimensions
def momenta(L): return itertools.product(*[ range(L_j) for L_j in L ])

# iterable for basis of fock states; returns index of fock state
#   and a tuple specifying the occupied single-particle states (by index)
def fock_state_basis(L, N):
    fock_states = int(binomial(2*product(L),N))
    return zip(range(fock_states), itertools.combinations(range(2*product(L)),N))

# get operator representation of annihilation operators in the fock basis
# depth is the number of times we need to apply annihilation operators
def get_c_op_mats(L, N, depth = None):
    if depth == None: depth = N
    single_particle_states = 2*product(L)
    c_op_mats = [ None ] * single_particle_states * depth

    for dd in range(depth):
        # determine dimensions of input and output hilbert spaces
        dim_in = int(binomial(single_particle_states, N-dd))
        dim_out = int(binomial(single_particle_states, N-dd-1))

        for state_num in range(single_particle_states):
            # initialize zero matrix for an annihilation operator addressing this state
            matrix = sparse.dok_matrix((dim_out,dim_in), dtype = int)

            # loop over all states in the input fock space
            for index_in, states_in in fock_state_basis(L, N-dd):
                if state_num not in states_in: continue

                # determine which single particle states are still occupied after
                #   applying the annihilation operator
                remaining_states = tuple( state for state in states_in
                                          if state != state_num )
                # determine whether we pick up a sign upon annihilation
                sign = sum( 1 for state in states_in if state > state_num ) % 2

                # loop over all states in the output fock space
                for index_out, states_out in fock_state_basis(L, N-dd-1):
                    if states_out == remaining_states:
                        matrix[index_out, index_in] = (-1)**sign

            # store the matrix for this annihilation operator
            c_op_mats[single_particle_states*dd + state_num] = matrix.tocsr()

    return c_op_mats

# creation / annihilation operator class
class c_op:
    def __init__(self, q, spin_up, creation = False):
        self.q = np.array(q).astype(int)
        self.spin_up = bool(spin_up)
        self.creation = bool(creation)

    def __repr__(self):
        spin_text = "up" if self.spin_up else "dn"
        if not self.creation:
            return str((list(self.q), spin_text))
        else:
            return str((list(self.q), spin_text, "dag"))

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
        return (not self.creation, tuple(sign * self.q), sign * self.spin_up)

    # return hermitian conjugate
    def dag(self):
        return c_op(self.q, self.spin_up, not self.creation)

    # index of the state addressed by this operator
    def index(self, L):
        site_index = sum( self.q[ii] * product( L[jj] for jj in range(ii+1,len(L)) )
                          for ii in range(len(self.q)) )
        return 2 * site_index + self.spin_up

    def vector(self, L, N):
        return c_seq(self).vector(L, N)

# product of creation / annihilation operators
class c_seq:
    def __init__(self, sequence = [], prefactor = 1):
        if type(sequence) is c_op:
            self.seq = [sequence]
        else:
            self.seq = sequence
        self.prefactor = prefactor

    def __repr__(self):
        op_text = " * ".join([ str(item) for item in self.seq ])
        return "+ {} * {}".format(self.prefactor, op_text)

    def __eq__(self, other):
        assert(type(other) is c_seq)
        if len(self.seq) != len(other.seq):
            return False
        if self.prefactor != other.prefactor and self.prefactor != -other.prefactor:
            return False
        self.sort()
        other.sort()
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
        return c_seq([ item.dag() for item in self.seq[::-1] ], np.conj(self.prefactor))

    # return vector or matrix corresponding to this product of operators
    def matrix(self, L, N, c_op_mats = None):
        assert(N <= 2*product(L)) # we cannot have more particles than states

        num_ops = len(self.seq)
        assert(num_ops % 2 == 0) # we must have an even number of operators

        # determine dimension of hilbert space
        hilbert_dim = int(binomial(2*product(L), N))
        matrix_shape = (hilbert_dim, hilbert_dim)

        # if we have an empty sequence, return the zero matrix
        if num_ops == 0: return sparse.csr_matrix(matrix_shape, dtype = int)

        # to strictly enforce conservation of particle number, we make sure that
        #   we have the same number of creation operators as annihilation operators
        assert( len([ op for op in self.seq if op.creation ]) ==
                len([ op for op in self.seq if not op.creation ]) )

        # sort all operators in a standard order
        self.sort()

        # identify creation / destruction operators and their indices
        created_states = [ op.index(L) for op in self.seq[:num_ops//2] ]
        destroyed_states = [ op.index(L) for op in self.seq[num_ops//2:][::-1] ]

        # if we address any states twice, return the zero matrix
        if len(set(created_states)) + len(set(destroyed_states)) != num_ops:
            return sparse.csr_matrix(matrix_shape, dtype = int)

        # if we provided matrix representations of the fermionic operators, use them!
        if c_op_mats != None:
            if len(c_op_mats) < (2*product(L))*(num_ops//2):
                error_msg = "we need {} operators, but have only {}!"
                sys.exit(error_msg.format((2*product(L))*(num_ops//2),len(c_op_mats)))

            op_list = ( [ c_op_mats[(2*product(L))*ii + created_states[ii]].T
                          for ii in range(num_ops//2) ] +
                        [ c_op_mats[(2*product(L))*ii + destroyed_states[ii]]
                          for ii in range(num_ops//2) ][::-1] )

            matrix = functools.reduce(sparse.csr_matrix.dot, op_list)
            return self.prefactor * matrix

        # we do not have a matrix representation of fermionic operators, so we have
        #   to "manually" loop over all elements of the fock basis to construct a matrix
        diagonal_term = created_states == destroyed_states
        matrix = sparse.dok_matrix(matrix_shape, dtype = int)
        for index_in, states_in in fock_state_basis(L, N):
            # if this combination of single particle states is not addressed
            #   by the destruction operators, continue to next combination
            if any([ state not in states_in for state in destroyed_states ]): continue

            # if this term is diagonal in the fock basis and we address this state,
            #   set the diagonal matrix element to 1
            if diagonal_term:
                matrix[index_in, index_in] = 1
                continue

            # determine which will states remain occupied after destruction
            remaining_states_in = [ state for state in states_in
                                    if state not in destroyed_states ]

            # count the number of minus signs we get from applying destruction operators
            signs_in = sum( sum( 1 for state in states_in
                                 if state > destroyed_state )
                            for destroyed_state in destroyed_states )

            # otherwise, we need to look at all off-diagonal matrix elements
            for index_out, states_out in fock_state_basis(L, N):
                if any([ state not in states_out for state in created_states ]): continue

                remaining_states_out = [ state for state in states_out
                                         if state not in created_states ]

                # if the remaining input/output states are different
                #   this matrix element is zero
                if remaining_states_in != remaining_states_out: continue

                signs_out = sum( sum( 1 for state in states_out
                                      if state > created_state )
                                 for created_state in created_states )

                # set this matrix element to 1 or -1 appropriately
                matrix[index_out, index_in] = (-1)**(signs_in + signs_out)

        return self.prefactor * matrix.tocsr()

    # return fock state created or destroyed by the given sequence of fermionic operators
    def vector(self, L, N):
        num_ops = len(self.seq)
        # assert we have the same nubmer of operators as particles
        assert(N == num_ops)

        # determine dimension of hilbert space
        hilbert_dim = int(binomial(2*product(L), N))

        # if we have an empty sequence, return the zero matrix
        if num_ops == 0: return sparse.csr_matrix((hilbert_dim,1), dtype = int)

        # assert that all operators are of the same type (i.e. creation / annihilation)
        assert( len(set( op.creation for op in self.seq )) == 1 )

        # sort operators, picking up a sign in the prefactor if necessary
        self.sort()

        # determine which sinle-particle states will be occupied
        occupied_states = tuple( op.index(L) for op in self.seq[::-1] )

        # if any operators are repeats, return the zero state
        if len(occupied_states) != len(set(occupied_states)):
            return sparse.csr_matrix((hilbert_dim,1), dtype = int)

        # loop over the fock basis to determine the appropriate vector
        for index, states in fock_state_basis(L, N):
            if states == occupied_states:
                data = [self.prefactor]
                location = ([index],[0])
                return sparse.csr_matrix((data, location), (hilbert_dim,1))

        sys.exit("state not found in fock basis...")

# sum of products of creation / annihilation operators
class c_sum:
    def __init__(self, item_list = None):
        if item_list == None or item_list == []:
            self.seq_list = []
            return None
        # assert that either all items are operators (c_op)
        #   or all items are sequences of operators (c_seq)
        assert(all( type(item) is c_op for item in item_list ) or
               all( type(item) is c_seq for item in item_list ))
        if type(item_list[0]) is c_op:
            self.seq_list = [ c_seq(item) for item in item_list ]
        else: # if items are of type c_seq
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

    def matrix(self, L, N, c_op_mats = None):
        return sum( item.matrix(L, N, c_op_mats) for item in self.seq_list )

    def vector(self, L, N):
        return sum( item.vector(L, N) for item in self.seq_list )


##########################################################################################
# useful operators
##########################################################################################

# collective spin operators for N particles on a lattice with dimensions L
def spin_op_z(L, N, c_op_mats = None):
    return sum( c_op(q,1).dag() * c_op(q,1) - c_op(q,0).dag() * c_op(q,0)
                for q in momenta(L) ).matrix(L, N, c_op_mats) / 2

def spin_op_m(L, N, c_op_mats = None):
    return sum( c_op(q,0).dag() * c_op(q,1)
              for q in momenta(L) ).matrix(L, N, c_op_mats)

def spin_op_x(L, N, c_op_mats = None):
    Sm = spin_op_m(L, N, c_op_mats)
    return ( Sm + Sm.getH() ) / 2

def spin_op_y(L, N, c_op_mats = None):
    Sm = spin_op_m(L, N, c_op_mats)
    return ( Sm - Sm.getH() ) * 1j / 2

def spin_op_vec_mat(L, N, c_op_mats = None):
    Sz = spin_op_z(L, N, c_op_mats)
    Sm = spin_op_m(L, N, c_op_mats)
    Sx = ( Sm + Sm.getH() ) / 2
    Sy = ( Sm - Sm.getH() ) * 1j / 2

    Szz = sparse.csr_matrix.dot(Sz, Sz)
    Sxx = sparse.csr_matrix.dot(Sx, Sx)
    Syy = sparse.csr_matrix.dot(Sy, Sy)
    Sxy = sparse.csr_matrix.dot(Sx, Sy)
    Syz = sparse.csr_matrix.dot(Sy, Sz)
    Szx = sparse.csr_matrix.dot(Sz, Sx)

    S_op_vec = [ Sz, Sx, Sy ]
    SS_op_mat = [ [ Szz,        Szx,        Syz.getH() ],
                  [ Szx.getH(), Sxx,        Sxy        ],
                  [ Syz,        Sxy.getH(), Syy        ] ]

    return S_op_vec, SS_op_mat

# density operators with completely mixed spatial degees of freedom,
#   but all spins pointing along principal axes
def polarized_states(L, N):
    # if we are at unit filling, return a state vector, otherwise return a density operator
    if N == product(L):
        vec_z = product( c_op(q,1) for q in momenta(L) ).vector(L, N)
        vec_x = product( c_op(q,1) + c_op(q,0) for q in momenta(L) ).vector(L, N)
        vec_y = product( c_op(q,1) + 1j * c_op(q,0) for q in momenta(L) ).vector(L, N)
        vec_z = vec_z.toarray() / sparse.linalg.norm(vec_z)
        vec_x = vec_x.toarray() / sparse.linalg.norm(vec_x)
        vec_y = vec_y.toarray() / sparse.linalg.norm(vec_y)
        return vec_z, vec_x, vec_y

    hilbert_dim = int(binomial(2*product(L), N))
    state_z = sparse.csr_matrix((hilbert_dim,hilbert_dim), dtype = float)
    state_x = sparse.csr_matrix((hilbert_dim,hilbert_dim), dtype = float)
    state_y = sparse.csr_matrix((hilbert_dim,hilbert_dim), dtype = complex)
    for momenta_comb in itertools.combinations(momenta(L), N):
        vec_z = product( c_op(q,1) for q in momenta_comb ).vector(L, N)
        vec_x = product( c_op(q,1) + c_op(q,0) for q in momenta_comb ).vector(L, N)
        vec_y = product( c_op(q,1) + 1j * c_op(q,0) for q in momenta_comb ).vector(L, N)
        state_z += vec_z * vec_z.conj().T
        state_x += vec_x * vec_x.conj().T
        state_y += vec_y * vec_y.conj().T

    state_z /= state_z.diagonal().sum()
    state_x /= state_x.diagonal().sum()
    state_y /= state_y.diagonal().sum()
    return state_z.toarray(), state_x.toarray(), state_y.toarray()


##########################################################################################
# Hamiltonians
##########################################################################################

# lattice Hamiltonian in quasi-momentum basis
def H_lat_q(L, N, J, phi, c_op_mats = None):
    try: len(J)
    except: J = J * np.ones(len(L))
    try: len(phi)
    except: phi = phi * np.ones(len(L))
    return 2 * sum( sum( J[ii] * np.cos(2*np.pi*q[ii]/L[ii] + s * phi[ii])
                         for ii in range(len(L)) if L[ii] > 1 ) *
                    c_op(q,s).dag() * c_op(q,s)
                    for q in momenta(L) for s in range(2) ).matrix(L, N, c_op_mats)

# lattice Hamiltonian in on-site basis
def H_lat_j(L, N, J, phi, c_op_mats = None, periodic = False):
    try: len(J)
    except: J = J * np.ones(len(L))
    try: len(phi)
    except: phi = phi * np.ones(len(L))
    H = c_seq()
    for j, s in itertools.product(momenta(L), range(2)):
        for ii in range(len(j)):
            if L[ii] > 1 and ( j[ii] + 1 < L[ii] or periodic ):
                j_add = np.zeros(len(j))
                j_add[ii] = 1
                H += ( J[ii] * np.exp(1j * s * phi[ii]) *
                       c_op(j+j_add, s).dag() * c_op(j, s) )
    H = H.matrix(L, N, c_op_mats)
    return H + H.getH()

# interaction Hamiltonian in quasi-momentum basis
def H_int_q(L, N, U, c_op_mats = None):
    H = c_seq()
    for p in momenta(L):
        for q in momenta(L):
            for r in momenta(L):
                s = ( np.array(p) + np.array(q) - np.array(r) ) % np.array(L)
                H += c_op(p,1).dag() * c_op(q,0).dag() * c_op(r,0) * c_op(s,1)
    return U / (product(L)) * H.matrix(L, N, c_op_mats)

# interaction Hamiltonian in on-site basis
def H_int_j(L, N, U, c_op_mats = None):
    return U * sum( c_op(j, 0).dag() * c_op(j, 1).dag() * c_op(j, 1) * c_op(j, 0)
                    for j in momenta(L) ).matrix(L, N, c_op_mats)


##########################################################################################
# methods for computing spin squeezing
##########################################################################################

# expectation value with sparse matrices
def val(X, state):
    if X.shape != state.shape: # we have a state vector
        return state.conj().T.dot(X.dot(state)).sum()
    else: # we have a density operator
        return X.multiply(state).sum()

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
    L = ( axis[0] * L_z + axis[1] * L_x + axis[2] * L_y)  / linalg.norm(axis)
    return linalg.expm(angle * L) @ vector

# variance of spin state about an axis
def spin_variance(state, axis, S_op_vec, SS_op_mat):
    S_vec = np.array([ np.real(val(X,state)) for X in S_op_vec ])
    SS_mat = np.array([ [ np.real(val(X,state)) for X in XS ] for XS in SS_op_mat ])
    return np.real(axis @ SS_mat @ axis - abs(S_vec @ axis)**2)

# return (\xi^2, axis), where:
#   "axis" is the axis of minimal spin variance in the plane orthogonal to <\vec S>
#   \xi^2 = N (<S_axis^2> - <S_axis>^2) / |<S>|^2 is the spin squeezing parameter
def spin_squeezing(state, S_op_vec, SS_op_mat):
    S_vec = np.array([ np.real(val(X,state)) for X in S_op_vec ])
    SS_mat = np.array([ [ np.real(val(X,state)) for X in XS ] for XS in SS_op_mat ])

    if S_vec[1] == 0 and S_vec[2] == 0:
        perp_vec = [0,1,0]
    else:
        rot_axis = np.cross([1,0,0], S_vec)
        perp_vec = rotate_vector(S_vec, rot_axis, np.pi/2) / linalg.norm(S_vec)

    def squeezing_axis(eta): # eta = angle in plane orthogonal to the spin vector
        return rotate_vector(perp_vec, S_vec, eta)
    def variance(eta):
        axis = squeezing_axis(eta)
        return np.real(axis @ SS_mat @ axis)

    optimum = optimize.minimize_scalar(variance, method = "bounded", bounds = (0, np.pi))
    if not optimum.success: sys.exit("squeezing optimization failed")

    optimal_phi = optimum.x
    minimal_variance = optimum.fun
    squeezing_parameter = minimal_variance * N / linalg.norm(S_vec)**2

    return squeezing_parameter, squeezing_axis(optimal_phi)

##########################################################################################
##########################################################################################

def plot_eigvals(M):
    vals = sorted(np.real(np.linalg.eigvals(M.toarray())))
    vals -= np.mean(vals)
    plt.plot(vals)
    return vals


time_steps = 100


L = (1,6)
N = 6

J_0 = 1
U = 25
Omega = 35
phi = np.pi

max_time = 10


# L_x = 1
# L_y = 6
# N = 6

# J_0 = 296.6 * 2*np.pi
# U = 1357 * 2*np.pi
# Omega = 0
# phi = np.pi / 50

# max_time = 1



hilbert_dim = int(binomial(2*product(L), N))
print("hilbert_dim:",hilbert_dim)

c_op_mats = get_c_op_mats(L, N)
S_op_vec, SS_op_mat = spin_op_vec_mat(L, N, c_op_mats)
state_z, state_x, state_y = polarized_states(L, N)

H_j = ( H_int_j(L, N, U, c_op_mats) +
        H_lat_j(L, N, J_0, phi, c_op_mats) )

H_q = ( H_int_q(L, N, U, c_op_mats) +
        H_lat_q(L, N, J_0, phi, c_op_mats) )

if Omega != 0:
    H_drive = - Omega * S_op_vec[2]
    H_q += H_drive
    H_j += H_drive

state_j = state_x
state_q = state_x
using_state_vectors = state_q.shape != H_q.shape

squeezing_j = np.zeros(time_steps)
squeezing_q = np.zeros(time_steps)
times = np.linspace(0, max_time, time_steps)
dt = max_time / time_steps
for ii in range(time_steps):
    squeezing_j[ii], _ = spin_squeezing(state_j, S_op_vec, SS_op_mat)
    squeezing_q[ii], _ = spin_squeezing(state_q, S_op_vec, SS_op_mat)
    state_j = sparse.linalg.expm_multiply(-1j*dt*H_j, state_j)
    state_q = sparse.linalg.expm_multiply(-1j*dt*H_q, state_q)
    if not using_state_vectors:
        state_j = sparse.linalg.expm_multiply(-1j*dt*H_j, state_j.conj().T).conj().T
        state_q = sparse.linalg.expm_multiply(-1j*dt*H_q, state_q.conj().T).conj().T

fig_dir = "../figures/"
figsize = (4,3)
params = { "text.usetex" : True }
plt.rcParams.update(params)

plt.figure(figsize = figsize)
plt.plot(times, -to_dB(squeezing_j), label = "closed")
plt.plot(times, -to_dB(squeezing_q), label = "periodic")
plt.xlim(0, times[-1])
plt.xlabel(r"Time (seconds)")
plt.ylabel(r"Squeezing: $-10\log_{10}(\xi^2)$")
plt.legend(loc="best")
plt.tight_layout()

import sys
if "show" in sys.argv: plt.show()
