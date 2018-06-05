#!/usr/bin/env python3

# FILE CONTENTS: (symbolic) methods relating to interaction Hamiltonians on a 3-D lattice

from sympy import *
from sympy.physics.quantum import TensorProduct as tensor
from mpmath import factorial
from itertools import permutations
from copy import deepcopy

from qubit_methods import *

##########################################################################################
# coupling constants, matrix elements, and fermionic operators
##########################################################################################

# Kronecker delta
def delta(i,j): return 1 if i == j else 0

# coupling constant for ( (l,s), (m,t) ) --> ( (l,ss), (m,tt) ) scattering,
# where l,m are nuclear spin indices; and s,t,ss,tt are pseudospin indices
def g(s,t,ss,tt,couplings):
    # assert that pseudospins are valid and conserved
    assert(s == 1 or s == -1)
    assert(t == 1 or t == -1)
    assert( (s,t) == (ss,tt) or (s,t) == (tt,ss) )

    if s == t:
        if s == -1: return couplings[0] # all spins down
        else: return couplings[-1] # all spins up

    else: # s != t
        if s == tt: return ( couplings[2] - couplings[1] ) / 2 # spin exchange
        else: return ( couplings[2] + couplings[1] ) / 2 # no spin exchange

# coupling constant for ( (l,s), (m,t) ) --> ( (ll,ss), (mm,tt) ) scattering,
# where l,m,ll,mm are nuclear spin indices; and s,t,ss,tt are pseudospin indices
def g_nuc(l,s,m,t,ll,ss,mm,tt,couplings):
    # assert conservation of nuclear spin
    assert( (l,m) == (ll,mm) or (l,m) == (mm,ll) )

    if l == m: # nuclear spins are identical
        if s == t: return 0 # these fermions are identical
        else: return g(s,t,ss,tt,couplings) # regular scattering

    else: # nuclear spins are different
        if l == mm: return -g(s,t,tt,ss,couplings) # nuclear spin exchanged
        if l == ll: return g(s,t,ss,tt,couplings) # nuclear spin not exchanged

# creation / annihilation operators
class c:
    def __init__(self, spin, pseudospin, band = 0, create = False):
        self.nuclear_spin = spin
        self.spin = pseudospin
        self.band = band
        self.create = True if create else False

    def id(self):
        sign = -1 if self.create else 1
        return (self.create, sign * self.band, sign * self.nuclear_spin, sign * self.spin)

    def dag(self):
        return c(self.nuclear_spin, self.spin, self.band, not self.create)

    def __eq__(self, other): return self.id() == other.id()
    def __ne__(self, other): return self.id() != other.id()
    def __lt__(self, other): return self.id() < other.id()
    def __gt__(self, other): return self.id() > other.id()

    def __repr__(self):
        self_id = [ self.band, self.nuclear_spin, self.spin ]
        if self.create:
            self_id += ["dag"]
        return str(self_id)

# product of creation / annihilation operators
class c_vec:
    def __init__(self, c_list, reverse = True):
        if not reverse: self.vec = c_list
        else: self.vec = c_list[::-1]

    def __mul__(self, other):
        return c_vec(other.vec + self.vec, reverse = False)

    # return matrix element corresponding to this product of operators
    def elem(self, atom_number = None, debug = False):
        assert(len(self.vec) % 2 == 0)
        if atom_number == None: atom_number = len(self.vec)//2
        vec = deepcopy(self.vec)

        sign = 1
        current_spin = [ 0 for n in range(atom_number) ]
        current_band = [ 0 for n in range(atom_number) ]
        zero_matrix = zeros(2**atom_number)
        while len(vec) > 0:

            if vec[0].create: return zero_matrix

            if vec[0].band != current_band[vec[0].nuclear_spin]:
                return zero_matrix

            if ( current_spin[vec[0].nuclear_spin] != 0 and
                 vec[0].spin != current_spin[vec[0].nuclear_spin] ):
                return zero_matrix

            found = False # have we found a matching fermionic operator?
            for ii in range(1,len(vec)):

                match = vec[0].nuclear_spin == vec[ii].nuclear_spin

                if match:
                    if not vec[ii].create: return zero_matrix

                    found = True
                    if debug:
                        print()
                        print(vec[0])
                        print(vec[ii])
                        print(vec[::-1])
                    current_spin[vec[0].nuclear_spin] = vec[ii].spin
                    current_band[vec[0].nuclear_spin] = vec[ii].band
                    vec.pop(ii)
                    vec.pop(0)
                    break

                else:
                    sign *= -1

            if not found: return zero_matrix

        if ( 0 in current_spin or
             current_band != [ 0 for n in range(atom_number) ] ):
            return zero_matrix

        if debug:
            print()
            print(self)

        # individual effective spin operators
        spin_ops = [ eye(2) for n in range(atom_number) ]
        for nuclear_spin in set([ operator.nuclear_spin for operator in self.vec ]):
            # loop over annihilation operators
            for operator in self.vec:
                if operator.nuclear_spin == nuclear_spin:
                    s_in = operator.spin
                    break
            # loop over creation operators
            for operator in self.vec[::-1]:
                if operator.nuclear_spin == nuclear_spin:
                    s_out = operator.spin
                    break

            if debug:
                print()
                print(nuclear_spin)
                print(s_in,"-->",s_out)

            spin_ops[nuclear_spin] = qubit_matrix_entry(s_in, s_out)

        if debug:
            print()
            print("sign:",sign)
            pprint(spin_ops)

        return sign * tensor(*spin_ops)

    def __repr__(self):
        return str(self.vec[::-1])

# methods for constructing matrix elements of M-body hamiltonians
def c_vec_g(spins): # ground-state
    assert(len(spins) >= 2)
    c_list = []
    for ss in range(len(spins)):
        c_list += [ c(spins[ss],-1).dag(), c(spins[ss],-1) ]
    return c_vec(c_list).elem()

def c_vec_D(spins): # excited (direct)
    assert(len(spins) >= 2)
    c_list = [ c(spins[0],1).dag(), c(spins[0],1) ]
    for ss in range(1,len(spins)):
        c_list += [ c(spins[ss],-1).dag(), c(spins[ss],-1) ]
    return c_vec(c_list).elem()

def c_vec_X(spins): # excited (exchange)
    assert(len(spins) >= 2)
    c_list = [ c(spins[0],-1).dag(), c(spins[1],1).dag(),
               c(spins[1],-1),  c(spins[0],1) ]
    for ss in range(2,len(spins)):
        c_list += [ c(spins[ss],-1).dag(), c(spins[ss],-1) ]
    return c_vec(c_list).elem()


##########################################################################################
# effective interaction Hamiltonians
# H_n_m is an n-body Hamiltonian at order m in the coupling constants
##########################################################################################

def H_2_1(couplings):
    return ( couplings[0] * dd * dd.H +
             couplings[1] * (ud-du) * (ud-du).H / 2 +
             couplings[2] * (ud+du) * (ud+du).H / 2 +
             couplings[3] * uu * uu.H )

def H_3_2(couplings):

    H_3_2 = zeros(8)

    mu, nu, rho = 0, 1, 2
    for s, t in qubit_states(2):
        for ss, tt in set(permutations([s,t])):
            g_1 = g(s,t,ss,tt,couplings)
            for u, in qubit_states(1):
                for ttt, uu in set(permutations([tt,u])):
                    g_2 = g(tt,u,ttt,uu,couplings)
                    elem = c_vec([ c(mu,ss).dag(), c(nu,ttt).dag(), c(rho,uu).dag(),
                                   c(rho,u), c(nu,t), c(mu,s) ]).elem()
                    H_3_2 += g_1 * g_2 * elem

    return act_on_permutations(H_3_2)

def H_3_3(couplings):

    H_3_3_S = zeros(8) # "star" diagram Hamiltonian
    H_3_3_O = zeros(8) # "OX" diagram Hamiltonian

    k, l, m = range(3)
    for r, s, t in qubit_states(3):
        for rr, ss in set(permutations([r,s])):
            g_1 = g(r,s,rr,ss,couplings)

            # "star" diagram
            for ll, mm in set(permutations([l,m])):
                for sss, tt in set(permutations([ss,t])):
                    g_2 = g_nuc(l,ss,m,t,ll,sss,mm,tt,couplings)
                    for rrr, ssss in set(permutations([rr,sss])):
                        g_3 = g(rr,sss,rrr,ssss,couplings)
                        elem = c_vec([ c(k,rrr).dag(), c(ll,ssss).dag(),
                                       c(mm,tt).dag(), c(m,t),
                                       c(l,s), c(k,r) ]).elem()
                        H_3_3_S += g_1 * g_2 * g_3 * elem

            # "OX" diagram
            for rrr, sss in set(permutations([rr,ss])):
                g_2 = g(rr,ss,rrr,sss,couplings)
                for ssss, tt in set(permutations([sss,t])):
                    g_3 = g(sss,t,ssss,tt,couplings)
                    elem = c_vec([ c(k,rrr).dag(), c(l,ssss).dag(), c(m,tt).dag(),
                                   c(m,t), c(l,s), c(k,r) ]).elem()
                    H_3_3_O += g_1 * g_2 * g_3 * elem

    return [ act_on_permutations(H_3_3_S), act_on_permutations(H_3_3_O) ]

def H_4_3(couplings):

    H_4_3_B = zeros(16) # "branch" diagram Hamiltonian
    H_4_3_C = zeros(16) # "chain" diagram Hamiltonian

    k, l, m, n = range(4)
    for q, r, s, t in qubit_states(4):
        for qq, rr in set(permutations([q,r])):
            g_1 = g(q,r,qq,rr,couplings)
            for qqq, ss in set(permutations([qq,s])):

                # "branch" diagram
                g_2 = g(qq,s,qqq,ss,couplings)
                for rrr, tt in set(permutations([rr,t])):
                    g_3 = g(rr,t,rrr,tt,couplings)
                    elem = c_vec([ c(k,qqq).dag(), c(l,rrr).dag(),
                                   c(m,ss).dag(), c(n,tt).dag(),
                                   c(n,t), c(m,s), c(l,r), c(k,q) ]).elem()
                    H_4_3_B += g_1 * g_2 * g_3 * elem

                # "chain" diagram
                for kk, mm in set(permutations([k,m])):
                    g_2 = g_nuc(k,qq,m,s,kk,qqq,mm,ss,couplings)
                    for qqqq, tt in set(permutations([qqq,t])):
                        g_3 = g(qqq,t,qqqq,tt,couplings)
                        elem = c_vec([ c(kk,qqqq).dag(), c(l,rr).dag(),
                                       c(mm,ss).dag(), c(n,tt).dag(),
                                       c(n,t), c(m,s), c(l,r), c(k,q) ]).elem()
                        H_4_3_C += g_1 * g_2 * g_3 * elem

    return [ act_on_permutations(H_4_3_B), act_on_permutations(H_4_3_C) ]


##########################################################################################
# M-body Hamiltonian coefficients and eigenvalues
##########################################################################################

# factorial and choose functions
def nCk(n,k):
    if k > n: return 0
    if k == n: return 1
    return factorial(n) // factorial(k) // factorial(n-k)

def coefficients_to_eigenvalues_sym(same = False):
    s = symbols
    if same:
        mat = [ [ s(r"M!"), 0, 0 ],
                [ 0, s(r"(M-1)!"), -s(r"(M-2)!") ],
                [ 0, s(r"(M-1)!"), s(r"(M-1)!") ] ]
    else:
        mat = [ [ s(r"M!") * s(r"nCk(N_M)"), 0, 0 ],
                [ s(r"M!") * s(r"nCk(N-1_M)"),
                  s(r"(M-1)!") * s(r"nCk(N-1_M-1)"),
                  -s(r"(M-2)!") * s(r"nCk(N-2_M-2)") ],
                [ s(r"M!") * s(r"nCk(N-1_M)"),
                  s(r"(M-1)!") * s(r"nCk(N-1_M-1)"),
                  s(r"(M-1)!") * s(r"nCk(N-1_M-1)") ] ]
    return Matrix(mat)

def coefficients_to_eigenvalues(M, N = None):
    f = factorial
    if N == None:
        N = M
    mat = [ [ f(M) * nCk(N,M), 0, 0 ],
            [ f(M) * nCk(N-1,M), f(M-1) * nCk(N-1,M-1), -f(M-2) * nCk(N-2,M-2) ],
            [ f(M) * nCk(N-1,M), f(M-1) * nCk(N-1,M-1), f(M-1) * nCk(N-1,M-1) ] ]
    return simplify(Matrix(mat))

# convert M-body eigenvalues to N-body eigenvalues
def convert_eigenvalues(M, N):
    return coefficients_to_eigenvalues(M,N) @ coefficients_to_eigenvalues(M).inv()

# coefficients of an M-body Hamiltonian, sorted as [ U_g, U_D, U_X ]
def sorted_coefficients(H_M):
    M = simplify(log(H_M.shape[0])/log(2))
    return Matrix([ 1/Integer(factorial(M)) * simplify(H_M[0,0]),
                    1/Integer(factorial(M-1)) * simplify(H_M[1,1]),
                    1/Integer(factorial(M-2)) * simplify(H_M[1,2]) ])

# eigenvalues of an M-body Hamiltonian, sorted as [ E_g, E_A, E_S ]
def sorted_eigenvalues(H_M):
    M = simplify(log(H_M.shape[0])/log(2))
    return coefficients_to_eigenvalues(M) @ sorted_coefficients(H_M)
