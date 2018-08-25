#!/usr/bin/env python3

# FILE CONTENTS: methods for computing collective spin correlators

import numpy as np
import scipy.sparse as sparse
import scipy.linalg as linalg

import itertools
from scipy.special import factorial, binom
from sympy.functions.combinatorial.numbers import stirling as sympy_stirling

from squeezing_methods import ln_binom

# unsigned stirling number of the first kind
def stirling(n,k): return float(sympy_stirling(n, k, kind = 1, signed = False))

# coefficient of transverse spin operator S_\mu for matrix element | m+\mu >< m |
def transverse_elem(mu, S, m):
    return np.sqrt((S-mu*m)*(S+mu*m+1))

# correlator < X | S_+^l S_-^m S_z^n | X >
def op_val_X(N, l, m, n):
    if (l,m,n) == (0,0,0): return 1
    S = N/2
    return sum([ k**n
                 * np.exp( ln_binom(N,S+k)/2 + ln_binom(N,S+k+l-m)/2 - N*np.log(2) )
                 * np.prod([ transverse_elem(1,S,k-m+p) for p in range(l) ])
                 * np.prod([ transverse_elem(-1,S,k-p) for p in range(m) ])
                 for k in np.arange(-S+m,S-l) ])

# correlator < -Z | S_+^l S_-^m S_z^n | -Z >
def op_val_nZ(N, l, m, n):
    if (l,m,n) == (0,0,0): return 1
    S = N/2
    z_fac = (-S)**n
    t_fac = sum([ np.exp( ln_binom(N,S+k)/2 + ln_binom(N,S+k+l-m)/2 - N*np.log(2) )
                  * np.prod([ transverse_elem(1,S,k-m+p) for p in range(l) ])
                  * np.prod([ transverse_elem(-1,S,k-p) for p in range(m) ])
                  for k in np.arange(-S+m,S-l) ])
    return z_fac * t_fac

# number of operators with weight less than T
def operators_below_weight(T): return T * (T+1) * (T+2) // 6

# collective spin operator commutator coefficient (see notes)
def epsilon(M,N,P,L):
    return 2**L * np.sum([ (-1)**(P-qq) * stirling(P,qq) * binom(qq,L) * (M-N)**(qq-L)
                           for qq in range(L,P+1) ])

# super-operator for coherent evolution of spin operators
def coherent_evolution_operator(vec_terms, idx, h_terms):
    vec_dim = len(vec_terms)
    op = sparse.dok_matrix((vec_dim,vec_dim))
    for l, m, n in vec_terms:
        idx_out = idx(l,m,n)
        for p, q, a in h_terms.keys():
            term_val = h_terms[(p,q,a)]
            for pp, qq, aa, rr, ss, bb, sign in [ (p,q,a,l,m,n,1), (l,m,n,p,q,a,-1) ]:
                for kk in range(min(qq,rr)+1):
                    val_kk = term_val * sign * factorial(kk) * binom(qq,kk) * binom(rr,kk)
                    for ll in range(kk+1):
                        val_ll = val_kk * (-1)**ll * epsilon(qq,rr,kk,ll)
                        for mm in range(ll+1):
                            val_mm = val_ll * binom(ll,mm) * (-ss)**(ll-mm)
                            for nn in range(aa+1):
                                val = val_mm * binom(aa,nn) * (rr-ss)**(aa-nn)
                                idx_in = idx(pp+rr-kk,qq+ss-kk,bb+mm+nn)
                                if idx_in < vec_dim and val != 0:
                                    op[idx_out,idx_in] += val
    op = op.tocsr()
    op.eliminate_zeros()
    return op

# super-operator for decoherence of spin operators
def decoherence_evolution_operator(vec_terms, idx, N):
    vec_dim = len(vec_terms)
    op = sparse.dok_matrix((vec_dim,vec_dim))
    for l, m, n in vec_terms:
        idx_out = idx(l,m,n)
        for kk in range(n):
            val = (-1)**(n-kk) * binom(n,kk)
            op[idx_out,idx(l,m,kk)] += (N/2-m) * val
            if kk+1 < vec_dim:
                op[idx_out,idx(l,m,kk+1)] += val
        op[idx_out,idx_out] += -1/2 * (l+m)
    return op.tocsr()

# net super-operator for evolution of spin operators
def evolution_operator(vec_terms, idx, N, h_terms, decay_rate_over_chi):
    op_im = coherent_evolution_operator(vec_terms, idx, h_terms)
    op_re = decoherence_evolution_operator(vec_terms, idx, N)
    return 1j * op_im + decay_rate_over_chi * op_re

# return correlators from evolution under a general Hamiltonian
def correlators_general(N, chi_times, decay_rate_over_chi, h_terms, initial_state,
                        max_weight = 30, print_updates = False):
    assert(initial_state in [ "X", "-Z" ])

    B = operators_below_weight
    def R(T,L): return L * (2*T-L+3) // 2
    def idx(l,m,n):
        return B(l+m+n) + R(l+m+n,l) + m

    vec_terms = [ (l, m, T-l-m)
                  for T in range(max_weight)
                  for l, m in itertools.product(range(T+1), repeat = 2)
                  if T - l - m >= 0 ]

    # construct differential super-operator
    diff_op = evolution_operator(vec_terms, idx, N, h_terms, decay_rate_over_chi)

    # construct vector of spin operators and set initial values
    op_vec = np.zeros((len(chi_times),len(vec_terms)), dtype = complex)
    if initial_state == "X":
        initial_val = op_val_X
    if initial_state == "-Z":
        initial_val = op_val_nZ
    for l, m, n in vec_terms:
        op_vec[:,idx(l,m,n)] = initial_val(N, l, m, n)

    # simulate!
    for ii in range(1,len(chi_times)):
        if print_updates:
            print(f"{ii}/{len(chi_times)}")
        dt = chi_times[ii] - chi_times[ii-1]
        op_vec[ii] = sparse.linalg.expm_multiply(dt * diff_op, op_vec[ii-1])

    Sz    = op_vec[:,1]
    Sm    = op_vec[:,2]
    Sp    = op_vec[:,3]
    Sz_Sz = op_vec[:,4]
    Sm_Sz = op_vec[:,5]
    Sp_Sz = op_vec[:,7]
    Sp_Sm = op_vec[:,8]
    Sp_Sp = op_vec[:,9]

    return Sz, Sz_Sz, Sp, Sp_Sz, Sm_Sz, Sp_Sp, Sp_Sm


##########################################################################################
# methods specific to one-axis twisting
##########################################################################################

# return correlators frome evolution under one-axis twisting
def correlators_OAT(N, chi_times, decay_rate_over_chi,
                    max_weight = 30, print_updates = False):

    h_terms = { (0,0,2) : 1 }
    def vec_terms(l,m):
        return [ (l, m, T-l-m) for T in range(max_weight) if T - l - m >= 0 ]
    def idx(l,m,n): return n

    # construct differential super-operators
    diff_op_p  = evolution_operator(vec_terms(1,0), idx, N, h_terms, decay_rate_over_chi)
    diff_op_m  = evolution_operator(vec_terms(0,1), idx, N, h_terms, decay_rate_over_chi)
    diff_op_pp = evolution_operator(vec_terms(2,0), idx, N, h_terms, decay_rate_over_chi)
    diff_op_pm = evolution_operator(vec_terms(1,1), idx, N, h_terms, decay_rate_over_chi)

    # construct vectors of spin corelators
    p_vec  = np.zeros((len(chi_times),max_weight-1), dtype = complex)
    m_vec  = np.zeros((len(chi_times),max_weight-1), dtype = complex)
    pp_vec = np.zeros((len(chi_times),max_weight-2), dtype = complex)
    pm_vec = np.zeros((len(chi_times),max_weight-2), dtype = complex)

    # set initial values for all spin operators
    p_vec[0]  = [ op_val_X(N, 1, 0, n) for n in range(len(p_vec[0]))  ]
    m_vec[0]  = [ op_val_X(N, 0, 1, n) for n in range(len(m_vec[0]))  ]
    pp_vec[0] = [ op_val_X(N, 2, 0, n) for n in range(len(pp_vec[0])) ]
    pm_vec[0] = [ op_val_X(N, 1, 1, n) for n in range(len(pm_vec[0])) ]

    # simulate!
    for ii in range(1,len(chi_times)):
        if print_updates:
            print(f"{ii}/{len(chi_times)}")
        dt = chi_times[ii] - chi_times[ii-1]
        p_vec[ii]  = sparse.linalg.expm_multiply(dt * diff_op_p,  p_vec[ii-1])
        m_vec[ii]  = sparse.linalg.expm_multiply(dt * diff_op_m,  m_vec[ii-1])
        pp_vec[ii] = sparse.linalg.expm_multiply(dt * diff_op_pp, pp_vec[ii-1])
        pm_vec[ii] = sparse.linalg.expm_multiply(dt * diff_op_pm, pm_vec[ii-1])

    Sp    = p_vec[:,0]
    Sp_Sz = p_vec[:,1]
    Sm_Sz = m_vec[:,1]
    Sp_Sp = pp_vec[:,0]
    Sp_Sm = pm_vec[:,0]

    decay_exp = np.exp( - decay_rate_over_chi * chi_times )
    Sz = N/2 * (decay_exp - 1)
    var_Sz = N/2 * (1 - decay_exp/2) * decay_exp
    Sz_Sz = var_Sz + Sz**2

    return Sz, Sz_Sz, Sp, Sp_Sz, Sm_Sz, Sp_Sp, Sp_Sm


##########################################################################################
# methods specific to two-axis twisting
##########################################################################################

# return correlators from evolution under (y-z)-type two-axis twisting
def correlators_TAT_yz(N, chi_times, decay_rate_over_chi,
                       max_weight = 30, print_updates = False):
    h_terms = { (0,0,2): 1,
                (2,0,0): 1/4,
                (0,2,0): 1/4,
                (1,1,0): -1/2,
                (0,0,1): 1/2 }
    for key in h_terms.keys():
        h_terms[key] /= 3

    def B(p,T):
        return ( operators_below_weight(T) + (T+1)//2 * (T%2) * (1-2*p) ) // 2
    def R(p,T,n):
        return n//2 * ( T - n//2 + 2 - (T+p)%2 )
    def idx(p,l,m,n):
        return B(p,l+m+n) + R(p,l+m+n,n) + m
    def idx_0(l,m,n): return idx(0,l,m,n)
    def idx_1(l,m,n): return idx(1,l,m,n)

    vec_terms_0 = [ (l, m, T-l-m)
                    for T in range(max_weight)
                    for l, m in itertools.product(range(T+1), repeat = 2)
                    if (l - m) % 2 == 0
                    if T - l - m >= 0 ]
    vec_terms_1 = [ (l, m, T-l-m)
                    for T in range(max_weight)
                    for l, m in itertools.product(range(T+1), repeat = 2)
                    if (l - m) % 2 == 1
                    if T - l - m >= 0 ]

    # construct differential super-operators
    diff_op_0 = evolution_operator(vec_terms_0, idx_0, N, h_terms, decay_rate_over_chi)
    diff_op_1 = evolution_operator(vec_terms_1, idx_1, N, h_terms, decay_rate_over_chi)

    # construct vector of spin operators and set initial values
    op_vec_0 = np.zeros((len(chi_times),len(vec_terms_0)), dtype = complex)
    op_vec_1 = np.zeros((len(chi_times),len(vec_terms_1)), dtype = complex)
    for l, m, n in vec_terms_0:
        op_vec_0[:,idx_0(l,m,n)] = op_val_X(N, l, m, n)
    for l, m, n in vec_terms_1:
        op_vec_1[:,idx_1(l,m,n)] = op_val_X(N, l, m, n)

    # simulate!
    for ii in range(1,len(chi_times)):
        if print_updates:
            print(f"{ii}/{len(chi_times)}")
        dt = chi_times[ii] - chi_times[ii-1]
        op_vec_0[ii] = sparse.linalg.expm_multiply(dt * diff_op_0, op_vec_0[ii-1])
        op_vec_1[ii] = sparse.linalg.expm_multiply(dt * diff_op_1, op_vec_1[ii-1])

    Sz    = op_vec_0[:,1]
    Sm    = op_vec_1[:,1]
    Sp    = op_vec_1[:,0]
    Sz_Sz = op_vec_0[:,5]
    Sm_Sz = op_vec_1[:,3]
    Sp_Sz = op_vec_1[:,2]
    Sp_Sm = op_vec_0[:,3]
    Sp_Sp = op_vec_0[:,2]

    return Sz, Sz_Sz, Sp, Sp_Sz, Sm_Sz, Sp_Sp, Sp_Sm
