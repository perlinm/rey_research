#!/usr/bin/env python3

# FILE CONTENTS: methods for computing collective spin correlators

import numpy as np
import scipy.sparse as sparse

from scipy.special import factorial, binom
from sympy.functions.combinatorial.numbers import stirling as sympy_stirling

from squeezing_methods import ln_factorial

# unsigned stirling number of the first kind
def stirling(n,k): return float(sympy_stirling(n, k, kind = 1, signed = False))

# coefficient of transverse spin operator S_\mu for matrix element | m+\mu >< m |
def transverse_elem(mu, S, m):
    return np.sqrt((S-mu*m)*(S+mu*m+1))

# correlator < X | S_+^l S_z^m S_-^n | X >
def op_val_pX(N, op):
    if op == (0,0,0): return 1
    l, m, n = op
    S = N/2
    def ln_factors(k,l,n):
        numerator = ln_factorial(N) + ln_factorial(S-k)
        denominator = ln_factorial(S+k) + ln_factorial(S-k-l) + ln_factorial(S-k-n)
        return numerator - denominator
    return sum([ k**m * np.exp( ln_factors(k,l,n) - N*np.log(2) )
                 for k in np.arange(-S,S-max(l,n)+1) ])

# collective spin operator commutator coefficients (see notes)
def epsilon(m,n,p,l):
    return 2**l * np.sum([ (-1)**(p-qq) * stirling(p,qq) * binom(qq,l) * (m-n)**(qq-l)
                           for qq in range(l,p+1) ])
def xi(m,n,p,q):
    return sum([ (-1)**(ll-q) * epsilon(m,n,p,ll) * binom(ll,q) * (m-p)**(ll-q)
                 for ll in range(q,p+1) ])

# compute pre-image of a single operator from coherent evolution
# operators in (L,M,N) format, and pre-image in dictionary format
def coherent_op_image(op, h_vals):
    if op == (0,0,0): return {}
    L, M, N = op
    image = {}
    for P,Q,R in h_vals.keys():
        h_term = h_vals[(P,Q,R)]
        for pp, aa, qq, rr, bb, ss, sign in [ (P,Q,R,L,M,N,1), (L,M,N,P,Q,R,-1) ]:
            for kk in range(min(qq,rr)+1):
                kk_fac = factorial(kk) * binom(qq,kk) * binom(rr,kk)
                for ll in range(kk+1):
                    kl_fac = kk_fac * (-1)**ll * xi(qq,rr,kk,ll)
                    for mm in range(aa+1):
                        klm_fac = kl_fac * (rr-kk)**(aa-mm) * binom(aa,mm)
                        for nn in range(bb+1):
                            klmn_fac = klm_fac * (qq-kk)**(bb-nn) * binom(bb,nn)
                            val = 1j * sign * h_term * klmn_fac
                            op_in = (pp+rr-kk, ll+mm+nn, qq+ss-kk)
                            try: image[op_in]
                            except: image[op_in] = 0
                            image[op_in] += val
    return image

# compute pre-image of a single operator from decoherence
# operators in (L,M,N) format, and pre-image in dictionary format
def decoherence_op_image(op, spin_number):
    if op == (0,0,0): return {}
    L, M, N = op
    val_kk = np.array([ (-1)**(M-kk) * binom(M,kk) for kk in range(M) ])
    image = { (L,kk,N) : spin_number/2 * val_kk[kk] for kk in range(M) }
    image[(L,M,N)] = -1/2*(L+N)
    for kk in range(M): image[(L,kk+1,N)] += val_kk[kk]
    return image

# compute pre-image of a single operator from infinitesimal time evolution
# operators in (L,M,N) format, and pre-image in dictionary format
def op_image(op, h_vals, spin_number, decay_rate_over_chi):
    image = coherent_op_image(op, h_vals)
    if decay_rate_over_chi != 0:
        image_dec = decoherence_op_image(op, spin_number)
        for key in image_dec.keys():
            try: image[key]
            except: image[key] = 0
            image[key] += decay_rate_over_chi * image_dec[key]
    null_keys = [ key for key in image.keys() if abs(image[key]) == 0 ]
    for key in null_keys: del image[key]
    return image

# generator of time translation for a vector of spin operators
def evolution_operator(vec_terms, idx, h_vals,  spin_number, decay_rate_over_chi):
    vec_dim = len(vec_terms)
    op_mat = sparse.dok_matrix((vec_dim,vec_dim), dtype = complex)
    for op_out in vec_terms:
        idx_out = idx(*op_out)
        image = op_image(op_out, h_vals, spin_number, decay_rate_over_chi)
        for op_in in image.keys():
            idx_in = idx(*op_in)
            if idx_in < vec_dim:
                op_mat[idx_out,idx_in] += image[op_in]
    return op_mat.tocsr()

# compute time derivative of a given vector of spin operators
def compute_time_derivative(diff_op, input_vector, op_image_args):
    output_vector = {}
    for input_op in input_vector.keys():
        try: diff_op[input_op]
        except: diff_op[input_op] = op_image(input_op, *op_image_args)
        for output_op in diff_op[input_op]:
            try: output_vector[output_op]
            except: output_vector[output_op] = 0
            output_vector[output_op] \
                += diff_op[input_op][output_op] * input_vector[input_op]
    return output_vector

# return correlators from evolution under a general Hamiltonian
def compute_correlators(N, chi_times, decay_rate_over_chi, h_vals, initial_state,
                        order_cap = 30):
    assert(initial_state in [ "+X", "-Z" ])
    if initial_state == "+X":
        initial_val = op_val_pX
    if initial_state == "-Z":
        initial_val = op_val_nZ

    # list of operators necessary for computing squeezing, namely:
    #                    Sz     S_z^2,     Sp     S_+^2   S_+ S_z  S_+ S_-
    squeezing_ops = [ (0,1,0), (0,2,0), (1,0,0), (2,0,0), (1,1,0), (1,0,1) ]

    # arguments for computing operator pre-image under infinitesimal time translation
    op_image_args = ( h_vals, N, decay_rate_over_chi )

    diff_op = {} # generator of time translations
    time_derivatives = {} # [ sqz_op ][ derivative_order ][ operator ] --> value
    for sqz_op in squeezing_ops:
        time_derivatives[sqz_op] = { 0 : { sqz_op : 1 } }
        for order in range(1,order_cap):
            time_derivatives[sqz_op][order] \
                = compute_time_derivative(diff_op,
                                          time_derivatives[sqz_op][order-1],
                                          op_image_args)

    # compute initial values of relevant operators
    relevant_ops = set.union(*[ set(time_derivatives[sqz_op][order].keys())
                                for sqz_op in squeezing_ops
                                for order in range(order_cap) ])
    initial_vals = { op : initial_val(N, op) for op in relevant_ops }

    T = np.array([ chi_times**kk / factorial(kk) for kk in range(order_cap) ])
    Q = {} # dictionary (l,m,n) --> < D_t^kk S_+^l S_z^m S_-^n >_0 for all kk
    vals = {}
    for sqz_op in squeezing_ops:
        Q[sqz_op] = np.array([ sum([ time_derivatives[sqz_op][order][op] * initial_vals[op]
                                     for op in time_derivatives[sqz_op][order].keys() ])
                               for order in range(order_cap) ])
        vals[sqz_op] = Q[sqz_op] @ T

    return [ vals[sqz_op] for sqz_op in squeezing_ops ]

# exact correlators for OAT with decoherence
# derivations in foss-feig2013nonequilibrium
def correlators_OAT(N, chi_t, decay_rate_over_chi):
    g = decay_rate_over_chi # shorthand for decay rate in units with \chi = 1
    t = chi_t # shorthand for time in units with \chi = 1

    Sz = N/2 * (np.exp(-g*t)-1)
    var_Sz = N/2 * (1 - np.exp(-g*t)/2) * np.exp(-g*t)
    Sz_Sz = var_Sz + Sz**2

    def s(J): return J + 1j*g/2
    def Phi(J): return np.exp(-g*t/2) * ( np.cos(s(J)*t) + g*t/2 * np.sinc(s(J)*t/np.pi) )
    def Psi(J): return np.exp(-g*t/2) * (1j*s(J)-g/2) * t * np.sinc(s(J)*t/np.pi)

    Sp = N/2 * np.exp(-g*t/2) * Phi(1)**(N-1)
    Sm = np.conj(Sp)

    Sp_Sp = 1/4 * N * (N-1) * np.exp(-g*t) * Phi(2)**(N-2)
    Sp_Sz = -1/2 * Sp + 1/4 * N * (N-1) * np.exp(-g*t/2) * Psi(1) * Phi(1)**(N-2)
    Sp_Sm = N/2 + Sz + 1/4 * N * (N-1) * np.exp(-g*t) # note that Phi(0) == 1

    return Sz, Sz_Sz, Sp, Sp_Sp, Sp_Sz, Sp_Sm
