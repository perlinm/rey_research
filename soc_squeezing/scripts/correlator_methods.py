#!/usr/bin/env python3

# FILE CONTENTS: methods for computing collective spin correlators

import numpy as np
import scipy.sparse as sparse
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
def op_val_X(N, op):
    if op == (0,0,0): return 1
    l, m, n = op
    S = N/2
    return sum([ k**n
                 * np.exp( ln_binom(N,S+k)/2 + ln_binom(N,S+k+l-m)/2 - N*np.log(2) )
                 * np.prod([ transverse_elem(1,S,k-m+p) for p in range(l) ])
                 * np.prod([ transverse_elem(-1,S,k-p) for p in range(m) ])
                 for k in np.arange(-S+m,S-l) ])

# correlator < -Z | S_+^l S_-^m S_z^n | -Z >
def op_val_nZ(N, op):
    if op == (0,0,0): return 1
    l, m, n = op
    S = N/2
    z_fac = (-S)**n
    t_fac = sum([ np.exp( ln_binom(N,S+k)/2 + ln_binom(N,S+k+l-m)/2 - N*np.log(2) )
                  * np.prod([ transverse_elem(1,S,k-m+p) for p in range(l) ])
                  * np.prod([ transverse_elem(-1,S,k-p) for p in range(m) ])
                  for k in np.arange(-S+m,S-l) ])
    return z_fac * t_fac

# collective spin operator commutator coefficient (see notes)
def epsilon(M,N,P,L):
    return 2**L * np.sum([ (-1)**(P-qq) * stirling(P,qq) * binom(qq,L) * (M-N)**(qq-L)
                           for qq in range(L,P+1) ])

# compute pre-image of a single operator from coherent evolution
# operators in (L,M,N) format, and pre-image in dictionary format
def coherent_op_image(op, h_vals):
    if op == (0,0,0): return {}
    L, M, N = op
    image = {}
    for P,Q,A in h_vals.keys():
        term_val = h_vals[(P,Q,A)]
        for pp, qq, aa, rr, ss, bb, sign in [ (P,Q,A,L,M,N,1), (L,M,N,P,Q,A,-1) ]:
            for kk in range(min(qq,rr)+1):
                val_kk = term_val * sign * factorial(kk) * binom(qq,kk) * binom(rr,kk)
                for ll in range(kk+1):
                    val_ll = val_kk * (-1)**ll * epsilon(qq,rr,kk,ll)
                    for mm in range(ll+1):
                        val_mm = val_ll * binom(ll,mm) * (-ss)**(ll-mm)
                        for nn in range(aa+1):
                            val = val_mm * binom(aa,nn) * (rr-ss)**(aa-nn)
                            op_in = (pp+rr-kk, qq+ss-kk, bb+mm+nn)
                            try: image[op_in]
                            except: image[op_in] = 0
                            image[op_in] += val
    null_keys = [ key for key in image.keys() if image[key] == 0 ]
    for key in null_keys: del image[key]
    return image

# compute pre-image of a single operator from decoherence
# operators in (L,M,N) format, and pre-image in dictionary format
def decoherence_op_image(op, spin_number):
    if op == (0,0,0): return {}
    L, M, N = op
    val_kk = np.array([ (-1)**(N-kk) * binom(N,kk) for kk in range(N) ])
    image = { (L,M,kk) : (spin_number/2-M) * val_kk[kk] for kk in range(N) }
    image[(L,M,N)] = -1/2*(L+M)
    for kk in range(N): image[(L,M,kk+1)] += val_kk[kk]
    return image

# compute pre-image of a single operator from infinitesimal time evolution
# operators in (L,M,N) format, and pre-image in dictionary format
def op_image(op, h_vals, spin_number, decay_rate_over_chi):
    op_image_im = coherent_op_image(op, h_vals)
    op_image_re = decoherence_op_image(op, spin_number)
    op_image = { op : 1j * op_image_im[op] for op in op_image_im.keys() }
    if decay_rate_over_chi != 0:
        for key in op_image_re.keys():
            try: op_image[key]
            except: op_image[key] = 0
            op_image[key] += decay_rate_over_chi * op_image_re[key]
    return op_image

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
                        max_order = 30):
    assert(initial_state in [ "X", "-Z" ])
    if initial_state == "X":
        initial_val = op_val_X
    if initial_state == "-Z":
        initial_val = op_val_nZ

    # list of operators necessary for computing squeezing, namely:
    #                    Sz     S_z^2,     Sp    S_+ S_z  S_- S_z   S_+^2   S_+ S_-
    squeezing_ops = [ (0,0,1), (0,0,2), (1,0,0), (1,0,1), (0,1,1), (2,0,0), (1,1,0) ]

    # arguments for computing operator pre-image under infinitesimal time translation
    op_image_args = ( h_vals, N, decay_rate_over_chi )

    diff_op = {} # generator of time translations
    time_derivatives = {} # [ sqz_op ][ derivative_order ][ operator ] --> value
    relevant_ops = set(squeezing_ops) # set of operators relevant for time evolution
    for sqz_op in squeezing_ops:
        time_derivatives[sqz_op] = { 0 : { sqz_op : 1 } }
        for order in range(1,max_order+1):
            time_derivatives[sqz_op][order] \
                = compute_time_derivative(diff_op,
                                          time_derivatives[sqz_op][order-1],
                                          op_image_args)
            relevant_ops |= time_derivatives[sqz_op][order].keys()

    initial_vals = { op : initial_val(N, op) for op in relevant_ops }

    T = np.array([ chi_times**kk / factorial(kk) for kk in range(max_order+1) ])
    Q = {} # dictionary (l,m,n) --> < D_t^k S_+^l S_-^m S_z^n >_0
    vals = {}
    for sqz_op in squeezing_ops:
        Q[sqz_op] = np.array([ sum([ time_derivatives[sqz_op][order][op] * initial_vals[op]
                                     for op in time_derivatives[sqz_op][order].keys() ])
                               for order in range(max_order+1) ])
        vals[sqz_op] = Q[sqz_op] @ T

    return [ vals[sqz_op] for sqz_op in squeezing_ops ]
