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

# compute pre-image of a single operator from coherent evolution
# operators in (L,M,N) format, and pre-image in dictionary format
def coherent_op_image(op, h_vals):
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
                            if op_in not in image.keys():
                                image[op_in] = val
                            else:
                                image[op_in] += val
    null_keys = [ key for key in image.keys() if image[key] == 0 ]
    for key in null_keys: del image[key]
    return image

# super-operator for coherent evolution of spin operators
def coherent_evolution_operator(vec_terms, idx, h_vals):
    vec_dim = len(vec_terms)
    op_mat = sparse.dok_matrix((vec_dim,vec_dim))
    for op_out in vec_terms:
        idx_out = idx(*op_out)
        image = coherent_op_image(op_out, h_vals)
        for op_in in image.keys():
            idx_in = idx(*op_in)
            if idx_in < vec_dim:
                op_mat[idx_out,idx_in] += image[op_in]
    op_mat = op_mat.tocsr()
    op_mat.eliminate_zeros()
    return op_mat

# super-operator for decoherence of spin operators
def decoherence_evolution_operator(vec_terms, idx, spin_num):
    vec_dim = len(vec_terms)
    op_mat = sparse.dok_matrix((vec_dim,vec_dim))
    for l, m, n in vec_terms:
        idx_out = idx(l,m,n)
        for kk in range(n):
            val = (-1)**(n-kk) * binom(n,kk)
            op_mat[idx_out,idx(l,m,kk)] += (spin_num/2-m) * val
            if kk+1 < vec_dim:
                op_mat[idx_out,idx(l,m,kk+1)] += val
        op_mat[idx_out,idx_out] += -1/2 * (l+m)
    return op_mat.tocsr()

# net super-operator for evolution of spin operators
def evolution_operator(vec_terms, idx, N, h_vals, decay_rate_over_chi):
    op_mat_im = coherent_evolution_operator(vec_terms, idx, h_vals)
    op_mat_re = decoherence_evolution_operator(vec_terms, idx, N)
    return 1j * op_mat_im + decay_rate_over_chi * op_mat_re

# return correlators from evolution under a general Hamiltonian
def compute_correlators(N, chi_times, decay_rate_over_chi, h_vals, initial_state,
                        max_order = 30, print_updates = False):
    assert(initial_state in [ "X", "-Z" ])
    if initial_state == "X":
        initial_val = op_val_X
    if initial_state == "-Z":
        initial_val = op_val_nZ

    # list of "seed" squeezing operators: S_z^2, S_+ S_z, S_- S_z, S_+^2, and S_+ S_-
    squeezing_ops = [ (0,0,2), (1,0,1), (0,1,1), (2,0,0), (1,1,0) ]

    # pre-images of operators under infinitesimal time evolution
    images = { op : set([op]) for op in squeezing_ops }

    new_img_ops = { op : set([op]) for op in squeezing_ops }
    p_img_ops = {}
    for p in range(max_order):
        for sqz_op in images.keys():
            p_img_ops[sqz_op] = set()
            for new_img_op in new_img_ops[sqz_op]:
                p_img_ops[sqz_op] |= coherent_op_image(new_img_op, h_vals).keys()
            new_img_ops[sqz_op] = set([ op for op in p_img_ops[sqz_op]
                                        if op not in images[sqz_op] ])
            images[sqz_op] |= new_img_ops[sqz_op]

    # add pre-images from decoherence
    # WARNING: assumes e --> g decay is the only source of decoherence
    for sqz_op in images.keys():
        for img_op in list(images[sqz_op]):
            images[sqz_op] |= set([ (img_op[0],img_op[1],kk) for kk in range(img_op[2]) ])

    transverse_images = { sqz_op: set([ (op[0],op[1]) for op in images[sqz_op] ])
                          for sqz_op in images.keys() }

    # combine intersecting pre-images
    for op_B in squeezing_ops[::-1]:
        for op_A in squeezing_ops:
            if op_A == op_B: break
            if len( transverse_images[op_A] & transverse_images[op_B] ) != 0:
                images[op_A] |= images[op_B]
                del images[op_B]
                break

    seed_ops, diff_ops, op_vecs = {}, {}, {}
    for seed in images.keys():
        # determine which operators are associated with this seed
        seed_ops[seed] = sorted(images[seed], key = lambda x: (x[0]+x[1]+x[2],x[0],x[1]))

        # initialize vector of operators for this seed
        op_vecs[seed] = np.zeros((len(chi_times),len(seed_ops[seed])), dtype = complex)

        # construct dictionary taking operator --> index
        idx_dict = {}
        for ii in range(len(seed_ops[seed])):
            idx_dict[seed_ops[seed][ii]] = ii
            # set initial values in operator vector
            op_vecs[seed][0,ii] = initial_val(N, *seed_ops[seed][ii])

        # construct function taking operator --> index
        number_of_ops = len(seed_ops[seed])
        def idx(l,m,n):
            try:
                return idx_dict[(l,m,n)]
            except:
                return number_of_ops

        # construct generator of time translation for this vector of operators
        diff_ops[seed] \
            = evolution_operator(seed_ops[seed], idx, N, h_vals, decay_rate_over_chi)

    # simulate!
    for ii in range(1,len(chi_times)):
        if print_updates:
            print(f"{ii}/{len(chi_times)}")
        dt = chi_times[ii] - chi_times[ii-1]
        for seed in images.keys():
            op_vecs[seed][ii] \
                = sparse.linalg.expm_multiply(dt * diff_ops[seed], op_vecs[seed][ii-1])

    # extract the correlators we need from the simulations
    for seed in images.keys():
        ops = seed_ops[seed]
        for ii in range(len(ops)):
            if sum(ops[ii]) > 2: break
            if ops[ii] == (0,0,1): Sz    = op_vecs[seed][:,ii]
            if ops[ii] == (0,0,2): Sz_Sz = op_vecs[seed][:,ii]
            if ops[ii] == (1,0,0): Sp    = op_vecs[seed][:,ii]
            if ops[ii] == (1,0,1): Sp_Sz = op_vecs[seed][:,ii]
            if ops[ii] == (0,1,1): Sm_Sz = op_vecs[seed][:,ii]
            if ops[ii] == (2,0,0): Sp_Sp = op_vecs[seed][:,ii]
            if ops[ii] == (1,1,0): Sp_Sm = op_vecs[seed][:,ii]

    return Sz, Sz_Sz, Sp, Sp_Sz, Sm_Sz, Sp_Sp, Sp_Sm
