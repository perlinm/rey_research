#!/usr/bin/env python3

# FILE CONTENTS: methods for computing collective spin correlators

import numpy as np
import scipy.sparse as sparse

from math import lgamma
from mpmath import hyper
from scipy.special import factorial, binom
from sympy.functions.combinatorial.numbers import stirling as sympy_stirling

# natural logarithm of factorial
def ln_factorial(n): return lgamma(n+1)

# unsigned stirling number of the first kind
def stirling(n,k): return float(sympy_stirling(n, k, kind = 1, signed = True))

# correlator < +X | S_\mu^ll S_z^mm S_\bmu^nn | +X >
def op_val_pX(spin_num, op, mu):
    ll, mm, nn = op
    if ll == 0 and nn == 0 and mm % 2 == 1: return 0

    total_spin = spin_num/2
    if mu == 1:
        k_vals = np.arange(-total_spin,total_spin-max(ll,nn)+1)
    else: # mu == -1
        k_vals = np.arange(-total_spin+max(ll,nn),total_spin+1)
    if k_vals.size == 0: return 0

    ln_prefactor = ln_factorial(spin_num) - spin_num*np.log(2)
    def ln_factors(kk,ll,nn):
        ln_numerator = ln_factorial(total_spin - mu*kk)
        ln_denominator = ( ln_factorial(total_spin + mu*kk)
                           + ln_factorial(total_spin - mu*kk - ll)
                           + ln_factorial(total_spin - mu*kk - nn) )
        return ln_numerator - ln_denominator
    ln_factors = np.vectorize(ln_factors)

    return np.sum(k_vals**mm * np.exp(ln_factors(k_vals,ll,nn) + ln_prefactor))

# correlator < -Z | S_\mu^ll S_z^mm S_\bmu^nn | -Z >
def op_val_nZ(spin_num, op, mu = None):
    ll, mm, nn = op
    if ll != 0 or nn != 0: return 0
    return (-spin_num/2)**mm

# collective spin operator commutator coefficients (see notes)
def epsilon(mm,nn,pp,ll):
    return 2**ll * sum([ stirling(pp,qq) * binom(qq,ll) * (mm-nn)**(qq-ll)
                         for qq in range(ll,pp+1) ])
def xi(mm,nn,pp,qq):
    return sum([ (-1)**(ll-qq) * epsilon(mm,nn,pp,ll) * binom(ll,qq) * (mm-pp)**(ll-qq)
                 for ll in range(qq,pp+1) ])

# compute pre-image of a single operator from coherent evolution
def op_image_coherent(op, h_vals, mu):
    if op == (0,0,0): return {}
    LL, MM, NN = op
    image = {}
    for PP,QQ,RR in h_vals.keys():
        h_term = h_vals[(PP,QQ,RR)]
        for pp, aa, qq, rr, bb, ss, sign in [ ( PP, QQ, RR, LL, MM, NN, +1),
                                              ( LL, MM, NN, PP, QQ, RR, -1) ]:
            for kk in range(min(qq,rr)+1):
                kk_fac = factorial(kk) * binom(qq,kk) * binom(rr,kk)
                for ll in range(kk+1):
                    kl_fac = kk_fac * (-1)**ll * xi(qq,rr,kk,ll)
                    for mm in range(aa+1):
                        klm_fac = kl_fac * (rr-kk)**(aa-mm) * binom(aa,mm)
                        for nn in range(bb+1):
                            klmn_fac = klm_fac * (qq-kk)**(bb-nn) * binom(bb,nn)
                            klmn_sign = mu**(aa+bb+ll-mm-nn)
                            val = 1j * sign * h_term * klmn_fac * klmn_sign
                            op_in = (pp+rr-kk, ll+mm+nn, qq+ss-kk)
                            try: image[op_in]
                            except: image[op_in] = 0
                            image[op_in] += val
    return image

# multiply dictionary vector by a scalar
def dict_mult(scalar,vec):
    return { key : scalar * vec[key] for key in vec.keys() }

# add two vectors in dictionary format
def add_dicts(*dicts):
    comb = dicts[0]
    for dict_jj in dicts[1:]:
        for key in dict_jj:
            try: comb[key]
            except: comb[key] = 0
            comb[key] += dict_jj[key]
    return comb

# compute pre-image of a single operator from single-spin excitation / decay
# via \D_\bmu ( S_\mu^ll S_z^mm S_\bmu^nn )
def op_image_decoherence_single_diff(op, spin_num, mu):
    ll, mm, nn = op
    val_kk = np.array([ (-mu)**(mm-kk) * binom(mm,kk) for kk in range(mm) ])
    image_0 = { (ll,kk,nn) : spin_num/2 * val_kk[kk] for kk in range(mm) }
    image_1 = { (ll,kk+1,nn) : mu * val_kk[kk] for kk in range(mm) }
    image_m = { (ll,mm,nn) : -1/2*(ll+nn) }
    return add_dicts(image_0, image_1, image_m)

# compute pre-image of a single operator from single-spin excitation / decay
# via \D_\mu ( S_\mu^l S_z^m S_\bmu^n )
def op_image_decoherence_single_same(op, spin_num, mu):
    ll, mm, nn = op

    val_kk = np.array([ mu**(mm-kk) * binom(mm,kk) for kk in range(mm) ])
    image_0_0 = { (ll,kk,nn) : (spin_num/2-ll-nn) * val_kk[kk] for kk in range(mm) }
    image_0_1 = { (ll,kk+1,nn) : -mu * val_kk[kk] for kk in range(mm) }
    image_0_m = { (ll,mm,nn) : -1/2*(ll+nn) }

    image = add_dicts(image_0_0, image_0_1, image_0_m)

    if ll >= 1 and nn >= 1:
        image_1 = { (ll-1,mm,nn-1) : ll*nn * (spin_num-ll-nn+2) }
        image = add_dicts(image, image_1)

    if ll >= 2 and nn >= 2:
        ln_fac = ll*nn*(ll-1)*(nn-1)
        val_kk = np.array([ (-mu)**(mm-kk) * binom(mm,kk) for kk in range(mm+1) ])
        image_2_0 = { (ll-2,kk,nn-2) : ln_fac * spin_num/2 * val_kk[kk]
                      for kk in range(mm+1) }
        image_2_1 = { (ll-2,kk+1,nn-2) : ln_fac * mu * val_kk[kk]
                      for kk in range(mm+1) }
        image = add_dicts(image, image_2_0, image_2_1)

    return image

# compute pre-image of a single operator from single-spin decay
# via \D_\nu ( S_\mu^l S_z^m S_\bmu^n )
def decay_op_image(op, spin_num, mu, nu):
    if mu == nu:
        return op_image_decoherence_single_same(op, spin_num, mu)
    else:
        return op_image_decoherence_single_diff(op, spin_num, mu)

# compute pre-image of a single operator from infinitesimal time evolution
# operators in (ll,mm,nn) format, and pre-image in dictionary format
def op_image(op, h_vals, spin_num, dec_rates, mu):
    image = op_image_coherent(op, h_vals, mu)

    g_z, g_p, g_m = dec_rates
    for g_nu, nu in [ (g_p,+1),
                      (g_m,-1) ]:
        if g_nu != 0:
            image_dec = decay_op_image(op, spin_num, mu, nu)
            image = add_dicts(image, dict_mult(g_nu, image_dec))

    null_keys = [ key for key in image.keys() if abs(image[key]) == 0 ]
    for key in null_keys: del image[key]

    return image

# take hermitian conjugate of a dictionary taking operator --> value,
#   i.e. return a dictionary taking operator* --> value*
def conjugate_op_vec(op_vec):
    return { op[::-1] : np.conj(op_vec[op]) for op in op_vec.keys() }

# compute time derivative of a given vector of spin operators
def compute_time_derivative(diff_op, input_vector, op_image_args):
    output_vector = {}
    # for each operator in the input vector
    for input_op in input_vector.keys():
        op_coefficient = input_vector[input_op]
        # if we do not know the time derivative of this operator, compute it
        try: diff_op[input_op]
        except:
            diff_op[input_op] = op_image(input_op, *op_image_args)
            # we get the time derivative of the conjugate operator for free, so save it
            if input_op[0] != input_op[-1]:
                diff_op[input_op[::-1]] = conjugate_op_vec(diff_op[input_op])
        for output_op in diff_op[input_op]:
            try: output_vector[output_op]
            except: output_vector[output_op] = 0
            output_vector[output_op] += op_coefficient * diff_op[input_op][output_op]
    return output_vector

# return correlators from evolution under a general Hamiltonian
def compute_correlators(spin_num, chi_times, h_vals, dec_rates, initial_state,
                        order_cap = 30, init_vals_pX = {}, init_vals_nZ = {}, mu = 1):
    assert(mu in [+1,-1])
    assert(initial_state in [ "+X", "-Z" ])
    if initial_state == "+X":
        initial_val = op_val_pX
        initial_vals = init_vals_pX
    if initial_state == "-Z":
        initial_val = op_val_nZ
        initial_vals = init_vals_nZ

    # list of operators necessary for computing squeezing, namely:
    #                    Sz     S_z^2,   S_\mu   S_\mu^2  S_\mu S_z  S_\mu S_\bmu
    squeezing_ops = [ (0,1,0), (0,2,0), (1,0,0), (2,0,0),  (1,1,0),    (1,0,1) ]

    # arguments for computing operator pre-image under infinitesimal time translation
    op_image_args = ( h_vals, spin_num, dec_rates, mu )

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
    for sqz_op in squeezing_ops:
        for order in range(order_cap):
            for op in time_derivatives[sqz_op][order].keys():
                try: initial_vals[op]
                except:
                    initial_vals[op] = initial_val(spin_num, op, mu)
                    # all our initial values are real, so no need to conjugate
                    if op[0] != op[-1]:
                        initial_vals[op[::-1]] = initial_vals[op]

    T = np.array([ chi_times**kk / factorial(kk) for kk in range(order_cap) ])
    Q = {} # dictionary (ll,mm,nn) --> < D_t^kk S_\mu^ll S_z^mm S_\bmu^nn >_0 for all kk
    correlators = {}
    for sqz_op in squeezing_ops:
        Q[sqz_op] = np.array([ sum([ time_derivatives[sqz_op][order][op] * initial_vals[op]
                                     for op in time_derivatives[sqz_op][order].keys() ])
                               for order in range(order_cap) ])
        correlators[sqz_op] = Q[sqz_op] @ T

    if mu == 1:
        return correlators

    else: # mu == -1
        reversed_corrs = {}
        reversed_corrs[(0,1,0)] = correlators[(0,1,0)]
        reversed_corrs[(0,2,0)] = correlators[(0,2,0)]
        reversed_corrs[(1,0,0)] = np.conj(correlators[(1,0,0)])
        reversed_corrs[(2,0,0)] = np.conj(correlators[(2,0,0)])
        reversed_corrs[(1,1,0)] = np.conj(correlators[(1,1,0)] - correlators[(1,0,0)])
        reversed_corrs[(1,0,1)] = np.conj(correlators[(1,0,1)]) + 2 * correlators[(0,1,0)]

        return reversed_corrs

# exact correlators for OAT with decoherence
# derivations in foss-feig2013nonequilibrium
def correlators_OAT(spin_num, chi_times, dec_rates):
    N = spin_num
    t = chi_times
    g_z, g_p, g_m = dec_rates
    g = g_m

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

    return { (0,1,0) : Sz,
             (0,2,0) : Sz_Sz,
             (1,0,0) : Sp,
             (2,0,0) : Sp_Sp,
             (1,1,0) : Sp_Sz,
             (1,0,1) : Sp_Sm }
