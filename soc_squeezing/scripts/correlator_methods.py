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
def op_val_pX(total_spin, op, mu):
    ll, mm, nn = op
    if ll == 0 and nn == 0 and mm % 2 == 1: return 0

    if mu == 1:
        k_vals = np.arange(-total_spin,total_spin-max(ll,nn)+1)
    else: # mu == -1
        k_vals = np.arange(-total_spin+max(ll,nn),total_spin+1)
    if k_vals.size == 0: return 0

    ln_prefactor = ln_factorial(2*total_spin) - 2*total_spin*np.log(2)
    def ln_factors(kk,ll,nn):
        ln_numerator = ln_factorial(total_spin - mu*kk)
        ln_denominator = ( ln_factorial(total_spin + mu*kk)
                           + ln_factorial(total_spin - mu*kk - ll)
                           + ln_factorial(total_spin - mu*kk - nn) )
        return ln_numerator - ln_denominator
    ln_factors = np.vectorize(ln_factors)

    return np.sum(k_vals**mm * np.exp(ln_factors(k_vals,ll,nn) + ln_prefactor))

# correlator < -Z | S_\mu^ll S_z^mm S_\bmu^nn | -Z >
def op_val_nZ(total_spin, op, mu = None):
    ll, mm, nn = op
    if ll != 0 or nn != 0: return 0
    return (-total_spin)**mm

# collective spin operator commutator coefficients (see notes)
def epsilon(mm,nn,pp,ll):
    return 2**ll * sum([ stirling(pp,qq) * binom(qq,ll) * (mm-nn)**(qq-ll)
                         for qq in range(ll,pp+1) ])
def xi(mm,nn,pp,qq):
    return sum([ (-1)**(ll-qq) * epsilon(mm,nn,pp,ll) * binom(ll,qq) * (mm-pp)**(ll-qq)
                 for ll in range(qq,pp+1) ])

# multiply dictionary vector by a scalar
def dict_mult(scalar,vec):
    return { key : scalar * vec[key] for key in vec.keys() }

# add dictionary vectors
def add_dicts(*dicts):
    comb = dicts[0]
    for dict_jj in dicts[1:]:
        for key in dict_jj:
            try: comb[key]
            except: comb[key] = 0
            comb[key] += dict_jj[key]
    return comb

# simplify product S_\mu^pp S_z^aa S_\bmu^qq S_\mu^rr S_z^bb S_\bmu^ss
def simplify_term(pp, aa, qq, rr, bb, ss, mu, prefactor = 1):
    term = {}
    for kk in range(min(qq,rr)+1):
        kk_fac = factorial(kk) * binom(qq,kk) * binom(rr,kk)
        for ll in range(kk+1):
            kl_fac = kk_fac * (-1)**ll * xi(qq,rr,kk,ll)
            for mm in range(aa+1):
                klm_fac = kl_fac * (rr-kk)**(aa-mm) * binom(aa,mm)
                for nn in range(bb+1):
                    klmn_fac = klm_fac * (qq-kk)**(bb-nn) * binom(bb,nn)
                    klmn_sign = mu**(aa+bb+ll-mm-nn)
                    op_in = (pp+rr-kk, ll+mm+nn, qq+ss-kk)
                    try: term[op_in]
                    except: term[op_in] = 0
                    term[op_in] += klmn_sign * klmn_fac * prefactor
    return term

# compute pre-image of a single operator from coherent evolution
def op_image_coherent(op, h_vals, mu):
    if op == (0,0,0): return {}
    LL, MM, NN = op
    image = {}
    for PP,QQ,RR in h_vals.keys():
        prefactor = 1j * h_vals[(PP,QQ,RR)]
        for pp, aa, qq, rr, bb, ss, sign in [ ( PP, QQ, RR, LL, MM, NN, +1),
                                              ( LL, MM, NN, PP, QQ, RR, -1) ]:
            image = add_dicts(image,
                              simplify_term(pp, aa, qq, rr, bb, ss, mu, sign * prefactor))
    return image

# return "binomial expansion" vector
def binom_vec(mm, sign):
    return np.array([ sign**(mm-kk) * binom(mm,kk) for kk in range(mm+1) ])

# return vector S_\mu^ll (x + S_z)^mm * S_\bmu^nn
def binom_op(ll, mm, nn, x, prefactor = 1):
    return { (ll,kk,nn) : prefactor * x**(mm-kk) * binom(mm,kk) for kk in range(mm+1) }

# takes S_\mu^ll S_z^mm + S_\bmu^nn --> S_\mu^ll ( \sum_jj x_jj S_z^jj ) S_z^mm + S_\bmu^nn
def insert_z_terms(vec, coefficients):
    output = dict_mult(coefficients[0],vec)
    for jj in range(1,len(coefficients)):
        for ll, mm, nn in vec.keys():
            try: output[(ll,mm+jj,nn)]
            except: output[(ll,mm+jj,nn)] = 0
            output[(ll,mm+jj,nn)] += coefficients[jj] * vec[(ll,mm,nn)]
    return output

# shorthand for operator term: "extended binomial operator"
def ext_binom_op(ll, mm, nn, x, terms, prefactor = 1):
    return insert_z_terms(binom_op(ll,mm,nn,x,prefactor), terms)

# decoherence image \D_z ( S_\mu^l S_z^m S_\bmu^n )
def op_image_single_dephasing(op, S, mu):
    ll, mm, nn = op
    if ll + nn == 0: return {}
    image = { (ll,mm,nn) : -(ll+nn) }
    if ll >= 1 and nn >= 1:
        image.update(ext_binom_op(ll-1, mm, nn-1, -mu, [S, mu], 2*ll*nn))
    return image

# decoherence image \D_\nu ( S_\mu^l S_z^m S_\bmu^n )
def op_image_single_decay(op, S, mu, nu):
    ll, mm, nn = op
    if mu == nu:
        term_0_1 = ext_binom_op(ll, mm, nn, mu, [ S - ll - nn, -mu ])
        term_0_2 = insert_z_terms({op:-1}, [ S - (ll+nn)/2, -mu ])
        image = add_dicts(term_0_1, term_0_2)
        if ll >= 1 and nn >= 1:
            image[(ll-1,mm,nn-1)] = ll*nn * (2*S-ll-nn+2)
        if ll >= 2 and nn >= 2:
            term_2 = ext_binom_op(ll-2, mm, nn-2, -mu, [ S, mu ])
            image.update(dict_mult(ll*nn*(ll-1)*(nn-1), term_2))
        return image
    else:
        term_1 = ext_binom_op(ll, mm, nn, -mu, [ S, mu ])
        term_2 = insert_z_terms({op:-1}, [ S + (ll+nn)/2, mu ])
        return add_dicts(term_1, term_2)

# single-spin sandwich with z, z
def op_image_single_san_z_z(op, S, mu):
    ll, mm, nn = op
    image = { (ll,mm,nn) : 2*(S-ll-nn) }
    if ll >= 1 and nn >= 1:
        factor = 4*ll*nn
        image.update(ext_binom_op(ll-1, mm, nn-1, -mu, [ S, mu ], factor))
    return image

# single-spin sandwich with (z,nu) and its conjugate
def op_image_single_san_z_nu(op, S, mu, nu):
    ll, mm, nn = op

    if mu == nu:
        image_reg = binom_op(ll+1, mm, nn, mu, mu)
        image_dag = binom_op(ll, mm, nn+1, mu, mu)
        if nn >= 1:
            image_reg[(ll,mm,nn-1)] = mu*nn * (-2*S + 2*ll + nn - 1)
        if ll >= 1:
            image_dag[(ll-1,mm,nn)] = mu*ll * (-2*S + 2*nn + ll - 1)
        if nn >= 2 and ll >= 1:
            factor = -2*mu*ll*nn*(nn-1)
            image_reg.update(ext_binom_op(ll-1, mm, nn-2, -mu, [ S, mu ], factor))
        if ll >= 2 and nn >= 1:
            factor = -2*mu*ll*nn*(ll-1)
            image_dag.update(ext_binom_op(ll-2, mm, nn-1, -mu, [ S, mu ], factor))

    else:
        image_reg = { (ll,mm,nn+1) : -mu }
        image_dag = { (ll+1,mm,nn) : -mu }
        if ll >= 1:
            factor = 2*mu*ll
            image_reg.update(ext_binom_op(ll-1, mm, nn, -mu, [ S, mu ], factor))
        if nn >= 1:
            factor = 2*mu*nn
            image_sag.update(ext_binom_op(ll, mm, nn-1, -mu, [ S, mu ], factor))

    return image_reg, image_dag

# single-spin sandwich with nu, nu
def op_image_single_san_nu_nu(op, S, mu):
    ll, mm, nn = op
    image_same = {}
    image_diff = {}
    if nn >= 1:
        image_same[(ll+1,mm,nn-1)] = nn
    if ll >= 1:
        image_diff[(ll-1,mm,nn+1)] = ll
    if nn >= 2:
        factor = -nn*(nn-1)
        image_same.update(ext_binom_op(ll, mm, nn-2, -mu, [ S, mu ], factor))
    if ll >= 2:
        factor = -ll*(ll-1)
        image_same.update(ext_binom_op(ll-2, mm, nn, -mu, [ S, mu ], factor))
    return image_same, image_diff

# single-spin sandwich with nu, bnu
def op_image_single_san_nu_bnu(op, S, mu, nu):
    ll, mm, nn = op
    image_same = ext_binom_op(ll, mm, nn, -mu, [ S, mu ])
    image_diff = ext_binom_op(ll, mm, nn, mu, [ S-ll-nn, -mu ])
    if ll >= 1 and nn >= 1:
        image_diff[(ll-1,mm,nn-1)] = ll*nn * (2*S-ll-nn+2)
    if ll >= 2 and nn >= 2:
        factor = ll*nn*(ll-1)*(nn-1)
        image_diff.update(ext_binom_op(ll-2, mm, nn-2, -mu, [ S, mu ], factor))
    return image_same, image_diff

# compute pre-image of a single operator from infinitesimal time evolution
# operators in (ll,mm,nn) format, and pre-image in dictionary format
def op_image(op, h_vals, S, dec_rates, mu):
    ll, mm, nn = op
    image = op_image_coherent(op, h_vals, mu)

    g_z, g_p, g_m = dec_rates

    if g_z != 0 and ll + nn != 0:
        image_dec = op_image_single_dephasing(op, S, mu)
        image = add_dicts(image, dict_mult(g_z, image_dec))

    for g_nu, g_bmu, nu in [ (g_p, g_m, +1),
                             (g_m, g_p, -1) ]:
        if g_nu != 0:
            image_dec = op_image_single_decay(op, S, mu, nu)
            image = add_dicts(image, dict_mult(g_nu, image_dec))
        if g_bmu * g_nu != 0:
            image_dec = op_image_single_san_nu_nu(op, S, mu, nu)
            image = add_dicts(image, dict_mult(np.conj(g_bnu)*g_nu, image_dec))

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
    op_image_args = ( h_vals, spin_num/2, dec_rates, mu )

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
                    initial_vals[op] = initial_val(spin_num/2, op, mu)
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
    S = spin_num/2
    t = chi_times
    g_z, g_p, g_m = dec_rates

    gam = -(g_p - g_m) / 2
    lam = (g_p + g_m) / 2
    rr = g_p * g_m
    Gam = g_z + lam

    if g_m != 0 and g_p != 0:
        Sz_unit = (g_p-g_m)/(g_p+g_m) * (1-np.exp(-(g_p+g_m)*t))
    else:
        Sz_unit = np.zeros(len(chi_times))
    Sz = S * Sz_unit
    Sz_Sz = S * (1/2 + (S-1/2) * Sz_unit**2)

    def s(J): return J + 1j*gam
    def Phi(J):
        return np.exp(-lam*t) * ( np.cos(t*np.sqrt(s(J)**2-rr))
                                  + lam*t * np.sinc(t*np.sqrt(s(J)**2-rr)/np.pi) )
    def Psi(J):
        return np.exp(-lam*t) * (1j*s(J)-gam) * t * np.sinc(t*np.sqrt(s(J)**2-rr)/np.pi)

    Sp = S * np.exp(-Gam*t) * Phi(1)**(N-1)
    Sp_Sz = -1/2 * Sp + S * (S-1/2) * np.exp(-Gam*t) * Psi(1) * Phi(1)**(N-2)
    Sp_Sp = S * (S-1/2) * np.exp(-2*Gam*t) * Phi(2)**(N-2)
    Sp_Sm = S + Sz + S * (S-1/2) * np.exp(-2*Gam*t) # note that Phi(0) == 1

    return { (0,1,0) : Sz,
             (0,2,0) : Sz_Sz,
             (1,0,0) : Sp,
             (2,0,0) : Sp_Sp,
             (1,1,0) : Sp_Sz,
             (1,0,1) : Sp_Sm }
