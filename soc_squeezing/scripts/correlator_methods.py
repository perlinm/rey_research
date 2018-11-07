#!/usr/bin/env python3

# FILE CONTENTS: methods for computing collective spin correlators

import itertools, scipy
import numpy as np

from scipy.special import gamma, gammaln
from scipy.special import binom as scipy_binom
from sympy.functions.combinatorial.numbers import stirling as sympy_stirling
from scipy.integrate import solve_ivp


##########################################################################################
# common methods modified to recycle values
##########################################################################################

# factorial
def factorial(nn, vals = {}):
    try: return vals[nn]
    except: None
    vals[nn] = gamma(nn+1)
    return vals[nn]

# logarithm of factorial
def ln_factorial(nn, vals = {}):
    try: return vals[nn]
    except: None
    vals[nn] = gammaln(nn+1)
    return vals[nn]

# binomial coefficient
def binom(nn, kk, vals = {}):
    try: return vals[nn,kk]
    except: None
    vals[nn,kk] = scipy_binom(nn,kk)
    return vals[nn,kk]

# unsigned stirling number of the first kind
def stirling(nn, kk, vals = {}):
    try: return vals[nn,kk]
    except: None
    val = int(sympy_stirling(nn, kk, kind = 1, signed = False))
    vals[nn,kk] = val
    return val

# coefficient for computing a product of spin operators
def zeta(mm, nn, pp, qq, vals = {}):
    try: return vals[mm,nn,pp,qq]
    except: None
    val = (-1)**pp * 2**qq * np.sum([ stirling(pp,ss) * binom(ss,qq)
                                      * (mm+nn-2*pp)**(ss-qq)
                                      for ss in range(qq,pp+1) ])
    vals[mm,nn,pp,qq] = val
    vals[nn,mm,pp,qq] = val
    return val


##########################################################################################
# expectation values
##########################################################################################

# natural logarithm of factors which appear in the expetation value for |X>
def ln_factors_X(SS, kk, ll, nn):
    return ( ln_factorial(SS - kk)
             - ln_factorial(SS + kk)
             - ln_factorial(SS - kk - ll)
             - ln_factorial(SS - kk - nn) )
ln_factors_X = np.vectorize(ln_factors_X)

# correlator < X | S_+^ll S_\z^mm S_-^nn | X >
def op_val_X(op, SS, vals = {}):
    ll, mm, nn = op
    ll, nn = max(ll,nn), min(ll,nn)
    try: return vals[ll,mm,nn,SS]
    except: None

    if ll == 0 and nn == 0 and mm % 2 == 1: return 0
    if ll > 2*SS: return 0

    ln_prefactor = ln_factorial(2*SS) - 2*SS*np.log(2)
    k_vals = np.arange(-SS,SS-ll+0.5)
    val = ( k_vals**mm * np.exp(ln_factors_X(SS,k_vals,ll,nn) + ln_prefactor) ).sum()
    vals[ll,mm,nn,SS] = val
    return val

# correlator < Z | S_+^ll S_\z^mm S_-^nn | Z >
def op_val_Z_p(op, SS, vals = {}):
    try: return vals[ll,mm,SS]
    except: None

    ll, mm, nn = op
    if ll != nn: return 0
    spin_num = int(round(2*SS))
    if nn > spin_num: return 0

    ln_factorials_num = ln_factorial(spin_num) + ln_factorial(nn)
    ln_factorials_den = ln_factorial(spin_num-nn)
    val = (SS-nn)**mm * np.exp(ln_factorials_num - ln_factorials_den)
    vals[ll,mm,SS] = val
    return val

# correlator < Z | S_-^ll S_\z^mm S_+^nn | Z >
def op_val_Z_m(op, SS):
    ll, mm, nn = op
    if ll != 0 or nn != 0: return 0
    return (-SS)**mm

# correlator ln | < X | S_+^ll S_\z^mm S_-^nn | X > |
def op_ln_val_X(op, SS, vals = {}):
    ll, mm, nn = op
    ll, nn = max(ll,nn), min(ll,nn)
    try: return vals[ll,mm,nn,SS]
    except: None

    if ll == 0 and nn == 0 and mm % 2 == 1: return None
    if ll > 2*SS: return None

    ln_prefactor = ln_factorial(2*SS) - 2*SS*np.log(2)
    ln_factors = lambda kk : ln_factors_X(SS,kk,ll,nn)

    k_vals = np.arange(-SS,SS-ll+0.5)

    # remove kk == 0 if necessary
    if mm > 0 and SS % 1 == 0 and k_vals[-1] >= 0:
        k_vals = np.delete(k_vals,int(SS+1/4))

    # compute the logarithm of the magnitude of each term
    if mm == 0:
        ln_terms = ln_factors(k_vals) + ln_prefactor
    else:
        ln_terms = mm * np.log(np.abs(k_vals)) + ln_factors(k_vals) + ln_prefactor

    # compute the absolute value of terms divided by the largest term
    ln_term_max = ln_terms.max()
    terms = np.exp(ln_terms-ln_term_max)

    # compute the logarithm of the sum of the terms (up to an overall sign)
    if mm % 2 == 1:
        # we need te account for the sign of kk in each term
        val = ln_term_max + np.log(np.abs(np.sum(np.sign(k_vals)*terms)))
    else:
        # the sign of the kk does not matter, as it is raised to an even power
        val = ln_term_max + np.log(abs(np.sum(terms)))

    vals[ll,mm,nn,SS] = val
    return val

# correlator ln | < Z | S_+^ll S_\z^mm S_-^nn | Z > |
def op_ln_val_Z_p(op, SS, vals = {}):
    try: return vals[ll,mm,SS]
    except: None

    ll, mm, nn = op
    if ll != nn: return None
    spin_num = int(round(2*SS))
    if ll > spin_num: return None

    ln_factorials_num = ln_factorial(spin_num) + ln_factorial(ll)
    ln_factorials_den = ln_factorial(spin_num-ll)
    val = mm*np.log(SS-ll) + ln_factorials_num - ln_factorials_den
    vals[ll,mm,SS] = val
    return val

# correlator ln | < Z | S_-^ll S_\z^mm S_+^nn | Z > |
def op_ln_val_Z_m(op, SS):
    ll, mm, nn = op
    if ll != 0 or nn != 0: return None
    return mm*np.log(SS)


##########################################################################################
# machinery for manipulating operator vectors
##########################################################################################

# clean up a dictionary vector
def clean(vec, spin_num = None):
    null_ops = [ op for op, val in vec.items() if abs(val) == 0 ]
    for op in null_ops: del vec[op]
    if spin_num != None:
        overflow_ops = [ op for op in vec.keys()
                         if op[0] > spin_num or op[2] > spin_num ]
        for op in overflow_ops: del vec[op]
    return vec

# take hermitian conjugate of a dictionary taking operator --> value,
#   i.e. return a dictionary taking operator* --> value*
def conj_vec(vec):
    return { op[::-1] : np.conj(val) for op, val in vec.items() }

# add the right vector to the left vector
def add_left(vec_left, vec_right, scalar = 1):
    for op, val in vec_right.items():
        try:
            vec_left[op] += scalar * val
        except:
            vec_left[op] = scalar * val

# return sum of all input vectors
def sum_vecs(*vecs):
    vec_sum = {}
    for vec in vecs:
        if vec == {}: continue
        add_left(vec_sum, vec)
    return vec_sum

# return vector S_\mu^ll (x + \mu S_\z)^mm * S_\nu^nn
def binom_op(ll, mm, nn, x, prefactor = 1):
    return { (ll,kk,nn) : prefactor * x**(mm-kk) * binom(mm,kk) for kk in range(mm+1) }

# takes S_\mu^ll (\mu S_\z)^mm S_\nu^nn
#   --> S_\mu^ll [ \sum_jj x_jj (\mu S_\z)^jj ] (\mu S_\z)^mm + S_\nu^nn
def insert_z_poly(vec, coefficients, prefactor = 1):
    output = { op : prefactor * coefficients[0] * val for op, val in vec.items() }
    for jj in range(1,len(coefficients)):
        for op, val in vec.items():
            ll, mm, nn = op
            try:
                output[(ll,mm+jj,nn)] += prefactor * coefficients[jj] * val
            except:
                output[(ll,mm+jj,nn)] = prefactor * coefficients[jj] * val
    return output

# shorthand for operator term: "extended binomial operator"
def ext_binom_op(ll, mm, nn, terms, x, prefactor = 1):
    return insert_z_poly(binom_op(ll,mm,nn,x), terms, prefactor)


##########################################################################################
# general commutator between ordered products of collective spin operators
##########################################################################################

# simplify product of two operators
def multiply_terms(op_left, op_right):
    pp, qq, rr = op_left
    ll, mm, nn = op_right
    vec = {}
    binom_qq = [ binom(qq,bb) for bb in range(qq+1) ]
    binom_mm = [ binom(mm,cc) for cc in range(mm+1) ]
    for kk in range(min(rr,ll)+1):
        kk_fac = factorial(kk) * binom(rr,kk) * binom(ll,kk)
        for aa, bb, cc in itertools.product(range(kk+1),range(qq+1),range(mm+1)):
            bb_fac = (ll-kk)**(qq-bb) * binom_qq[bb]
            cc_fac = (rr-kk)**(mm-cc) * binom_mm[cc]
            kabc_fac = kk_fac * zeta(rr,ll,kk,aa) * bb_fac * cc_fac
            op_in = (pp+ll-kk, aa+bb+cc, rr+nn-kk)
            try:
                vec[op_in] += kabc_fac
            except:
                vec[op_in] = kabc_fac
    return clean(vec)

# simplify product of two vectors
def multiply_vecs(vec_left, vec_right, prefactor = 1):
    vec = {}
    for term_left, val_left in vec_left.items():
        for term_right, val_right in vec_right.items():
            fac = val_left * val_right * prefactor
            add_left(vec, multiply_terms(term_left, term_right), fac)
    return vec


##########################################################################################
# miscellaneous methods for changing frames and operator vectors
##########################################################################################

# decoherence transformation matrix from a periodic drive; here A = J_0(\beta), where:
#   J_0 is the zero-order bessel function of the first kind
#   \beta is the modulation index
def dec_mat_drive(A):
    const = np.array([[ 1, 0, 1 ],
                      [ 0, 0, 0 ],
                      [ 1, 0, 1 ]]) * 1/2
    var = np.array([[  1, 0, -1 ],
                    [  0, 2,  0 ],
                    [ -1, 0,  1 ]]) * 1/2
    return const + A * var

# convert vector from (z,x,y) format to (mu,z,bmu) format
def convert_zxy(vec_zxy, mu = 1):
    vec = {}
    # define vectors for (2 * Sx) and (-i * 2 * Sy)
    Sx_2 = { (1,0,0) : 1,
             (0,0,1) : 1 }
    Sy_ni2 = { (1,0,0) : -1,
               (0,0,1) :  1 }
    for op_zxy, val_zxy in vec_zxy.items():
        ll, mm, nn = op_zxy
        lmn_fac = val_zxy * 1j**nn * mu**(ll+nn) / 2**(mm+nn)
        # starting from the left, successively multiply all factors on the right
        lmn_vec = { (0,ll,0) : 1 }
        for jj in range(mm): lmn_vec = multiply_vecs(lmn_vec, Sx_2)
        for kk in range(nn): lmn_vec = multiply_vecs(lmn_vec, Sy_ni2)
        add_left(vec, lmn_vec, lmn_fac)
    if np.array([ np.imag(val) == 0 for val in vec.values() ]).all():
        vec = { op : np.real(val) for op, val in vec.items() }
    return clean(vec)


##########################################################################################
# single-spin decoherence
##########################################################################################

# diagonal terms of single-spin decoherence
def op_image_decoherence_diag_individual(op, SS, dec_vec, mu):
    ll, mm, nn = op
    D_p, D_z, D_m = abs(np.array(dec_vec))**2
    if mu == 1:
        D_mu, D_nu = D_p, D_m
    else:
        D_mu, D_nu = D_m, D_p

    image_mu = {}
    if D_mu != 0:
        image_mu = ext_binom_op(*op, [ SS-ll-nn, -1 ], 1, D_mu)
        add_left(image_mu, insert_z_poly({op:1}, [ SS-(ll+nn)/2, -1 ]), -D_mu)
        if ll >= 1 and nn >= 1:
            image_mu[(ll-1, mm, nn-1)] = ll*nn * (2*SS-ll-nn+2) * D_mu
        if ll >= 2 and nn >= 2:
            op_2 = (ll-2, mm, nn-2)
            factor = ll*nn*(ll-1)*(nn-1)
            image_mu.update(ext_binom_op(*op_2, [ SS, 1 ], -1, factor * D_mu))

    image_nu = {}
    if D_nu != 0:
        image_nu = ext_binom_op(*op, [ SS, 1 ], -1, D_nu)
        add_left(image_nu, insert_z_poly({op:1}, [ SS+(ll+nn)/2, 1 ]), -D_nu)

    image_z = {}
    if D_z != 0 and ll + nn != 0:
        image_z = { (ll,mm,nn) : -2*(ll+nn) * D_z }
        if ll >= 1 and nn >= 1:
            image_z.update(ext_binom_op(ll-1, mm, nn-1, [ SS, 1 ], -1, 4*ll*nn * D_z))

    return sum_vecs(image_mu, image_nu, image_z)

# single-spin decoherence "Q" cross term
def op_image_decoherence_Q_individual(op, SS, dec_vec, mu):
    ll, mm, nn = op
    g_p, g_z, g_m = dec_vec
    if mu == 1:
        g_mu, g_nu = g_p, g_m
    else:
        g_mu, g_nu = g_m, g_p

    gg_mp = np.conj(g_nu) * g_mu
    gg_zp = np.conj(g_z) * g_mu
    gg_mz = np.conj(g_nu) * g_z

    image_P = {}
    if gg_mp != 0 and nn != 0:
        if nn >= 2:
            image_P = ext_binom_op(ll, mm, nn-2, [ SS, 1 ], -1, -nn*(nn-1) * gg_mp)
        image_P.update({ (ll+1,mm,nn-1) : nn * gg_mp })

    image_K = {}
    if gg_zp + gg_mz != 0:
        image_K = binom_op(ll+1, mm, nn, 1, mu/2 * (gg_zp + gg_mz))
        del image_K[(ll+1,mm,nn)]

    image_L = {}
    if gg_zp != 0 and nn != 0:
        if nn >= 2 and ll >= 1:
            factor = -2*mu*ll*nn*(nn-1)
            image_L = ext_binom_op(ll-1, mm, nn-2, [ SS, 1 ], -1, factor * gg_zp)
        coefficients = [ -2*SS+2*ll+3/2*(nn-1), 1 ]
        image_L.update(insert_z_poly({(ll,mm,nn-1):1}, coefficients, mu*nn * gg_zp))

    image_M = {}
    if gg_mz != 0 and nn != 0:
        image_M = ext_binom_op(ll, mm, nn-1, [ SS, 1 ], -1, 2*mu*nn * gg_mz)
        coefficients = [ (nn-1)/2, 1 ]
        add_left(image_M, insert_z_poly({(ll,mm,nn-1):1}, coefficients, -mu*nn * gg_mz))

    return sum_vecs(image_P, image_K, image_L, image_M)


##########################################################################################
# collective-spin decoherence
##########################################################################################

# diagonal terms of collective-spin decoherence
def op_image_decoherence_diag_collective(op, SS, dec_vec, mu):
    ll, mm, nn = op
    D_p, D_z, D_m = abs(np.array(dec_vec))**2
    if mu == 1:
        D_mu, D_nu = D_p, D_m
    else:
        D_mu, D_nu = D_m, D_p

    image_mu = {}
    if D_mu != 0:
        image_mu = { (ll+1,kk,nn+1) : D_mu * (2**(mm-kk)-1) * binom(mm,kk)
                     for kk in range(mm) }
        coefficients = [ ll*(ll+1) + nn*(nn+1), 2*(ll+nn+1) ]
        image_mu.update(ext_binom_op(*op, coefficients, 1, -D_mu))
        coefficients = [ ll*(ll+1) + nn*(nn+1), 2*(ll+nn+2) ]
        add_left(image_mu, insert_z_poly({op:1}, coefficients, D_mu/2))
        if ll >= 1 and nn >= 1:
            vec = { (ll-1,mm,nn-1) : 1 }
            coefficients = [ (ll-1)*(nn-1), 2*(ll+nn-2), 4 ]
            image_mu.update(insert_z_poly(vec, coefficients, ll*nn * D_mu))

    image_nu = {}
    if D_nu != 0:
        image_nu = binom_op(ll+1, mm, nn+1, 1, -D_nu)
        del image_nu[(ll+1,mm,nn+1)]
        coefficients = [ ll*(ll-1) + nn*(nn-1), 2*(ll+nn) ]
        image_nu.update(insert_z_poly({op:1}, coefficients, D_nu/2))

    image_z = {}
    if D_z != 0 and ll != nn:
        image_z = { (ll,mm,nn) : -D_z/2 * (ll-nn)**2 }

    return sum_vecs(image_mu, image_nu, image_z)

# collective-spin decoherence "Q" cross term
def op_image_decoherence_Q_collective(op, SS, dec_vec, mu):
    ll, mm, nn = op
    g_p, g_z, g_m = dec_vec
    if mu == 1:
        g_mu, g_nu = g_p, g_m
    else:
        g_mu, g_nu = g_m, g_p

    gg_mp = np.conj(g_nu) * g_mu
    gg_zp = np.conj(g_z) * g_mu
    gg_mz = np.conj(g_nu) * g_z
    gg_P = ( gg_zp + gg_mz ) / 2
    gg_M = ( gg_zp - gg_mz ) / 2

    image_P = {}
    if gg_mp != 0:
        image_P = { (ll+2,kk,nn) : -gg_mp * (2**(mm-kk-1)-1) * binom(mm,kk)
                    for kk in range(mm-1) }
        if nn >= 1:
            op_1 = (ll+1, mm, nn-1)
            add_left(image_P, ext_binom_op(*op_1, [ nn, 2 ], 1), nn * gg_mp)
            del image_P[(ll+1,mm+1,nn-1)]
            add_left(image_P, {op_1 : nn*(-nn+1) * gg_mp})
        if nn >= 2:
            vec = { (ll, mm, nn-2) : -nn*(nn-1) * gg_mp }
            coefficients = [ (nn-1)*(nn-2)/2, (2*nn-3), 2 ]
            add_left(image_P, insert_z_poly(vec, coefficients))

    image_L = {}
    image_M = {}
    if gg_P != 0 or gg_M != 0:
        factor_ll = mu * ( (ll-nn+1/2) * gg_P + (ll+1/2) * gg_M )
        factor_nn = mu * ( (ll-nn+1/2) * gg_P + (nn+1/2) * gg_M )
        image_L = binom_op(ll+1, mm, nn, 1, factor_ll)
        image_L[(ll+1,mm,nn)] -= factor_nn
        add_left(image_L, ext_binom_op(ll+1, mm, nn, [ 0, 1 ], 1, mu * gg_M))
        del image_L[(ll+1,mm+1,nn)]

        if nn >= 1:
            factor_mm_0 = (ll-nn+1/2) * gg_P + (ll-1/2) * gg_M
            factor_mm_1 = (ll-nn+1/2) * gg_P + (ll+nn/2-1) * gg_M
            factors = [ -mu*nn*(nn-1) * factor_mm_0,
                        -2*mu*nn * factor_mm_1,
                        -2*mu*nn * gg_M ]
            image_M = { (ll,mm+jj,nn-1) : factors[jj] for jj in range(3) }

    return sum_vecs(image_P, image_L, image_M)


##########################################################################################
# image of operators under the time derivative operator
##########################################################################################

# convert decoherence rates and transformation matrix to decoherence vectors
def get_dec_vecs(dec_rates, dec_mat):
    if dec_rates == []: return []
    dec_vecs = []
    for jj in range(3):
        dec_vec_g = dec_mat[:,jj] * np.sqrt(dec_rates[0][jj])
        dec_vec_G = dec_mat[:,jj] * np.sqrt(dec_rates[1][jj])
        if max(abs(dec_vec_g)) == 0 and max(abs(dec_vec_G)) == 0: continue
        dec_vecs.append((dec_vec_g,dec_vec_G))
    return dec_vecs

# compute image of a single operator from decoherence
def op_image_decoherence(op, SS, dec_vec, mu):
    dec_vec_g, dec_vec_G = dec_vec

    image = {}
    image = sum_vecs(op_image_decoherence_diag_individual(op, SS, dec_vec_g, mu),
                     op_image_decoherence_diag_collective(op, SS, dec_vec_G, mu))

    for image_Q, dec_vec in [ ( op_image_decoherence_Q_individual, dec_vec_g ),
                              ( op_image_decoherence_Q_collective, dec_vec_G ) ]:
        Q_lmn = image_Q(op, SS, dec_vec, mu)
        if op[0] == op[2]:
            Q_nml = Q_lmn
        else:
            Q_nml = image_Q(op[::-1], SS, dec_vec, mu)
        add_left(image, Q_lmn)
        add_left(image, conj_vec(Q_nml))

    return image

# compute image of a single operator from coherent evolution
def op_image_coherent(op, h_vec):
    if op == (0,0,0): return {}
    image = {}
    for h_op, h_val in h_vec.items():
        add_left(image, multiply_terms(h_op, op), +1j*h_val)
        add_left(image, multiply_terms(op, h_op), -1j*h_val)
    return image

# full image of a single operator under the time derivative operator
def op_image(op, h_vec, SS, dec_vecs, mu):
    image = op_image_coherent(op, h_vec)
    for dec_vec in dec_vecs:
        add_left(image, op_image_decoherence(op, SS, dec_vec, mu))
    return clean(image)


##########################################################################################
# collective-spin correlators
##########################################################################################

# list of operators necessary for computing squeezing with (\mu,\z,\nu) exponents
squeezing_ops = [ (0,1,0), (0,2,0), (1,0,0), (2,0,0), (1,1,0), (1,0,1) ]

# return correlators from evolution under a general Hamiltonian
def compute_correlators(spin_num, order_cap, chi_times, initial_state, h_vec,
                        dec_rates = [], dec_mat = None, method = "taylor", mu = 1,
                        return_derivs = False):
    assert(method == "taylor" or method == "diffeq")
    if return_derivs: assert(method == "taylor")

    state_sign, state_dir = initial_state
    state_dir = initial_state[1]
    assert(state_sign in [ "+", "-" ])
    assert(state_dir in [ "Z", "X", "Y" ])
    nu = +1 if state_sign == "+" else -1

    total_spin = spin_num/2
    if state_dir == "Z":
        if mu == nu:
            init_ln_val = lambda op : op_ln_val_Z_p(op, total_spin)
            init_val_sign = lambda op : 1
        else:
            init_ln_val = lambda op : op_ln_val_Z_m(op, total_spin)
            init_val_sign = lambda op : (-1)**op[1]
    else: # if state_dir in [ "X", "Y" ]
        init_ln_val = lambda op : op_ln_val_X(op, total_spin)
        if state_dir == "X":
            init_val_sign = lambda op : (-1)**op[1] * nu**(op[0]-op[2])
        if state_dir == "Y":
            init_val_sign = lambda op : (-1)**op[1] * (1j*mu*nu)**(op[0]-op[2])

    if dec_mat is None: dec_mat = np.eye(3)
    dec_vecs = get_dec_vecs(dec_rates, dec_mat)

    # arguments for computing operator pre-image under infinitesimal time translation
    op_image_args = ( convert_zxy(h_vec,mu), total_spin, dec_vecs, mu )

    if method == "taylor":
        derivs = compute_squeezing_derivs(order_cap, op_image_args,
                                          init_ln_val, init_val_sign, initial_state)
        if return_derivs: return derivs
        times_k = np.array([ chi_times**order for order in range(order_cap) ])
        correlators = { op : derivs[op] @ times_k for op in squeezing_ops }

    else: # method == "diffeq"
        correlators = compute_correlators_diffeq(chi_times, order_cap, op_image_args,
                                                 init_ln_val, init_val_sign)

    if mu == 1:
        return correlators
    else:
        reversed_corrs = {}
        reversed_corrs[(0,1,0)] = -correlators[(0,1,0)]
        reversed_corrs[(0,2,0)] = correlators[(0,2,0)]
        reversed_corrs[(1,0,0)] = np.conj(correlators[(1,0,0)])
        reversed_corrs[(2,0,0)] = np.conj(correlators[(2,0,0)])
        reversed_corrs[(1,1,0)] = np.conj(-correlators[(1,1,0)] - correlators[(1,0,0)])
        reversed_corrs[(1,0,1)] = correlators[(1,0,1)] - 2 * correlators[(0,1,0)]
        return reversed_corrs


# compute derivatives of squeezing operators:
#   derivs[op][kk] = < (d/dt)^kk op >_0 / kk!
def compute_squeezing_derivs(order_cap, op_image_args, init_ln_val, init_val_sign,
                             initial_state):

    if initial_state == "-Z":
        chop_operators = True
        h_vec, dec_vecs = op_image_args[0], op_image_args[-2]
        max_transverse_step = max( op[0] for op in h_vec.keys() )
        pp, zz, mm = 0, 1, 2 # conventional ordering for decoherence vectors
        for vec in dec_vecs:
            if vec[zz][zz] != 0: max_transverse_step = max(max_transverse_step,1)
            if vec[zz][pp] != 0: max_transverse_step = max(max_transverse_step,2)
            if vec[pp][zz] != 0:
                if vec[pp][pp] != 0 or vec[pp][mm] != 0:
                    # through M_{\ell mn}
                    max_transverse_step = max(max_transverse_step,1)
            if vec[pp][pp] != 0:
                if vec[pp][mm] != 0:
                    # through \tilde P_{\ell mn}
                    max_transverse_step = max(max_transverse_step,2)
                else:
                    # through S_\mu jump operator
                    max_transverse_step = max(max_transverse_step,1)
    else:
        chop_operators = False

    diff_op = {} # single time derivative operator
    time_derivs = {} # [ sqz_op, derivative_order ] --> vector
    for sqz_op in squeezing_ops:
        time_derivs[sqz_op,0] = { sqz_op : 1 }
        for order in range(1,order_cap):
            # compute relevant matrix elements of the time derivative operator
            time_derivs[sqz_op,order] = {}
            for op, val in time_derivs[sqz_op,order-1].items():
                try: add_left(time_derivs[sqz_op,order], diff_op[op], val/order)
                except:
                    diff_op[op] = op_image(op, *op_image_args)
                    if op[0] != op[-1]:
                        diff_op[op[::-1]] = conj_vec(diff_op[op])
                    add_left(time_derivs[sqz_op,order], diff_op[op], val/order)
            clean(time_derivs[sqz_op,order])

            if chop_operators and order > order_cap // 2:
                # throw out operators with no contribution to correlators
                max_steps = (order_cap-order) * max_transverse_step
                irrelevant_ops = [ op for op in time_derivs[sqz_op,order].keys()
                                   if op[0] > max_steps or op[2] > max_steps ]
                for op in irrelevant_ops:
                    del time_derivs[sqz_op,order][op]

    # compute initial values of relevant operators
    init_ln_vals = {}
    for sqz_op in squeezing_ops:
        for order in range(order_cap):
            for op in time_derivs[sqz_op,order]:
                if init_ln_vals.get(op) == None:
                    init_ln_vals[op] = init_ln_val(op)
                    if op[0] != op[-1]:
                        try: init_ln_vals[op[::-1]] = init_ln_vals[op]
                        except: init_ln_vals[op[::-1]] = None

    derivs = {}
    for sqz_op in squeezing_ops:
        derivs[sqz_op] = np.zeros(order_cap, dtype = complex)
        for order in range(order_cap):
            for T_op, T_val in time_derivs[sqz_op,order].items():
                init_ln_val = init_ln_vals[T_op]
                if init_ln_val is None: continue
                term_ln_mag = np.log(complex(T_val)) + init_ln_val
                derivs[sqz_op][order] += np.exp(term_ln_mag) * init_val_sign(T_op)

    return derivs

# compute correlators by solving an initial value problem (i.e. differential equation)
def compute_correlators_diffeq(chi_times, order_cap, op_image_args,
                               init_ln_val, init_val_sign, ivp_tolerance = 1e-10):
    op_num = 0 # counts number of operaters we keep track of
    op_idx = {} # dictionary taking operator to unique integer index

    # construct derivative operator and initial values
    init_vals = {}
    diff_op = {}
    new_ops = squeezing_ops
    for order in range(order_cap):
        for op in new_ops:
            try: diff_op[op]; continue
            except: None
            diff_op[op] = op_image(op, *op_image_args)
            op_idx[op] = op_num
            op_num += 1
            try: init_vals[op] = np.exp(init_ln_val(op)) * init_val_sign(op)
            except: init_vals[op] = 0
            if op[0] != op[-1]:
                diff_op[op[::-1]] = conj_vec(diff_op[op])
                op_idx[op[::-1]] = op_num
                op_num += 1
                try: init_vals[op[::-1]] = np.conj(init_vals[op])
                except: init_vals[op[::-1]] = None
        new_ops = set([ op for new_op in new_ops for op in diff_op[new_op].keys() ])

    init_vec = np.array([ val for op, val in init_vals.items() ]).astype(complex)
    diff_mat = scipy.sparse.dok_matrix((op_num,op_num), dtype = complex)
    for op_out, vec_out in diff_op.items():
        for op_in, val in vec_out.items():
            try: diff_mat[op_idx[op_out],op_idx[op_in]] = val
            except: None # todo: deal with this exception properly!
    diff_mat = diff_mat.tobsr()

    def time_derivative(time, vec): return diff_mat.dot(vec)

    ivp_solution = solve_ivp(time_derivative, (0,chi_times[-1]), init_vec,
                             t_eval = chi_times, rtol = ivp_tolerance)

    return { op : ivp_solution.y[op_idx[op],:] for op in squeezing_ops }

# exact correlators for OAT with decoherence; derivations in foss-feig2013nonequilibrium
# here D_\mu are decoherence rates associated with jump operators \sqrt{D_\mu} \sigma_\mu,
#   which implies ( D_p, D_z, D_m ) = ( \Gamma_{du}, \Gamma_{el}/4, \Gamma_{ud} )
#   for decoherence rates \Gamma_X appearing in foss-feig2013nonequilibrium
def correlators_OAT(spin_num, chi_times, dec_rates):
    N = spin_num
    SS = spin_num/2
    t = chi_times
    D_p, D_z, D_m = dec_rates

    gam = -(D_p - D_m) / 2
    lam = (D_p + D_m) / 2
    rr = D_p * D_m
    Gam = 2*D_z + lam

    if D_m != 0 or D_p != 0:
        Sz_unit = (D_p-D_m)/(D_p+D_m) * (1-np.exp(-(D_p+D_m)*t))
    else:
        Sz_unit = np.zeros(len(chi_times))
    Sz = SS * Sz_unit
    Sz_Sz = SS * (1/2 + (SS-1/2) * Sz_unit**2)

    def s(X): return X + 1j*gam
    def sup_t_sinc(t,z): # e^(-\lambda t) t \sinc(t z)
        if z != 0: return ( np.exp(-(lam-1j*z)*t) - np.exp(-(lam+1j*z)*t) ) / 2j / z
        else: return np.exp(-lam*t) * t
    def Phi(X):
        z = np.sqrt(s(X)**2-rr)
        val_cos = ( np.exp(-(lam-1j*z)*t) + np.exp(-(lam+1j*z)*t) ) / 2
        val_sin = lam * sup_t_sinc(t,z)
        return val_cos + val_sin
    def Psi(X):
        z = np.sqrt(s(X)**2-rr)
        return (1j*s(X)-gam) * sup_t_sinc(t,z)

    Sp = SS * np.exp(-Gam*t) * Phi(1)**(N-1)
    Sp_Sz = -1/2 * Sp + SS * (SS-1/2) * np.exp(-Gam*t) * Psi(1) * Phi(1)**(N-2)
    Sp_Sp = SS * (SS-1/2) * np.exp(-2*Gam*t) * Phi(2)**(N-2)
    Sp_Sm = SS + Sz + SS * (SS-1/2) * np.exp(-2*Gam*t) # note that Phi(0) == 1

    return { (0,1,0) : Sz,
             (0,2,0) : Sz_Sz,
             (1,0,0) : Sp,
             (2,0,0) : Sp_Sp,
             (1,1,0) : Sp_Sz,
             (1,0,1) : Sp_Sm }
