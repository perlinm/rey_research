#!/usr/bin/env python3

# FILE CONTENTS: methods for computing collective spin correlators

import itertools, scipy
import numpy as np

from scipy.integrate import solve_ivp

from special_functions import *


##########################################################################################
# expectation values
##########################################################################################

# natural logarithm of factors which appear in the expetation value for |X>
def ln_factors_X(NN, kk, ll, nn):
    return ( ln_factorial(NN - kk)
             - ln_factorial(kk)
             - ln_factorial(NN - ll - kk)
             - ln_factorial(NN - nn - kk) )
ln_factors_X = np.vectorize(ln_factors_X)

# correlator < X | S_+^ll S_\z^mm S_-^nn | X >
def op_val_X(op, NN, vals = {}):
    ll, mm, nn = op
    ll, nn = max(ll,nn), min(ll,nn)
    try: return vals[ll,mm,nn,NN]
    except: None

    if ll == 0 and nn == 0 and mm % 2 == 1: return 0
    if ll > NN: return 0

    ln_prefactor = ln_factorial(NN) - NN*np.log(2)
    k_vals = np.arange(NN-ll+0.5, dtype = int)
    val = ((k_vals-NN/2)**mm * np.exp(ln_factors_X(NN,k_vals,ll,nn)+ln_prefactor)).sum()
    vals[ll,mm,nn,NN] = val
    return val

# correlator < Z | S_+^ll S_\z^mm S_-^nn | Z >
def op_val_Z_p(op, NN, vals = {}):
    try: return vals[ll,mm,NN]
    except: None

    ll, mm, nn = op
    if ll != nn: return 0
    if nn > NN: return 0

    ln_factorials_num = ln_factorial(NN) + ln_factorial(nn)
    ln_factorials_den = ln_factorial(NN-nn)
    val = (NN/2-nn)**mm * np.exp(ln_factorials_num - ln_factorials_den)
    vals[ll,mm,NN] = val
    return val

# correlator < Z | S_-^ll S_\z^mm S_+^nn | Z >
def op_val_Z_m(op, NN):
    ll, mm, nn = op
    if ll != 0 or nn != 0: return 0
    return (-NN/2)**mm

# correlator ln | < X | S_+^ll S_\z^mm S_-^nn | X > |
def op_ln_val_X(op, NN, vals = {}):
    ll, mm, nn = op
    ll, nn = max(ll,nn), min(ll,nn)
    try: return vals[ll,mm,nn,NN]
    except: None

    if ll == 0 and nn == 0 and mm % 2 == 1: return None
    if ll > NN: return None

    ln_prefactor = ln_factorial(NN) - NN*np.log(2)
    ln_factors = lambda kk : ln_factors_X(NN,kk,ll,nn)

    k_vals = np.arange(NN-ll+0.5, dtype = int)

    # remove kk == NN/2 term if necessary
    if mm > 0 and NN % 2 == 0 and k_vals[-1] >= NN/2:
        k_vals = np.delete(k_vals, NN//2)

    # compute the logarithm of the magnitude of each term
    ln_terms = ln_factors(k_vals) + ln_prefactor
    if mm != 0: ln_terms += mm * np.log(abs(NN/2-k_vals))

    # compute the absolute value of terms divided by the largest term
    ln_term_max = ln_terms.max()
    terms = np.exp(ln_terms-ln_term_max)

    # compute the logarithm of the sum of the terms
    if mm % 2 == 1:
        val = ln_term_max + np.log(np.sum(np.sign(NN/2-k_vals)*terms))
    else:
        val = ln_term_max + np.log(np.sum(terms))

    vals[ll,mm,nn,NN] = val
    return val

# correlator ln | < Z | S_+^ll S_\z^mm S_-^nn | Z > |
def op_ln_val_Z_p(op, NN, vals = {}):
    try: return vals[ll,mm,NN]
    except: None

    ll, mm, nn = op
    if ll != nn: return None
    if ll > NN: return None

    ln_factorials_num = ln_factorial(NN) + ln_factorial(ll)
    ln_factorials_den = ln_factorial(NN-ll)
    val = mm*np.log(abs(NN/2-ll)) + ln_factorials_num - ln_factorials_den
    vals[ll,mm,NN] = val
    return val

# correlator ln | < Z | S_-^ll S_\z^mm S_+^nn | Z > |
def op_ln_val_Z_m(op, NN):
    ll, mm, nn = op
    if ll != 0 or nn != 0: return None
    return mm*np.log(NN/2)

# return functions to compute initial values
def init_ln_val_functions(spin_num, init_state, mu = 1):
    state_sign, state_dir = init_state
    state_dir = init_state[1]
    assert(state_sign in [ "+", "-" ])
    assert(state_dir in [ "Z", "X", "Y" ])
    nu = +1 if state_sign == "+" else -1

    if state_dir == "Z":
        if mu == nu:
            init_ln_val = lambda op : op_ln_val_Z_p(op, spin_num)
            init_val_sign = lambda op : 1 if spin_num/2 > op[0] else (-1)**op[1]
        else:
            init_ln_val = lambda op : op_ln_val_Z_m(op, spin_num)
            init_val_sign = lambda op : (-1)**op[1]
    else: # if state_dir in [ "X", "Y" ]
        init_ln_val = lambda op : op_ln_val_X(op, spin_num)
        if state_dir == "X":
            init_val_sign = lambda op : (-1)**op[1] * nu**(op[0]-op[2])
        if state_dir == "Y":
            init_val_sign = lambda op : (-1)**op[1] * (1j*mu*nu)**(op[0]-op[2])

    return init_val_sign, init_ln_val


##########################################################################################
# machinery for manipulating operator vectors
##########################################################################################

# clean up a dictionary vector
def clean(vec, spin_num = None):
    null_ops = [ op for op, val in vec.items() if abs(val) == 0 ]
    for op in null_ops: del vec[op]
    if spin_num is not None:
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
    return clean(vec)


##########################################################################################
# miscellaneous methods for changing frames and operator vectors
##########################################################################################

# decoherence transformation matrix from a periodic drive; here A = J_0(\beta), where:
#   J_0 is the zero-order bessel function of the first kind
#   \beta is the modulation index
def dec_mat_drive(A, mu = 1): # in (mu,z,bmu) format
    const = np.array([[ 1, 0, 1 ],
                      [ 0, 0, 0 ],
                      [ 1, 0, 1 ]]) * 1/2
    var = np.array([[  mu, 0, -mu ],
                    [   0, 2,   0 ],
                    [ -mu, 0,  mu ]]) * 1/2
    return const + A * var

# convert 3D vectors between (z,x,y) and (mu,z,bmu) formats
def pzm_to_zxy_mat(mu = 1):
    return np.array([ [      0, 1,      0 ],
                      [      1, 0,      1 ],
                      [ +mu*1j, 0, -mu*1j ] ])

def mat_zxy_to_pzm(mat_zxy, mu = 1):
    pzm_to_zxy = pzm_to_zxy_mat(mu)
    zxy_to_pzm = np.linalg.inv(pzm_to_zxy)
    return zxy_to_pzm @ mat_zxy @ pzm_to_zxy

def mat_pzm_to_zxy(mat_pzm, mu = 1):
    pzm_to_zxy = pzm_to_zxy_mat(mu)
    zxy_to_pzm = np.linalg.inv(pzm_to_zxy)
    return pzm_to_zxy @ mat_pzm @ zxy_to_pzm

# convert operator vectors between (z,x,y) and (mu,z,bmu) formats
def vec_zxy_to_pzm(vec_zxy, mu = 1):
    vec_pzm = {}
    # define vectors for (2 * Sx) and (i * 2 * Sy)
    Sx_2 = { (1,0,0) : 1,
             (0,0,1) : 1 }
    Sy_i2 = { (1,0,0) : +mu,
              (0,0,1) : -mu }
    for op_zxy, val_zxy in vec_zxy.items():
        ll, mm, nn = op_zxy
        lmn_fac = val_zxy * mu**ll * (-1j)**nn / 2**(mm+nn)
        # starting from the left, successively multiply all factors on the right
        lmn_vec = { (0,ll,0) : 1 }
        for jj in range(mm): lmn_vec = multiply_vecs(lmn_vec, Sx_2)
        for kk in range(nn): lmn_vec = multiply_vecs(lmn_vec, Sy_i2)
        add_left(vec_pzm, lmn_vec, lmn_fac)
    if np.array([ np.imag(val) == 0 for val in vec_pzm.values() ]).all():
        vec_pzm = { op : np.real(val) for op, val in vec_pzm.items() }
    return clean(vec_pzm)

# given correlators of the form < S_\mu^ll (\mu S_z)^mm S_\bmu^nn >,
#   convert them into correlators of the form < S_\bmu^ll (\bmu S_z)^mm S_\mu^nn >
def invert_vals(vals):
    inverted_vals = {}
    target_ops = list(vals.keys())
    for ll, mm, nn in target_ops:
        if vals.get((nn,mm,ll)) is None:
            vals[(nn,mm,ll)] = np.conj(vals[(ll,mm,nn)])
    for ll, mm, nn in target_ops:
        coeffs_mm_nn = multiply_terms((0,mm,0),(nn,0,0))
        coeffs_ll_mm_nn = multiply_vecs({(0,0,ll):1}, coeffs_mm_nn, (-1)**mm)
        inverted_vals[(ll,mm,nn)] = sum([ coeff * vals[op]
                                          for op, coeff in coeffs_ll_mm_nn.items() ])
    return inverted_vals


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
        image_z = { (ll,mm,nn) : -1/2*(ll+nn) * D_z }
        if ll >= 1 and nn >= 1:
            image_z.update(ext_binom_op(ll-1, mm, nn-1, [ SS, 1 ], -1, ll*nn * D_z))

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
        image_K = binom_op(ll+1, mm, nn, 1, mu/4 * (gg_zp + gg_mz))
        del image_K[(ll+1,mm,nn)]

    image_L = {}
    if gg_zp != 0 and nn != 0:
        if nn >= 2 and ll >= 1:
            factor = -mu*ll*nn*(nn-1)
            image_L = ext_binom_op(ll-1, mm, nn-2, [ SS, 1 ], -1, factor * gg_zp)
        coefficients = [ SS-ll-3/4*(nn-1), -1/2 ]
        image_L.update(insert_z_poly({(ll,mm,nn-1):-1}, coefficients, mu*nn * gg_zp))

    image_M = {}
    if gg_mz != 0 and nn != 0:
        image_M = ext_binom_op(ll, mm, nn-1, [ SS, 1 ], -1, mu*nn * gg_mz)
        coefficients = [ (nn-1)/2, 1 ]
        add_left(image_M, insert_z_poly({(ll,mm,nn-1):1}, coefficients, -mu*nn/2 * gg_mz))

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
    if dec_mat is None: dec_mat = np.eye(3)
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
def op_image(op, h_vec, spin_num, dec_vecs, mu):
    image = op_image_coherent(op, h_vec)
    for dec_vec in dec_vecs:
        add_left(image, op_image_decoherence(op, spin_num/2, dec_vec, mu))
    return clean(image)


##########################################################################################
# collective-spin correlators
##########################################################################################

# list of operators necessary to compute squeezing, specified by (\mu,\z,\nu) exponents
squeezing_ops = [ (0,1,0), (0,2,0), (1,0,0), (2,0,0), (1,1,0), (1,0,1) ]

# compute (factorially suppresed) derivatives of operators,
# returning an operator vector for each derivative order:
# deriv_op_vec[mm,kk] = (d/dt)^kk S_mm / kk!
#                     = \sum_nn T^kk_{mm,nn} S_nn / kk!
def get_deriv_op_vec(order_cap, spin_num, init_state, h_vec,
                     dec_rates = [], dec_mat = None, deriv_ops = squeezing_ops,
                     prepend_op = None, append_op = None, mu = 1):
    dec_vecs = get_dec_vecs(dec_rates, dec_mat)

    if init_state == "-Z":
        chop_operators = True
        max_transverse_step = max([ op[0] for op in h_vec.keys() ] + [ 0 ])
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
        max_step_offset = 0
        for op_vec in [ prepend_op, append_op ]:
            if op_vec is not None:
                op_vec_offset = max( max(op[0],op[-1]) for op in op_vec.keys() )
                max_step_offset = max(op_vec_offset, max_step_offset)
    else:
        chop_operators = False

    diff_op = {} # single time derivative operator
    deriv_op_vec = {} # deriv_op_vec[mm,kk] = (1/kk!) (d/dt)^kk S_mm
                      #                     = (1/kk!) \sum_nn T^kk_{mm,nn} S_nn
    for deriv_op in deriv_ops:
        deriv_op_vec[deriv_op,0] = { deriv_op : 1 }
        for order in range(1,order_cap):
            # compute relevant matrix elements of the time derivative operator
            deriv_op_vec[deriv_op,order] = {}
            for op, val in deriv_op_vec[deriv_op,order-1].items():
                try: add_left(deriv_op_vec[deriv_op,order], diff_op[op], val/order)
                except:
                    diff_op[op] = op_image(op, h_vec, spin_num, dec_vecs, mu)
                    if op[0] != op[-1]:
                        diff_op[op[::-1]] = conj_vec(diff_op[op])
                    add_left(deriv_op_vec[deriv_op,order], diff_op[op], val/order)
            clean(deriv_op_vec[deriv_op,order])

            if chop_operators and order > order_cap // 2:
                # throw out operators with no contribution to correlators
                max_steps = (order_cap-order) * max_transverse_step + max_step_offset
                irrelevant_ops = [ op for op in deriv_op_vec[deriv_op,order].keys()
                                   if op[0] > max_steps or op[2] > max_steps ]
                for op in irrelevant_ops:
                    del deriv_op_vec[deriv_op,order][op]

    return deriv_op_vec

# compute (factorially suppresed) derivatives of operators,
# returning a value for each order:
# deriv_vals[op][kk] = < prepend_zxy * [ (d/dt)^kk op ] * append_zxy >_0 / kk!
def compute_deriv_vals(order_cap, spin_num, init_state, h_vec,
                       dec_rates = [], dec_mat = None, deriv_ops = squeezing_ops,
                       prepend_op = None, append_op = None, mu = 1):
    init_val_sign, init_ln_val = init_ln_val_functions(spin_num, init_state, mu)

    deriv_op_vec = get_deriv_op_vec(order_cap, spin_num, init_state, h_vec,
                                    dec_rates, dec_mat, deriv_ops,
                                    prepend_op, append_op, mu)
    if prepend_op is not None:
        for deriv_op, order in itertools.product(deriv_ops, range(order_cap)):
            deriv_op_vec[deriv_op,order] \
                = multiply_vecs(prepend_op,deriv_op_vec[deriv_op,order])
    if append_op is not None:
        for deriv_op, order in itertools.product(deriv_ops, range(order_cap)):
            deriv_op_vec[deriv_op,order] \
                = multiply_vecs(deriv_op_vec[deriv_op,order],append_op)

    deriv_vals = {} # deriv_vals[op][kk]
    init_ln_vals = {} # initial values of relevant operators
    for deriv_op in deriv_ops:
        deriv_vals[deriv_op] = np.zeros(order_cap, dtype = complex)
        for order in range(order_cap):
            for op, val in deriv_op_vec[deriv_op,order].items():

                init_ln_val_op = init_ln_vals.get(op)
                if init_ln_val_op is None:
                    init_ln_val_op = init_ln_val(op)
                    if init_ln_val_op is None: continue
                    init_ln_vals[op] = init_ln_val_op
                    if op[0] != op[-1]: init_ln_vals[op[::-1]] = init_ln_vals[op]

                term_ln_mag = np.log(complex(val)) + init_ln_val_op
                deriv_vals[deriv_op][order] += np.exp(term_ln_mag) * init_val_sign(op)

    return deriv_vals

# compute correlators from evolution under a general Hamiltonian with decoherence
def compute_correlators(chi_times, order_cap, spin_num, init_state, h_vec,
                        dec_rates = [], dec_mat = None, correlator_ops = squeezing_ops,
                        prepend_op = None, append_op = None, mu = 1):
    deriv_vals = compute_deriv_vals(order_cap, spin_num, init_state, h_vec,
                                    dec_rates, dec_mat, correlator_ops,
                                    prepend_op, append_op, mu)
    times_k = np.array([ chi_times**order for order in range(order_cap) ])
    correlators = { op : deriv_vals[op] @ times_k for op in correlator_ops }

    if mu == 1:
        return correlators
    else:
        # we computed correlators of the form < S_-^ll (-S_z)^mm S_+^nn >,
        #   so we need to invert them into correlators of the form < S_+^ll S_z^mm S_-^nn >
        return invert_vals(correlators)

# exact correlators for OAT with decoherence; derivations in foss-feig2013nonequilibrium
def correlators_OAT(spin_num, chi_times, dec_rates):
    N = spin_num
    SS = spin_num/2
    t = chi_times
    D_p, D_z, D_m = dec_rates

    gam = -(D_p - D_m) / 2
    lam = (D_p + D_m) / 2
    rr = D_p * D_m
    Gam = D_z/2 + lam

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
