#!/usr/bin/env python3

# FILE CONTENTS: methods for computing collective spin correlators

import itertools, scipy
import numpy as np

from scipy.special import gamma, gammaln
from scipy.special import binom as scipy_binom
from sympy.functions.combinatorial.numbers import stirling as sympy_stirling


##########################################################################################
# common methods modified to recycle values
##########################################################################################

# factorial
def factorial(nn, factorials = {}):
    try: return factorials[nn]
    except:
        factorials[nn] = gamma(nn+1)
        return factorials[nn]

# logarithm of factorial
def ln_factorial(nn, ln_factorials = {}):
    try: return ln_factorials[nn]
    except:
        ln_factorials[nn] = gammaln(nn+1)
        return ln_factorials[nn]

# binomial coefficient
def binom(nn, kk, binoms = {}):
    try: return binoms[(nn,kk)]
    except:
        binoms[(nn,kk)] = scipy_binom(nn,kk)
        return binoms[(nn,kk)]

# unsigned stirling number of the first kind
def stirling(nn, kk, stirlings = {}):
    try: return stirlings[(nn,kk)]
    except:
        stirlings[(nn,kk)] = float(sympy_stirling(nn, kk, kind = 1, signed = False))
        return stirlings[(nn,kk)]

# coefficient for computing a product of spin operators
def zeta(mm, nn, pp, qq, zetas = {}):
    try: return zetas[(mm,nn,pp,qq)]
    except:
        prefactor = (-1)**(pp-qq) * 2**qq
        sum_ss = sum([ binom(ss,qq) * stirling(pp,ss) * (mm+nn-2*pp)**(ss-qq)
                       for ss in range(qq,pp+1) ])
        zetas[(mm,nn,pp,qq)] = prefactor * sum_ss
        zetas[(nn,mm,pp,qq)] = prefactor * sum_ss
    return zetas[(mm,nn,pp,qq)]


##########################################################################################
# expectation values
##########################################################################################

# natural logarithm of factors which appear in the expetation value for |X>
def ln_factors_X(SS, kk, ll, nn):
    return ( ln_factorial(SS + kk)
             - ln_factorial(SS - kk)
             - ln_factorial(SS + kk - ll)
             - ln_factorial(SS + kk - nn) )
ln_factors_X = np.vectorize(ln_factors_X)

# correlator < X | S_+^ll S_\z^mm S_-^nn | X >
def op_val_X(op, SS):
    ll, mm, nn = op
    if ll == 0 and nn == 0 and mm % 2 == 1: return 0
    if max(ll,nn) > 2*SS: return 0
    ln_prefactor = ln_factorial(2*SS) - 2*SS*np.log(2)
    k_vals = np.arange(-SS+max(ll,nn),SS+0.5)
    terms = k_vals**mm * np.exp(ln_factors_X(SS,k_vals,ll,nn) + ln_prefactor)
    return terms.sum()

# correlator < Z | S_+^ll S_\z^mm S_-^nn | Z >
def op_val_Z_p(op, SS):
    ll, mm, nn = op
    if ll != nn: return 0
    spin_num = int(round(2*SS))
    if nn > spin_num: return 0
    ln_factorials_num = ln_factorial(spin_num) + ln_factorial(nn)
    ln_factorials_den = ln_factorial(spin_num-nn)
    return (SS-nn)**mm * np.exp(ln_factorials_num - ln_factorials_den)

# correlator < Z | S_-^ll S_\z^mm S_+^nn | Z >
def op_val_Z_m(op, SS):
    ll, mm, nn = op
    if ll != 0 or nn != 0: return 0
    return SS**mm

# correlator ln < X | S_+^ll S_\z^mm S_-^nn | X >
def op_ln_val_X(op, SS):
    ll, mm, nn = op
    if ll == 0 and nn == 0 and mm % 2 == 1: return None
    if max(ll,nn) > 2*SS: return None

    offset = SS % 1
    k_vals_p = np.arange(1-offset,SS+offset)
    k_vals_n = np.arange(1-offset,SS-max(ll,nn)+offset)

    sign_n = ( 1 if mm % 2 == 0 else -1 )
    ln_prefactor = ln_factorial(2*SS) - 2*SS*np.log(2)
    ln_factors = lambda kk : ln_factors_X(SS,kk,ll,nn)
    ln_terms_p = mm * np.log(k_vals_p) + ln_factors(+k_vals_p) + ln_prefactor
    ln_terms_n = mm * np.log(k_vals_n) + ln_factors(-k_vals_n) + ln_prefactor
    ln_term_max = max(ln_terms_p.max(),ln_terms_n.max())

    terms_p = np.exp(ln_terms_p - ln_term_max)
    terms_n = np.exp(ln_terms_n - ln_term_max)

    return ln_term_max + np.log( terms_p.sum() + sign_n**mm * terms_n.sum() )

# correlator < Z | S_+^ll S_\z^mm S_-^nn | Z >
def op_ln_val_Z_p(op, SS):
    ll, mm, nn = op
    if ll != nn: return None
    spin_num = int(round(2*SS))
    if nn > spin_num: return None
    ln_factorials_num = ln_factorial(spin_num) + ln_factorial(nn)
    ln_factorials_den = ln_factorial(spin_num-nn)
    return mm*np.log(SS-nn) + ln_factorials_num - ln_factorials_den

# correlator < Z | S_-^ll S_\z^mm S_+^nn | Z >
def op_ln_val_Z_m(op, SS):
    ll, mm, nn = op
    if ll != 0 or nn != 0: return None
    return mm*np.log(SS)


##########################################################################################
# machinery for manipulating operator vectors
##########################################################################################

# clean up a dictionary vector
def clean(vec):
    null_ops = [ op for op, val in vec.items() if abs(val) == 0 ]
    for op in null_ops: del vec[op]
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

# return vector S_\mu^ll (x + S_\z)^mm * S_\bmu^nn
def binom_op(ll, mm, nn, x, prefactor = 1):
    return { (ll,kk,nn) : prefactor * x**(mm-kk) * binom(mm,kk) for kk in range(mm+1) }

# takes S_\mu^ll S_\z^mm S_\bmu^nn
#   --> S_\mu^ll ( \sum_jj x_jj S_\z^jj ) S_\z^mm + S_\bmu^nn
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
def ext_binom_op(ll, mm, nn, x, terms, prefactor = 1):
    return insert_z_poly(binom_op(ll,mm,nn,x), terms, prefactor)


##########################################################################################
# general commutator between ordered products of collective spin operators
##########################################################################################

# simplify product of two operators
def multiply_terms(op_left, op_right, mu):
    pp, qq, rr = op_left
    ll, mm, nn = op_right
    vec = {}
    binom_qq = [ binom(qq,bb) for bb in range(qq+1) ]
    binom_mm = [ binom(mm,cc) for cc in range(mm+1) ]
    for kk in range(min(rr,ll)+1):
        kk_fac = factorial(kk) * binom(rr,kk) * binom(ll,kk)
        aa_facs = [ (-1)**aa * zeta(rr,ll,kk,aa) for aa in range(kk+1) ]
        for aa, bb, cc in itertools.product(range(kk+1),range(qq+1),range(mm+1)):
            bb_fac = (ll-kk)**(qq-bb) * binom_qq[bb]
            cc_fac = (rr-kk)**(mm-cc) * binom_mm[cc]
            kabc_fac = kk_fac * aa_facs[aa] * bb_fac * cc_fac
            sign = mu**(qq+mm+aa-bb-cc)
            op_in = (pp+ll-kk, aa+bb+cc, rr+nn-kk)
            try:
                vec[op_in] += sign * kabc_fac
            except:
                vec[op_in] = sign * kabc_fac
    return clean(vec)

# simplify product of two vectors
def multiply_vecs(vec_left, vec_right, mu, prefactor = 1):
    vec = {}
    for term_left, val_left in vec_left.items():
        for term_right, val_right in vec_right.items():
            fac = val_left * val_right * prefactor
            add_left(vec, multiply_terms(term_left, term_right, mu), fac)
    return vec


##########################################################################################
# miscellaneous methods for changing frames and operator vectors
##########################################################################################

# decoherence transformation matrix from a periodic drive; here A = J_0(2\beta), where:
#   J_0 is the zero-order bessel function of the first kind
#   \beta is the modulation index
def dec_mat_drive(A):
    const = np.array([[ 0, 0, 0 ],
                      [ 0, 1, 1 ],
                      [ 0, 1, 1 ]]) * 1/2
    var = np.array([[ 2,  0,  0 ],
                    [ 0,  1, -1 ],
                    [ 0, -1,  1 ]]) * 1/2
    return const + A * var

# convert vector from (z,x,y) format to (mu,z,nu) format
def convert_zxy(vec_zxy, mu = 1):
    vec = {}
    # define vectors for (2 * Sx) and (-i * 2 * Sy)
    Sx_2 = { (1,0,0) : 1,
             (0,0,1) : 1 }
    Sy_ni2 = { (1,0,0) : -1,
               (0,0,1) :  1 }
    for op_zxy, val_zxy in vec_zxy.items():
        ll, mm, nn = op_zxy
        lmn_fac = (1j*mu)**nn / 2**(mm+nn) * val_zxy
        # starting from the left, successively multiply all factors on the right
        lmn_vec = { (0,ll,0) : 1 }
        for jj in range(mm): lmn_vec = multiply_vecs(lmn_vec, Sx_2, mu)
        for kk in range(nn): lmn_vec = multiply_vecs(lmn_vec, Sy_ni2, mu)
        add_left(vec, lmn_vec, lmn_fac)
    if np.array([ np.imag(val) == 0 for val in vec.values() ]).all():
        vec = { op : np.real(val) for op, val in vec.items() }
    return clean(vec)


##########################################################################################
# single-spin decoherence
##########################################################################################

# diagonal terms of single-spin decoherence
def op_image_decoherence_diag_individual(op, S, dec_vec, mu):
    ll, mm, nn = op
    D_z, D_p, D_m = abs(np.array(dec_vec))**2
    if mu == 1:
        D_mu, D_nu = D_p, D_m
    else:
        D_mu, D_nu = D_m, D_p

    image_mu = {}
    if D_mu != 0:
        image_mu = ext_binom_op(*op, mu, [ S-ll-nn, -mu ], D_mu)
        add_left(image_mu, insert_z_poly({op:1}, [ S-(ll+nn)/2, -mu ]), -D_mu)
        if ll >= 1 and nn >= 1:
            image_mu[(ll-1,mm,nn-1)] = ll*nn * (2*S-ll-nn+2) * D_mu
        if ll >= 2 and nn >= 2:
            op_2 = (ll-2, mm, nn-2)
            factor = ll*nn*(ll-1)*(nn-1)
            image_mu.update(ext_binom_op(*op_2, -mu, [ S, mu ], factor * D_mu))

    image_nu = {}
    if D_nu != 0:
        image_nu = ext_binom_op(*op, -mu, [ S, mu ], D_nu)
        add_left(image_nu, insert_z_poly({op:1}, [ S+(ll+nn)/2, mu ]), -D_nu)

    image_z = {}
    if D_z != 0 and ll + nn != 0:
        image_z = { (ll,mm,nn) : -2*(ll+nn) * D_z }
        if ll >= 1 and nn >= 1:
            image_z.update(ext_binom_op(ll-1, mm, nn-1, -mu, [ S, mu ], 4*ll*nn * D_z))

    return sum_vecs(image_mu, image_nu, image_z)

# single-spin decoherence "Q" cross term
def op_image_decoherence_Q_individual(op, S, dec_vec, mu):
    ll, mm, nn = op
    g_z, g_p, g_m = dec_vec
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
            image_P = ext_binom_op(ll, mm, nn-2, -mu, [ S, mu ], -nn*(nn-1) * gg_mp)
        image_P.update({ (ll+1,mm,nn-1) : nn * gg_mp })

    image_K = {}
    if gg_zp + gg_mz != 0:
        image_K = binom_op(ll+1, mm, nn, mu, mu/2 * (gg_zp + gg_mz))
        del image_K[(ll+1,mm,nn)]

    image_L = {}
    if gg_zp != 0 and nn != 0:
        if nn >= 2 and ll >= 1:
            factor = -2*mu*ll*nn*(nn-1)
            image_L = ext_binom_op(ll-1, mm, nn-2, -mu, [ S, mu ], factor * gg_zp)
        coefficients = [ -2*S+2*ll+3/2*(nn-1), mu ]
        image_L.update(insert_z_poly({(ll,mm,nn-1):1}, coefficients, mu*nn * gg_zp))

    image_M = {}
    if gg_mz != 0 and nn != 0:
        image_M = ext_binom_op(ll, mm, nn-1, -mu, [ S, mu ], 2*mu*nn * gg_mz)
        coefficients = [ (nn-1)/2, mu ]
        add_left(image_M, insert_z_poly({(ll,mm,nn-1):1}, coefficients, -mu*nn * gg_mz))

    return sum_vecs(image_P, image_K, image_L, image_M)


##########################################################################################
# collective-spin decoherence
##########################################################################################

# diagonal terms of collective-spin decoherence
def op_image_decoherence_diag_collective(op, S, dec_vec, mu):
    ll, mm, nn = op
    D_z, D_p, D_m = abs(np.array(dec_vec))**2
    if mu == 1:
        D_mu, D_nu = D_p, D_m
    else:
        D_mu, D_nu = D_m, D_p

    image_mu = {}
    if D_mu != 0:
        image_mu = { (ll+1,kk,nn+1) : D_mu * (2**(mm-kk)-1) * mu**(mm-kk) * binom(mm,kk)
                     for kk in range(mm) }
        coefficients = [ ll*(ll+1) + nn*(nn+1), 2*mu*(ll+nn+1) ]
        image_mu.update(ext_binom_op(*op, mu, coefficients, -D_mu))
        coefficients = [ ll*(ll+1) + nn*(nn+1), 2*mu*(ll+nn+2) ]
        add_left(image_mu, insert_z_poly({op:1}, coefficients, D_mu/2))
        if ll >= 1 and nn >= 1:
            vec = { (ll-1,mm,nn-1) : 1 }
            coefficients = [ (ll-1)*(nn-1), 2*mu*(ll+nn-2), 4 ]
            image_mu.update(insert_z_poly(vec, coefficients, ll*nn * D_mu))

    image_nu = {}
    if D_nu != 0:
        image_nu = binom_op(ll+1, mm, nn+1, mu, -D_nu)
        del image_nu[(ll+1,mm,nn+1)]
        coefficients = [ ll*(ll-1) + nn*(nn-1), 2*mu*(ll+nn) ]
        image_nu.update(insert_z_poly({op:1}, coefficients, D_nu/2))

    image_z = {}
    if D_z != 0 and ll != nn:
        image_z = { (ll,mm,nn) : -D_z/2 * (ll-nn)**2 }

    return sum_vecs(image_mu, image_nu, image_z)

# collective-spin decoherence "Q" cross term
def op_image_decoherence_Q_collective(op, S, dec_vec, mu):
    ll, mm, nn = op
    g_z, g_p, g_m = dec_vec
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
        image_P = { (ll+2,kk,nn) : -gg_mp * (2**(mm-kk-1)-1) * mu**(mm-kk) * binom(mm,kk)
                    for kk in range(mm-1) }
        if nn >= 1:
            op_1 = (ll+1, mm, nn-1)
            add_left(image_P, ext_binom_op(*op_1, mu, [ nn, 2*mu ]), nn * gg_mp)
            add_left(image_P, insert_z_poly({op_1:-1}, [ nn-1, 2*mu ]), nn * gg_mp)
        if nn >= 2:
            vec = { (ll, mm, nn-2) : -nn*(nn-1) * gg_mp }
            coefficients = [ (nn-1)*(nn-2)/2, mu*(2*nn-3), 2 ]
            add_left(image_P, insert_z_poly(vec, coefficients))

    image_L = {}
    image_M = {}
    if gg_P != 0 or gg_M != 0:
        factor = mu * ( (ll-nn+1/2) * gg_P + (ll+1/2) * gg_M )
        image_L = binom_op(ll+1, mm, nn, mu, factor)
        image_L[(ll+1,mm,nn)] += -mu * ( (ll-nn+1/2) * gg_P + (nn+1/2) * gg_M )
        add_left(image_L, ext_binom_op(ll+1, mm, nn, mu, [0,1], gg_M))
        del image_L[(ll+1,mm+1,nn)]

        if nn >= 1:
            factors = [ mu*(nn-1) * ( (ll-nn+1/2) * gg_P + (ll-1/2) * gg_M ),
                        2 * ( (ll-nn+1/2) * gg_P + (ll+nn/2-1) * gg_M ),
                        2*mu * gg_M ]
            image_M = { (ll,mm+jj,nn-1) : -nn * factors[jj] for jj in range(3) }

    return sum_vecs(image_P, image_L, image_M)


##########################################################################################
# image of operators under the time derivative operator
##########################################################################################

# convert decoherence rates and transformation matrix to decoherence vectors
def get_dec_vecs(dec_rates, dec_mat):
    dec_vecs = []
    for jj in range(3):
        dec_vec_g = dec_mat[:,jj] * np.sqrt(dec_rates[0][jj])
        dec_vec_G = dec_mat[:,jj] * np.sqrt(dec_rates[1][jj])
        if dec_vec_g is np.zeros(3) and dec_vec_G is np.zeros(3): continue
        if jj == 0: dec_vec_g /= np.sqrt(2)
        dec_vecs.append((dec_vec_g,dec_vec_G))
    return dec_vecs

# compute image of a single operator from decoherence
def op_image_decoherence(op, S, dec_vec, mu):
    dec_vec_g, dec_vec_G = dec_vec

    image = {}
    image = sum_vecs(op_image_decoherence_diag_individual(op, S, dec_vec_g, mu),
                     op_image_decoherence_diag_collective(op, S, dec_vec_G, mu))

    for image_Q, dec_vec in [ ( op_image_decoherence_Q_individual, dec_vec_g ),
                              ( op_image_decoherence_Q_collective, dec_vec_G ) ]:
        Q_lmn = image_Q(op, S, dec_vec, mu)
        if op[0] == op[2]:
            Q_nml = Q_lmn
        else:
            Q_nml = image_Q(op[::-1], S, dec_vec, mu)
        add_left(image, Q_lmn)
        add_left(image, conj_vec(Q_nml))

    return image

# compute image of a single operator from coherent evolution
def op_image_coherent(op, h_vec, mu):
    if op == (0,0,0): return {}
    image = {}
    for h_op, h_val in h_vec.items():
        add_left(image, multiply_terms(h_op, op, mu), +1j*h_val)
        add_left(image, multiply_terms(op, h_op, mu), -1j*h_val)
    return image

# full image of a single operator under the time derivative operator
def op_image(op, h_vec, S, dec_vecs, mu):
    image = op_image_coherent(op, h_vec, mu)
    for dec_vec in dec_vecs:
        add_left(image, op_image_decoherence(op, S, dec_vec, mu))
    return clean(image)

# compute time derivative of a given vector of spin operators
def compute_time_deriv(diff_op, deriv_order, input_vector, op_image_args):
    output_vector = {}
    # for each operator in the input vector
    for input_op, input_val in input_vector.items():
        prefactor = input_val / deriv_order
        try:
            add_left(output_vector, diff_op[input_op], prefactor)
        except: # we do not know the time derivative of this operator, so compute it
            diff_op[input_op] = op_image(input_op, *op_image_args)
            # we get the time derivative of the conjugate operator for free
            if input_op[0] != input_op[-1]:
                diff_op[input_op[::-1]] = conj_vec(diff_op[input_op])
            add_left(output_vector, diff_op[input_op], prefactor)
    return clean(output_vector)


##########################################################################################
# collective-spin correlators
##########################################################################################

# return correlators from evolution under a general Hamiltonian
def compute_correlators(spin_num, order_cap, chi_times, initial_state, h_vec,
                        dec_rates = [(0,0,0),(0,0,0)], dec_mat = None, mu = 1,
                        init_vals_XY = {}, init_vals_Z_p = {}, init_vals_Z_m = {}):
    state_sign, state_dir = initial_state
    state_dir = initial_state[1]
    assert(state_sign in [ "+", "-" ])
    assert(state_dir in [ "Z", "X", "Y" ])
    nu = +1 if state_sign == "+" else -1

    total_spin = spin_num/2
    if state_dir == "Z":
        if mu == nu:
            initial_val = lambda op : op_ln_val_Z_p(op, total_spin)
            initial_vals = init_vals_Z_p
        else:
            initial_val = lambda op : op_ln_val_Z_m(op, total_spin)
            initial_vals = init_vals_Z_m
        init_val_sign = lambda op : nu**op[1]
    else: # if state_dir in [ "X", "Y" ]
        initial_vals = init_vals_XY
        initial_val = lambda op : op_ln_val_X(op, total_spin)
        if state_dir == "X":
            init_val_sign = lambda op : (-mu)**op[1] * nu**(op[0]-op[2])
        if state_dir == "Y":
            init_val_sign = lambda op : (-mu)**op[1] * (1j*mu*nu)**(op[0]-op[2])

    if dec_mat is None:
        dec_mat = np.eye(3)
    dec_vecs = get_dec_vecs(dec_rates, dec_mat)

    # list of operators necessary for computing squeezing with (\mu,\z,\bmu) exponents
    squeezing_ops = [ (0,1,0), (0,2,0), (1,0,0), (2,0,0), (1,1,0), (1,0,1) ]

    # arguments for computing operator pre-image under infinitesimal time translation
    op_image_args = ( convert_zxy(h_vec,mu), total_spin, dec_vecs, mu )

    # compute all images under time derivatives
    diff_op = {} # time derivative operator
    time_derivs = {} # [ sqz_op ][ derivative_order ][ operator ] --> value
    for sqz_op in squeezing_ops:
        time_derivs[sqz_op] = { 0 : { sqz_op : 1 } }
        for order in range(1,order_cap):
            time_derivs[sqz_op][order] \
                = compute_time_deriv(diff_op, order, time_derivs[sqz_op][order-1],
                                     op_image_args)

    # compute initial values of relevant operators
    for sqz_op in squeezing_ops:
        for order in range(order_cap):
            for op in time_derivs[sqz_op][order]:
                if initial_vals.get(op) == None:
                    initial_vals[op] = initial_val(op)
                    # all our initial values are real, so no need to conjugate
                    if op[0] != op[-1]:
                        initial_vals[op[::-1]] = initial_vals[op]

    T = np.array([ chi_times**order for order in range(order_cap) ]).astype(complex)
    correlators = {}
    for sqz_op in squeezing_ops:
        # compute < (d/dt)^kk S_\mu^ll S_\z^mm S_\bmu^nn >_0 / kk! for all kk
        Q = np.zeros(order_cap, dtype = complex)
        for order in range(order_cap):
            Q[order] = sum([ np.exp(np.log(complex(time_derivs[sqz_op][order][op]))
                                    + initial_vals[op]) * init_val_sign(op)
                             for op in time_derivs[sqz_op][order]
                          if initial_vals[op] is not None ])
            correlators[sqz_op] = Q @ T

    if mu == 1:
        return correlators

    else:
        reversed_corrs = {}
        reversed_corrs[(0,1,0)] = correlators[(0,1,0)]
        reversed_corrs[(0,2,0)] = correlators[(0,2,0)]
        reversed_corrs[(1,0,0)] = np.conj(correlators[(1,0,0)])
        reversed_corrs[(2,0,0)] = np.conj(correlators[(2,0,0)])
        reversed_corrs[(1,1,0)] = np.conj(correlators[(1,1,0)] - correlators[(1,0,0)])
        reversed_corrs[(1,0,1)] = correlators[(1,0,1)] + 2 * correlators[(0,1,0)]

        return reversed_corrs

# exact correlators for OAT with decoherence
# derivations in foss-feig2013nonequilibrium
def correlators_OAT(spin_num, chi_times, dec_rates):
    N = spin_num
    S = spin_num/2
    t = chi_times
    g_z, g_p, g_m = dec_rates[0]

    gam = -(g_p - g_m) / 2
    lam = (g_p + g_m) / 2
    rr = g_p * g_m
    Gam = g_z + lam

    if g_m != 0 or g_p != 0:
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
