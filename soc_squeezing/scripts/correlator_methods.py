#!/usr/bin/env python3

# FILE CONTENTS: methods for computing collective spin correlators

import numpy as np
import scipy.sparse as sparse
import scipy.linalg as linalg

from scipy.special import binom

from squeezing_methods import ln_binom

# correlator < X | S_\mu S_z^n | X > for \mu \in \set{+1,-1}
def val_mu_X(N, n, mu):
    assert(mu == 1 or mu == -1)
    S = N/2
    return sum([ m**n * np.exp( ln_binom(N,S+m) - N*np.log(2) ) * abs( m - mu*S )
                 for m in np.arange(-S,S+1) ])

# correlator < X | S_\mu S_\nu S_z^n | X > for \mu,\nu \in \set{+1,-1} and \mu >= \nu
def val_mu_nu_X(N, n, mu, nu):
    assert(mu == 1 or mu == -1)
    assert(nu == 1 or nu == -1)
    assert(mu >= nu)
    S = N/2
    return sum([ m**n
                 * np.exp( ln_binom(N,S+m)/2 + ln_binom(N,S+m+mu+nu)/2 - N*np.log(2) )
                 * np.sqrt( (S-mu*(m+nu)) * (S+mu*(m+nu)+1) * (S-nu*m) * (S+nu*m+1) )
                 for m in np.arange(-S,S-mu-nu+1) ])

# return correlators after regular one-axis twisting protocol
def correlators_OAT(N, chi_t, decay_rate_over_chi,
                    max_weight = None, print_updates = False):
    # set a default number of operators in+ea+h vector
    if max_weight == None: max_weight = 10

    # shorthand for time and decay rate in units with \chi = 1
    t, g = chi_t, decay_rate_over_chi

    Sz = N/2 * (np.exp(-g*t)-1)
    var_Sz = N/2 * (1 - np.exp(-g*t)/2) * np.exp(-g*t)
    Sz_Sz = var_Sz + Sz**2

    # construct vectors of spin corelators
    p_vec = np.zeros((len(t),max_weight), dtype = complex)
    m_vec = np.zeros((len(t),max_weight), dtype = complex)
    pp_vec = np.zeros((len(t),max_weight), dtype = complex)
    pm_vec = np.zeros((len(t),max_weight), dtype = complex)

    # set initial values for all operators
    p_vec[0] = np.array([ val_mu_X(N,n, 1) for n in range(max_weight) ])
    m_vec[0] = np.array([ val_mu_X(N,n,-1) for n in range(max_weight) ])
    pp_vec[0] = np.array([ val_mu_nu_X(N,n,1, 1) for n in range(max_weight) ])
    pm_vec[0] = np.array([ val_mu_nu_X(N,n,1,-1) for n in range(max_weight) ])

    # construct differential operators for p- and m-type operators
    diff_op_p_im = ( sparse.diags(np.ones(max_weight))
                      + 2 * sparse.diags(np.ones(max_weight-1),1) )
    diff_op_m_im = ( sparse.diags(np.ones(max_weight))
                      - 2 * sparse.diags(np.ones(max_weight-1),1) )

    diff_op_p_re = -1/2 * sparse.diags(np.ones(max_weight)).todok()
    diff_op_m_re = -1/2 * sparse.diags(np.ones(max_weight)).todok()
    for n in range(max_weight):
        for k in range(n):
            coefficient = (-1)**(n-k) * binom(n,k)
            diff_op_p_re[n,k] += N/2 * coefficient
            diff_op_m_re[n,k] += (N/2-1) * coefficient
            diff_op_p_re[n,k+1] += coefficient
            diff_op_m_re[n,k+1] += coefficient

    diff_op_p = 1j * diff_op_p_im + g * diff_op_p_re
    diff_op_m = 1j * diff_op_m_im + g * diff_op_m_re

    # construct differential operator for pp- and pm-family operators

    diff_op_pp_im = 4 * ( sparse.diags(np.ones(max_weight))
                             + sparse.diags(np.ones(max_weight-1),1) )

    diff_op_pp_re = diff_op_p_re - 1/2 * sparse.diags(np.ones(max_weight))
    diff_op_pm_re = diff_op_m_re - 1/2 * sparse.diags(np.ones(max_weight))

    diff_op_pp = 1j * diff_op_pp_im + g * diff_op_pp_re
    diff_op_pm = g * diff_op_pm_re

    # simulate!
    for ii in range(1,len(t)):
        if print_updates:
            print(f"{ii}/{len(t)}")
        dt = t[ii] - t[ii-1]
        p_vec[ii] = sparse.linalg.expm_multiply(dt * diff_op_p, p_vec[ii-1])
        m_vec[ii] = sparse.linalg.expm_multiply(dt * diff_op_m, m_vec[ii-1])
        pp_vec[ii] = sparse.linalg.expm_multiply(dt * diff_op_pp, pp_vec[ii-1])
        pm_vec[ii] = sparse.linalg.expm_multiply(dt * diff_op_pm, pm_vec[ii-1])

    return Sz, Sz_Sz, p_vec[:,0], p_vec[:,1], m_vec[:,1], pp_vec[:,0], pm_vec[:,0]
