#!/usr/bin/env python3

import numpy as np

import itertools
from scipy.special import factorial, binom
from sympy.functions.combinatorial.numbers import stirling as sympy_stirling

from dicke_methods import *

from squeezing_methods import val

N = 50
S = N/2

def stirling(nn,kk):
    return float(sympy_stirling(nn, kk, kind = 1, signed = False))

II = sparse.identity(N+1)
ZZ = sparse.dok_matrix((N+1,N+1))

S_z = spin_op_z_dicke(N)
S_p = spin_op_p_dicke(N)
S_m = spin_op_m_dicke(N)

S_z_r = [ II ]
S_p_r = [ II ]
S_m_r = [ II ]
for n in range(N):
    S_z_r.append(S_z_r[n].dot(S_z))
    S_p_r.append(S_p_r[n].dot(S_p))
    S_m_r.append(S_m_r[n].dot(S_m))


def S_op(mu):
    if mu == 1: return S_p
    else: return S_m

def S_ops(mu):
    return S_op(mu), S_op(-mu)

def res(X,Y):
    diff = X - Y
    return abs(diff).max() / abs(X).max()

def eps(mm,nn,pp,ll):
    num = np.sum([ (-1)**(pp-qq) * stirling(pp,qq) * binom(qq,ll) * (mm-nn)**(qq-ll)
                   for qq in range(ll,pp+1) ])
    return 2**ll * num

def zeta(mm,nn,pp,qq):
    sum_val = np.sum([ stirling(pp,ss) * binom(ss,qq) * (mm+nn-2*pp)**(ss-qq)
                       for ss in range(qq,pp+1) ])
    return (-1)**pp * 2**qq * sum_val

def Z(mm,nn,pp,mu):
    return np.sum([ zeta(mm,nn,pp,qq) * (mu*S_z)**qq
                    for qq in range(pp+1) ])

def tZ(qq,rr,ll,mm,kk,mu):
    op = ZZ.copy()
    aa_facs = [ zeta(rr,ll,kk,aa) for aa in range(kk+1) ]
    bb_facs = [ (ll-kk)**(qq-bb) * binom(qq,bb) for bb in range(qq+1) ]
    cc_facs = [ (rr-kk)**(mm-cc) * binom(mm,cc) for cc in range(mm+1) ]
    for aa in range(kk+1):
        for bb in range(qq+1):
            for cc in range(mm+1):
                op += aa_facs[aa] * bb_facs[bb] * cc_facs[cc] * (mu*S_z)**(aa+bb+cc)
    return op

MN = 10
exps_iter = itertools.product(range(10), repeat = 6)
for exps, mu in itertools.product(exps_iter, (1,-1)):
    pp, qq, rr, ll, mm, nn = exps
    S_mu, S_nu = S_ops(mu)

    arst_1 = S_mu**pp * (mu*S_z)**qq * S_nu**rr
    arst_2 = S_mu**ll * (mu*S_z)**mm * S_nu**nn
    arst = arst_1 * arst_2

    neio = np.sum([ factorial(kk) * binom(rr,kk) * binom(ll,kk)
                    * S_mu**(pp+ll-kk)
                    * tZ(qq,rr,ll,mm,kk,mu)
                    * S_nu**(rr+nn-kk)
                    for kk in range(min(rr,ll)+1) ])

    if res(arst,neio) >= 1e-10:
        print(mm,nn,mu,res(arst,neio))
