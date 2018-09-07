#!/usr/bin/env python3

import sys, scipy
import numpy as np
import matplotlib.pyplot as plt

import itertools
from scipy.special import factorial, binom
from sympy.functions.combinatorial.numbers import stirling as sympy_stirling

from dicke_methods import *

from squeezing_methods import val

N = 50
S = N/2

def stirling(n,k):
    return float(sympy_stirling(n, k, kind = 1, signed = False))

II = sparse.identity(N+1)
ZZ = sparse.dok_matrix((N+1,N+1))

Sz = spin_op_z_dicke(N)
Sp = spin_op_p_dicke(N)
Sm = spin_op_m_dicke(N)

Sz_r = [ II ]
Sp_r = [ II ]
Sm_r = [ II ]
for n in range(N):
    Sz_r.append(Sz_r[n].dot(Sz))
    Sp_r.append(Sp_r[n].dot(Sp))
    Sm_r.append(Sm_r[n].dot(Sm))


def S_ops(mu):
    if mu == 1: return Sp_r, Sm_r
    else: return Sm_r, Sp_r

def eps(m,n,p,l):
    num = np.sum([ (-1)**(p-qq) * stirling(p,qq) * binom(qq,l) * (m-n)**(qq-l)
                   for qq in range(l,p+1) ])
    return 2**l * num

def xi(m,n,p,q):
    return sum([ (-1)**(ll-q) * eps(m,n,p,ll) * binom(ll,q) * (m-p)**(ll-q)
                 for ll in range(q,p+1) ])

def tZ(mu,q,r,a,b,k):
    op = ZZ.copy()
    for ll in range(k+1):
        ll_fac = (-1)**ll * xi(q,r,k,ll)
        for mm in range(a+1):
            lm_fac = ll_fac * (r-k)**(a-mm) * binom(a,mm)
            for nn in range(b+1):
                lmn_fac = lm_fac * (q-k)**(b-nn) * binom(b,nn)
                op += lmn_fac * mu**(a+b+ll-mm-nn) * Sz_r[ll+mm+nn]
    return op

def S(mu,p,a,q,r,b,s):
    S_mu_r, S_nu_r = S_ops(mu)
    op = ZZ.copy()
    for kk in range(min(q,r)+1):
        k_fac = factorial(kk) * binom(q,kk) * binom(r,kk)
        op += k_fac * S_mu_r[p+r-kk].dot(tZ(mu,q,r,a,b,kk)).dot(S_nu_r[q+s-kk])
    return op

def C(mu,p,a,q,r,b,s):
    return S(mu,p,a,q,r,b,s) - S(mu,r,b,s,p,a,q)

def C_full(mu,p,a,q,r,b,s):
    S_mu_r, S_nu_r = S_ops(mu)
    A = S_mu_r[p].dot(Sz_r[a]).dot(S_nu_r[q])
    B = S_mu_r[r].dot(Sz_r[b]).dot(S_nu_r[s])
    return A.dot(B) - B.dot(A)

for mu in [ 1, -1 ]:
    S_mu_r, S_nu_r = S_ops(mu)
    for p,q,a,r,s,b in itertools.product(range(1,5),repeat=6):
        print(p,q,a,r,s,b)
        arst = C_full(mu,p,a,q,r,b,s)
        neio = C(mu,p,a,q,r,b,s)
        max_val = abs(arst - neio).max() / (N/2)**(p+q+a+r+s+b)
        if max_val > 1e-10:
            print(max_val)
