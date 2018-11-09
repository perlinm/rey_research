#!/usr/bin/env python3

# FILE CONTENTS: special functions used in other files

import numpy as np

from scipy.special import gamma, gammaln
from scipy.special import binom as scipy_binom
from sympy.functions.combinatorial.numbers import stirling as sympy_stirling

# factorial and its logarithm
def factorial(nn, vals = {}):
    try: return vals[nn]
    except: None
    vals[nn] = gamma(nn+1)
    return vals[nn]
def ln_factorial(nn, vals = {}):
    try: return vals[nn]
    except: None
    vals[nn] = gammaln(nn+1)
    return vals[nn]

# falling factorial and its logarithm
def poch(nn, kk, vals = {}):
    try: return vals[nn,kk]
    except: None
    vals[nn,kk] = np.prod([ nn-cc for cc in range(kk) ])
    return vals[nn,kk]
def ln_poch(nn, kk, vals = {}):
    try: return vals[nn,kk]
    except: None
    vals[nn,kk] = np.sum([ np.log(nn-cc) for cc in range(kk) ])
    return vals[nn,kk]

# binomial coefficient and its logarithm
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
