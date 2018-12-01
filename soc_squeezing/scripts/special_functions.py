#!/usr/bin/env python3

# FILE CONTENTS: special functions used in other files

import numpy as np

import scipy.special
from sympy.functions.combinatorial.numbers import stirling as sympy_stirling

# factorial and its logarithm
def factorial(nn, vals = {}):
    if vals.get(nn) == None:
        vals[nn] = scipy.special.factorial(nn, exact = (type(nn) == int))
    return vals[nn]
def ln_factorial(nn, vals = {}):
    if vals.get(nn) == None:
        vals[nn] = scipy.special.gammaln(nn+1)
    return vals[nn]

# falling factorial and its logarithm
def poch(nn, kk, vals = {}):
    if vals.get((nn,kk)) == None:
        vals[nn,kk] = np.prod([ nn-cc for cc in range(kk) ], dtype = int)
    return vals[nn,kk]
def ln_poch(nn, kk, vals = {}):
    if vals.get((nn,kk)) == None:
        vals[nn,kk] = np.sum([ np.log(nn-cc) for cc in range(kk) ])
    return vals[nn,kk]

# binomial coefficient and its logarithm
def binom(nn, kk, vals = {}):
    if vals.get((nn,kk)) == None:
        integer_inputs = ( type(nn) is int ) and ( type(kk) is int )
        vals[nn,kk] = scipy.special.comb(nn, kk, exact = integer_inputs)
    return vals[nn,kk]

# unsigned stirling number of the first kind
def stirling(nn, kk, vals = {}):
    if vals.get((nn,kk)) == None:
        vals[nn,kk] = int(sympy_stirling(nn, kk, kind = 1, signed = False))
    return vals[nn,kk]

# coefficient for computing a product of spin operators
def zeta(mm, nn, pp, qq, vals = {}):
    if vals.get((mm,nn,pp,qq)) == None:
        val = (-1)**pp * 2**qq * np.sum([ stirling(pp,ss) * binom(ss,qq)
                                          * (mm+nn-2*pp)**(ss-qq)
                                          for ss in range(qq,pp+1) ], dtype = int)
        vals[mm,nn,pp,qq] = val
        vals[nn,mm,pp,qq] = val
    return vals[mm,nn,pp,qq]
