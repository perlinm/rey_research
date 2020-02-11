#!/usr/bin/env python3

# FILE CONTENTS: special functions used in other files

import numpy as np

import scipy
from scipy import linalg

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

# return vector in (z,x,y) format along an axis specified by text
def axis_str(text):
    sign, axis = text
    axis = axis.lower()
    assert(sign in [ "+", "-" ])
    assert(axis in [ "z", "x", "y" ])
    if axis == "z": vec = np.array([ 1, 0, 0 ])
    if axis == "x": vec = np.array([ 0, 1, 0 ])
    if axis == "y": vec = np.array([ 0, 0, 1 ])
    if sign == "-": vec = -vec
    return vec

# get polar and azimulthal angles of a vector (v_z, v_x, v_y)
def vec_theta_phi(vec):
    return np.array([ np.arccos(vec[0]/scipy.linalg.norm(vec)),
                      np.arctan2(vec[2],vec[1]) ])

# trig functions that check for "special" values
def my_cos(phi):
    if phi == 0 or abs(phi) == 2*np.pi: return 1
    if abs(phi) in [ np.pi/2, 3*np.pi/2 ]: return 0
    if abs(phi) == np.pi: return -1
    return np.cos(phi)
def my_sin(phi):
    if phi == 0 or abs(phi) in [ np.pi, 2*np.pi ]: return 0
    if abs(phi) == np.pi/2: return np.sign(phi)
    if abs(phi) == 3*np.pi/2: return -np.sign(phi)
    return np.sin(phi)
def my_expi(phi):
    return my_cos(phi) + 1j*my_sin(phi)
