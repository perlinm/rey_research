#!/usr/bin/env python3

import numpy as np
import sympy as sym
from sympy.physics.quantum.cg import CG as sym_cg

def cg_coef(l1, m1, l2, m2, L, M):
    return sym_cg(l1, m1, l2, m2, L, M).doit()

# transition and drive operators, exact
def transition_op_exact(dim, L, M):
    L, M = sym.S(L), sym.S(M)
    I = sym.S(dim-1)/2
    mu_min = max(-M,0)
    mu_max = min(dim-M,dim)
    diag_vals = [ cg_coef(I, -I+mu, L, M, I, -I+mu+M) for mu in range(mu_min,mu_max) ]
    return sym.sqrt(sym.S(2*L+1)/sym.S(2*I+1)) * np.diag(diag_vals, -M)

def drive_op_exact(dim, L, M):
    T_LM = transition_op_exact(dim, L, M)
    if M == 0: return T_LM
    eta_M = (-1)**M if M > 0 else sym.I
    return eta_M/sym.sqrt(2) * ( T_LM + np.sign(M) * T_LM.conj().T )

# transition and drive operators, numerical
def transition_op(dim, L, M):
    return transition_op_exact(dim, L, M).astype(float)

def drive_op(dim, L, M):
    dtype = float if M >= 0 else complex
    return drive_op_exact(dim, L, M).astype(dtype)

# drive scale factor
def drive_scale_exact(dim, L):
    L = sym.S(L)
    return sym.sqrt(2*L+1) * sym.factorial(L) / sym.factorial(2*L+1) * \
           sym.sqrt(np.prod([ dim + l for l in range(-L,L+1) ]))

def drive_scale(dim, L):
    return float(drive_scale_exact(dim, L))
