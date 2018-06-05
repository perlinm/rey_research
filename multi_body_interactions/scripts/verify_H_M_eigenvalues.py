#!/usr/bin/env python3

import sys
import numpy as np
import sympy as sp
from itertools import permutations

from qubit_methods import act_on_subsets, print_eigensystem
from interaction_sym_methods import c, c_vec, \
    coefficients_to_eigenvalues, coefficients_to_eigenvalues_sym

N = 5 # for N-body states
M = 2 # for M-body Hamiltonian

# if true, express N-body eigenvalues in terms of M-body eigenvalues
#     else express N-body eigenvalues in terms of M-body coefficients
use_eigenvalues = False

##########################################################################################
# print symbolic eigenvalue tables
##########################################################################################

# collect vectors of coefficients and eigenvalues
U_g = sp.symbols("U_g")
U_X = sp.symbols("U_X")
U_D = sp.symbols("U_D")
E_0 = sp.symbols("E_0")
E_A = sp.symbols("E_A")
E_S = sp.symbols("E_S")
coefficients_sym = sp.Matrix([ U_g, U_X, U_D ])
eigenvalues_sym = sp.Matrix([ E_0, E_A, E_S ])

coefficients_table = coefficients_to_eigenvalues_sym()
eigenvalues_table = sp.simplify(coefficients_to_eigenvalues_sym() @
                                coefficients_to_eigenvalues_sym(True).inv())

print("N-body eigenvalues of M-body Hamiltonian")
print()
print("  in terms of M-body coefficients:")
sp.pprint(coefficients_table)
print()
print("  in terms of M-body eigenvalues:")
sp.pprint(eigenvalues_table)

print()
print("-"*80)
print()

##########################################################################################
# print numeric eigenvectors and eigenvalues (i.e. for fixed N, M)
##########################################################################################

print("{}-body eigenvalues of {}-body Hamiltonian".format(N,M))

if use_eigenvalues:
    coefficients = coefficients_to_eigenvalues(M).inv() @ eigenvalues_sym
else:
    coefficients = coefficients_sym

eigenvalues = coefficients_to_eigenvalues(M,N) @ coefficients
eigenvalues = sp.factor(eigenvalues)

print("ground:",eigenvalues[0])
print("excited, asymmetric:",eigenvalues[1])
print("excited, symmetric:",eigenvalues[2])

print()
print("-"*80)
print()

# construct M-body Hamiltonian
H_M = sp.zeros(2**M)
for mu_vec in permutations(range(M)):
    mu, nu = mu_vec[:2]
    spectators = []
    for rho in mu_vec[2:]:
        spectators += [ c(rho,-1).dag(),  c(rho,-1) ]

    vec = c_vec([ c(mu,-1).dag(), c(mu,-1),
                  c(nu,-1).dag(),  c(nu,-1) ] + spectators)

    H_M += coefficients[0] * c_vec([ c(mu,-1).dag(), c(mu,-1),
                                     c(nu,-1).dag(),  c(nu,-1) ] + spectators).elem()
    H_M += coefficients[1] * c_vec([ c(mu,-1).dag(), c(nu,1).dag(),
                                     c(nu,-1),  c(mu,1) ] + spectators).elem()
    H_M += coefficients[2] * c_vec([ c(mu,1).dag(), c(mu,1),
                                     c(nu,-1).dag(),  c(nu,-1) ] + spectators).elem()

print_eigensystem(act_on_subsets(H_M,N), "atom_number: %i" % N)
