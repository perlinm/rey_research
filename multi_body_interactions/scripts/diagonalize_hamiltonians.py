#!/usr/bin/env python3

import sys
from sympy import *
init_printing()

from qubit_methods import fixed_spin_projector, act_on_subsets
from qubit_methods import qvec_print, print_eigensystem
from interaction_sym_methods import H_2_1, H_3_2, H_3_3, H_4_3

# process inputs
if len(sys.argv) != 2:
    print("usage: {} atom_number".format(sys.argv[0]))
    exit()

atom_number = int(sys.argv[1])
assert(atom_number in [ 2, 3, 4 ])

# coupling constants
g, D, X, e = symbols("g D X e")
couplings = [ g, D-X, D+X, e ]

if atom_number == 2:

    print("----------------------------------------------------------------------")
    print("first order two-body Hamiltonian")
    print_eigensystem(act_on_subsets(H_2_1(couplings),atom_number))

if atom_number == 3:

    print("----------------------------------------------------------------------")
    print("second order three-body Hamiltonian")
    print_eigensystem(H_3_2(couplings))

    H_3_3_S, H_3_3_O = H_3_3(couplings)

    print("----------------------------------------------------------------------")
    print("third order three-body 'star' Hamiltonian")
    print_eigensystem(H_3_3_S)

    print("----------------------------------------------------------------------")
    print("third order three-body 'O' Hamiltonian")
    print_eigensystem(H_3_3_O)

if atom_number == 4:

    H_4_3_B, H_4_3_C = H_4_3(couplings)

    print("----------------------------------------------------------------------")
    print("third order four-body 'branch' Hamiltonian")
    print_eigensystem(H_4_3_B)

    print("----------------------------------------------------------------------")
    print("third order four-body 'chain' Hamiltonian")
    print_eigensystem(H_4_3_C)

