#!/usr/bin/env python3

# import re
from mpmath import factorial
from itertools import permutations

from sympy import *
init_printing()

from qubit_methods import fixed_spin_projector
from interaction_sym_methods import H_2_1, H_3_2, H_3_3, H_4_3, \
    c_vec_g, c_vec_D, c_vec_X, sorted_eigenvalues, sorted_coefficients

# construct actual hamiltonians from the coupling constants
construct_actual_hamiltonians = True

print_coefficients = True
print_eigenvalues = False
latexify_output = True

# define two-body coupling constants
g, D, X, e = symbols("g D X e")
couplings = [ g, D-X, D+X, e ]

# define printing method, depending on whether we wish to latexify output
if not latexify_output:
    def my_print(expression):
        print(expression)
else:
    def my_print(expression):
        string = str(expression)
        string = string.replace(str(g),"G_\g")
        string = string.replace(str(D),"G_+")
        string = string.replace(str(X),"G_-")
        string = string.replace("**","^")
        string = string.replace("*"," ")
        string = string.replace("(","\p{")
        string = string.replace(")","}")
        print(string)

if construct_actual_hamiltonians:
    print("constructing actual hamiltonians")

    int_H_2_1 = H_2_1(couplings)
    int_H_3_2 = H_3_2(couplings)
    int_H_3_3_S, int_H_3_3_O = H_3_3(couplings)
    int_H_4_3_B, int_H_4_3_C = H_4_3(couplings)

else: # if not construct_actual_hamiltonians
    print("constructing effective hamiltonians")

    U_2_G = g/2
    U_2_D = D
    U_2_X = X

    U_3_2_G = g**2
    U_3_2_D = D*(D + 2*g)
    U_3_2_X = X*(2*D + X + 2*g)

    U_3_3_S_G = 2*g**3
    U_3_3_S_D = D**3 + 4*D**2*g + D*X**2 + D*g**2 + X**3 + X**2*g
    U_3_3_S_X = X*(3*D**2 + 2*D*X + 8*D*g + 3*X*g + g**2)

    U_3_3_O_G = g**3
    U_3_3_O_D = D**3 + D**2*g + D*X**2 + D*g**2 + X**2*g
    U_3_3_O_X = X*(3*D**2 + 2*D*X + 2*D*g + X**2 + g**2)

    U_4_3_B_G = g**3
    U_4_3_B_D = 2*D*g*(D + g)
    U_4_3_B_X = 2*X*g*(2*D + X + g)

    U_4_3_C_G = 2*g**3
    U_4_3_C_D = D*(D**2 + 2*D*g + 5*g**2)
    U_4_3_C_X = X*(3*D**2 + 3*D*X + 4*D*g + X**2 + 2*X*g + 5*g**2)

    # now that we have determined our M-body coefficients, construct hamiltonians

    int_H_2_1 = zeros(4,4)
    for permutation in permutations(range(2)):
        int_H_2_1 += U_2_G * c_vec_g(permutation)
        int_H_2_1 += U_2_D * c_vec_D(permutation)
        int_H_2_1 += U_2_X * c_vec_X(permutation)

    int_H_3_2 = zeros(8,8)
    for permutation in permutations(range(3)):
        int_H_3_2 += U_3_2_G * c_vec_g(permutation)
        int_H_3_2 += U_3_2_D * c_vec_D(permutation)
        int_H_3_2 += U_3_2_X * c_vec_X(permutation)

    int_H_3_3_S, int_H_3_3_O = zeros(8,8), zeros(8,8)
    for permutation in permutations(range(3)):
        int_H_3_3_S += U_3_3_S_G * c_vec_g(permutation)
        int_H_3_3_S += U_3_3_S_D * c_vec_D(permutation)
        int_H_3_3_S += U_3_3_S_X * c_vec_X(permutation)

        int_H_3_3_O += U_3_3_O_G * c_vec_g(permutation)
        int_H_3_3_O += U_3_3_O_D * c_vec_D(permutation)
        int_H_3_3_O += U_3_3_O_X * c_vec_X(permutation)

    int_H_4_3_B, int_H_4_3_C = zeros(16,16), zeros(16,16)
    for permutation in permutations(range(4)):
        int_H_4_3_B += U_4_3_B_G * c_vec_g(permutation)
        int_H_4_3_B += U_4_3_B_D * c_vec_D(permutation)
        int_H_4_3_B += U_4_3_B_X * c_vec_X(permutation)

        int_H_4_3_C += U_4_3_C_G * c_vec_g(permutation)
        int_H_4_3_C += U_4_3_C_D * c_vec_D(permutation)
        int_H_4_3_C += U_4_3_C_X * c_vec_X(permutation)


# print stuff!
for H, tag in [ [ int_H_2_1, "H_2_1" ],
                [ int_H_3_2, "H_3_2" ],
                [ int_H_3_3_S, "H_3_3_S" ],
                [ int_H_3_3_O, "H_3_3_O" ],
                [ int_H_4_3_B, "H_4_3_B" ],
                [ int_H_4_3_C, "H_4_3_C" ] ]:
    print("-"*80)
    print(tag)
    if print_coefficients:
        print()
        for coefficient in sorted_coefficients(H):
            my_print(coefficient)
    if print_eigenvalues:
        print()
        for eigenvalue in sorted_eigenvalues(H):
            my_print(eigenvalue)
