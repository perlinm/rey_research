#!/usr/bin/env python3

# FILE CONTENTS: (symbolic) methods for performing qubit operations

import sympy as sym
from itertools import product as cartesian_product
from itertools import combinations, permutations
from sympy.physics.quantum import TensorProduct as tensor

# single-atom pseudospin states
dn = sym.Matrix([1,0])
up = sym.Matrix([0,1])

# two-atom pseudospin states
uu = tensor(up,up)
ud = tensor(up,dn)
du = tensor(dn,up)
dd = tensor(dn,dn)

# all states of n qubits
def qubit_states(n):
    return cartesian_product([-1,1], repeat = n)

# single-qubit matrix entry: | final >< initial |
def qubit_matrix_entry(initial, final):
    state_in = (up if initial == 1 else dn)
    state_out = (up if final == 1 else dn)
    return tensor(state_out, state_in.H)

# generate a matrix which acts A on the qubits indixed by ns (out of N qubits total)
def act(A, N, target_qubits):
    ancilla_qubits = [ n for n in range(N) if n not in target_qubits ]
    D = 2**N
    B = sym.zeros(D)
    for total_input_index in range(D):
        total_input = bin(total_input_index)[2:].zfill(N)
        ancilla_input = [ total_input[n] for n in ancilla_qubits ]
        target_input = "".join([ total_input[n] for n in target_qubits ])
        target_input_index = int(target_input,2)

        for total_output_index in range(D):
            state_out = bin(total_output_index)[2:].zfill(N)
            ancilla_output = [ state_out[n] for n in ancilla_qubits ]

            if ancilla_output == ancilla_input:
                target_output = "".join([ state_out[n] for n in target_qubits ])
                target_output_index = int(target_output,2)
                B[total_output_index, total_input_index] = A[target_output_index,
                                                             target_input_index]
    return B

# act an operator on all appropriate subsets of N qubits
def act_on_subsets(mat, N):
    qubits = sym.simplify(sym.log(mat.cols)/sym.log(2))

    if qubits == N: return mat
    if qubits > N: return sym.zeros(2**N)

    total_mat = sym.zeros(2**N)
    for targets in combinations(range(N), qubits):
        total_mat += act(mat, N, targets)
    return total_mat

# act operator on all permutations of its qubits
def act_on_permutations(mat):
    qubits = sym.simplify(sym.log(mat.cols)/sym.log(2))
    mat_out = sym.zeros(mat.rows)
    for permutation in permutations(range(qubits)):
        mat_out += act(mat, qubits, permutation)
    return mat_out

# return projector onto the subspace of states with
#   a definite number of qubits in state "1"
def fixed_spin_projector(qubits_up, total_qubits):
    projector = sym.zeros(2**total_qubits)
    for permutation in set(permutations([ 1 for ii in range(qubits_up) ] +
                                        [ 0 for jj in range(total_qubits-qubits_up) ])):
        state = sym.zeros(2**total_qubits,1)
        unit_index = sum( 2**ii for ii in range(total_qubits) if permutation[ii] == 1 )
        state[unit_index] = 1
        projector += state * state.H
    return projector

# projector onto the fully symmetric subspace of a given number of qubits
def symmetric_projector(qubits):
    projector = sym.zeros(2**qubits)
    projector[0,0] = 1
    projector[-1,-1] = 1
    for n in range(1,qubits):
        symmetric_state = sym.zeros(2**qubits,1)
        for permutation in set(permutations([ 1 for ii in range(n) ] +
                                            [ 0 for jj in range(qubits-n) ])):
            unit_index = sum( 2**ii for ii in range(qubits) if permutation[ii] == 1 )
            symmetric_state[unit_index] = 1
        projector += symmetric_state * symmetric_state.H / sum(symmetric_state)
    return projector

##########################################################################################
# printing methods
##########################################################################################

# print a multi-qubit state in human-readable form
def qvec_print(v):
    N = len(v)
    qbits = int(sym.log(N)/sym.log(2))
    for n in range(N):
        if v[n] != 0:
            s = bin(n)[2:].zfill(qbits)
            s = s.replace("0","d").replace("1","u")
            print("%s:"%s,v[n])

# print eigenvalues and eigenvectors of an operator
def print_eigensystem(M, print_text = None):
    print("----------------------------------------")
    print("----------------------------------------")
    if print_text == None:
        print(M)
    else:
        print(print_text)
    dim = M.rows
    atom_number = sym.simplify(sym.log(dim)/sym.log(2))
    for n in range(atom_number+1):
        spin_projector = fixed_spin_projector(n,atom_number)
        inds = [ n for n in range(dim) if spin_projector[n,n] == 1 ]
        for val, num, vecs in M[inds,inds].eigenvects():
            if val == 0: continue
            val = sym.simplify(val)
            print("----------------------------------------")
            print("----------------------------------------")
            print(sym.factor(val))
            for vec in vecs:
                print("--------------------")
                full_vec = sym.zeros(dim,1)
                for ii in range(len(vec)):
                    full_vec[inds[ii]] = sym.simplify(vec[ii])
                qvec_print(full_vec)
