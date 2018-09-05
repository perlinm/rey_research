#!/usr/bin/env python3

# FILE CONTENTS: model-independent methods for computing spin squeezing

import numpy as np
import scipy.sparse as sparse
import scipy.linalg as linalg
import scipy.optimize as optimize

from correlator_methods import correlators_OAT

# constrect vector of generators about z, x, and y
g_vec = np.array([ np.array([ [  0,  0,  0 ],
                              [  0,  0, -1 ],
                              [  0,  1,  0 ] ]),
                   np.array([ [  0,  0,  1 ],
                              [  0,  0,  0 ],
                              [ -1,  0,  0 ] ]),
                   np.array([ [  0, -1,  0 ],
                              [  1,  0,  0 ],
                              [  0,  0,  0 ] ]) ])

# rotate a target vector by an angl about an axis
def rotate_vector(vector, angle, axis):
    rotation_generator = angle * np.einsum("i,iAB->AB", axis/linalg.norm(axis), g_vec)
    return linalg.expm(rotation_generator) @ vector

# expectation value of X with respect to a state
def val(X, state):
    if X.shape != state.shape: # we have a state vector
        return state.conj().T.dot(X.dot(state)).sum()
    else: # we have a density operator
        if sparse.issparse(X):
            return X.T.multiply(state).sum()
        else:
            return (X.T*state).sum()

# expectation value of vector spin operator
def spin_vec_vals(state, S_op_vec):
    return np.array([ val(X,state) for X in S_op_vec ])

# expectation value of matrix spin-spin operator
def spin_mat_vals(state, SS_op_mat):
    SS_mat = np.zeros((3,3), dtype = complex)
    for ii in range(3):
        for jj in range(ii+1):
            SS_mat[ii,jj] = val(SS_op_mat[ii][jj], state)
            SS_mat[jj,ii] = np.conj(SS_mat[ii,jj])
    return SS_mat

# expectation values of spin operator and spin-spin matrix
def spin_vec_mat_vals(state, S_op_vec, SS_op_mat):
    return spin_vec_vals(state, S_op_vec), spin_mat_vals(state, SS_op_mat)

# variance of spin state about an axis
def spin_variance(axis, S_vec, SS_mat, state = None):
    axis = np.array(axis)
    # if we were given a state, then S_vec and SS_mat are operator-valued,
    #   so we need to compute their expectation values
    if state is not None:
        S_vec, SS_mat = spin_vec_mat_vals(state, S_vec, SS_mat)
    return np.real(axis @ SS_mat @ axis - abs(S_vec @ axis)**2) / (axis @ axis)

# return minimum spin variance in the plane orthogonal to the mean spin vector
def minimal_orthogonal_variance(S_vec, SS_mat):
    if S_vec[1] == 0 and S_vec[2] == 0:
        perp_vec = [0,1,0]
    else:
        rot_axis = np.cross([1,0,0], S_vec)
        perp_vec = rotate_vector(S_vec, np.pi/2, rot_axis) / linalg.norm(S_vec)

    def variance(eta): # eta = angle in plane orthogonal to the spin vector
        axis = rotate_vector(perp_vec, eta, S_vec)
        return np.real(axis @ SS_mat @ axis)

    optimum = optimize.minimize_scalar(variance, method = "bounded", bounds = (0, np.pi))
    if not optimum.success: sys.exit("squeezing optimization failed")

    return optimum.fun

# return spin squeezing parameter
def spin_squeezing(spin_num, state, S_op_vec, SS_op_mat, in_dB = True):
    S_vec, SS_mat = spin_vec_mat_vals(state, S_op_vec, SS_op_mat)
    variance = minimal_orthogonal_variance(S_vec, SS_mat)
    squeezing = variance * spin_num / linalg.norm(S_vec)**2
    if in_dB: squeezing = -10*np.log10(squeezing)
    return squeezing

# return  from a set of spin correlators
def squeezing_from_correlators(spin_num, correlators, in_dB = True):
    Sz    = correlators[(0,1,0)]
    Sz_Sz = correlators[(0,2,0)]
    Sp    = correlators[(1,0,0)]
    Sp_Sp = correlators[(2,0,0)]
    Sp_Sz = correlators[(1,1,0)]
    Sp_Sm = correlators[(1,0,1)]

    Sx = np.real(Sp)
    Sy = np.imag(Sp)

    Sm_Sm = np.conj(Sp_Sp)
    Sm_Sp = Sp_Sm - 2*Sz
    Sm_Sz = np.conj(Sp_Sz + Sp)

    Sx_Sz =   1/2 * ( Sp_Sz + Sm_Sz )
    Sy_Sz = -1j/2 * ( Sp_Sz - Sm_Sz )
    Sz_Sx = np.conj(Sx_Sz)
    Sz_Sy = np.conj(Sy_Sz)

    Sx_Sx =   1/4 * ( Sp_Sp + Sm_Sm + Sp_Sm + Sm_Sp )
    Sy_Sy =  -1/4 * ( Sp_Sp + Sm_Sm - Sp_Sm - Sm_Sp )
    Sx_Sy = -1j/4 * ( Sp_Sp - Sm_Sm - Sp_Sm + Sm_Sp )
    Sy_Sx = np.conj(Sx_Sy)

    S_vec = np.array([ Sz, Sx, Sy ]).T
    SS_mat = np.array([ [ Sz_Sz, Sz_Sx, Sz_Sy ],
                        [ Sx_Sz, Sx_Sx, Sx_Sy ],
                        [ Sy_Sz, Sy_Sx, Sy_Sy ] ]).T

    if type(Sz) == np.ndarray:
        var_min = np.array([ minimal_orthogonal_variance(S_vec[ii], SS_mat[ii])
                             for ii in range(len(Sz)) ])
    else:
        var_min = minimal_orthogonal_variance(S_vec, SS_mat)

    squeezing = var_min * spin_num / np.real(Sz*Sz + Sx*Sx + Sy*Sy)
    if in_dB: squeezing = -10*np.log10(squeezing)
    return squeezing

# act with a time-evolution unitary from the left, right, or both sides
def evolve_left(state, hamiltonian, time):
    return sparse.linalg.expm_multiply(-1j * time * hamiltonian, state)
def evolve_right(state, hamiltonian, time):
    return evolve_left(state.conj().T, hamiltonian, time).conj().T
def evolve_left_right(state, hamiltonian, time):
    return evolve_right(evolve_left(state, hamiltonian, time), hamiltonian, time)

# time evolution of a state with a sparse hamiltonian
def evolve(state, hamiltonian, time):
    assert(sparse.issparse(hamiltonian))
    assert(state.shape[0] == hamiltonian.shape[0])

    if state.shape != hamiltonian.shape: # state is a vector
        return evolve_left(state, hamiltonian, time)
    else: # state is a density operator
        return evolve_left_right(state, hamiltonian, time)

# squeezing parameter from one-axis twisting
def squeezing_OAT(spin_num, chi_t, dec_rates = (0,0,0), in_dB = True):

    # if there is no decoherence, we have simple analytical formulas
    if dec_rates is (0,0,0):

        S = spin_num/2
        var_Sy = S/2 + S/2 * (S-1/2) * ( 1 - np.cos(2*chi_t)**(spin_num-2) )
        var_Sz = S/2

        A = var_Sy - var_Sz
        C = var_Sy + var_Sz
        B = 2 * S * (S-1/2) * np.sin(chi_t) * np.cos(chi_t)**(spin_num-2)

        var_min = 1/2 * np.real( C - np.sqrt(A**2 + B**2) )
        Sx = S * np.cos(chi_t)**(spin_num-1)

        squeezing = var_min * spin_num / Sx**2
        if in_dB: squeezing = -10*np.log10(squeezing)
        return squeezing

    # otherwise, use more complex but exact spin correlators to compute squeezing
    correlators = correlators_OAT(spin_num, chi_t, dec_rates)

    return squeezing_from_correlators(spin_num, correlators, in_dB)
