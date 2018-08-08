#!/usr/bin/env python3

# FILE CONTENTS: model-independent methods for computing spin squeezing

import numpy as np
import scipy.sparse as sparse
import scipy.linalg as linalg
import scipy.optimize as optimize

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

# take expectation values of spin operator and matrix
def spin_vec_mat_vals(state, S_op_vec, SS_op_mat):
    S_vec = np.array([ np.real(val(X,state)) for X in S_op_vec ])
    SS_mat = np.array([ [ np.real(val(X,state)) for X in XS ] for XS in SS_op_mat ])
    return S_vec, SS_mat

# variance of spin state about an axis
def spin_variance(axis, S_vec, SS_mat, state = None):
    axis = np.array(axis)
    # if we were given a state, then S_vec and SS_mat are operator-valued,
    #   so we need to compute their expectation values
    if type(state) != type(None):
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
def spin_squeezing(state, S_op_vec, SS_op_mat, N, axis = None):
    S_vec, SS_mat = spin_vec_mat_vals(state, S_op_vec, SS_op_mat)
    if type(axis) == type(None):
        variance = minimal_orthogonal_variance(S_vec, SS_mat)
    else:
        axis = np.array(axis)
        variance = np.real(axis @ SS_mat @ axis) / (axis @ axis)
    return variance * N / linalg.norm(S_vec)**2

# time evolution of a state with a sparse hamiltonian
def evolve(state, hamiltonian, time):
    assert(sparse.issparse(hamiltonian))
    new_state = sparse.linalg.expm_multiply(-1j * time * hamiltonian, state)
    if state.shape == hamiltonian.shape: # the state is a density operator
        new_state = sparse.linalg.expm_multiply(-1j * time * hamiltonian,
                                                new_state.conj().T).conj().T
    return new_state
