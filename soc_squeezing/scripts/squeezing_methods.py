#!/usr/bin/env python3

import numpy as np
import scipy.sparse as sparse
import scipy.linalg as linalg
import scipy.optimize as optimize
import matplotlib.pyplot as plt

from scipy.special import binom as binomial
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors

np.set_printoptions(linewidth = 200)

##########################################################################################
# collective spin vectors in Dicke basis
##########################################################################################

def spin_op_z(N):
    Sz = sparse.lil_matrix((N+1,N+1))
    Sz.setdiag(np.arange(N+1)-N/2)
    return Sz.tocsr()

def spin_op_m(N):
    S = N/2
    m_z = np.arange(N) - N/2
    Sm = sparse.lil_matrix((N+1,N+1))
    Sm.setdiag(np.sqrt((S-m_z)*(S+m_z+1)),1)
    return Sm.tocsr()

def spin_op_x(N):
    Sp = spin_op_m(N)
    return ( Sm + Sm.transpose() ) / 2

def spin_op_y(N):
    Sp = spin_op_p(N)
    return ( Sm - Sm.transpose() ) * 1j / 2

def spin_op_vec_mat(N):
    Sz = spin_op_z(N)
    Sm = spin_op_m(N)
    Sx = ( Sm + Sm.transpose() ) / 2
    Sy = ( Sm - Sm.transpose() ) * 1j / 2

    Szz = sparse.csr_matrix.dot(Sz, Sz)
    Sxx = sparse.csr_matrix.dot(Sx, Sx)
    Syy = sparse.csr_matrix.dot(Sy, Sy)
    Sxy = sparse.csr_matrix.dot(Sx, Sy)
    Syz = sparse.csr_matrix.dot(Sy, Sz)
    Szx = sparse.csr_matrix.dot(Sz, Sx)

    S_op_vec = [ Sz, Sx, Sy ]
    SS_op_mat = [ [ Szz,        Szx,        Syz.getH() ],
                  [ Szx.getH(), Sxx,        Sxy        ],
                  [ Syz,        Sxy.getH(), Syy        ] ]

    return S_op_vec, SS_op_mat


##########################################################################################
# coherent states and squeezing

# expectation value with sparse matrices
def val(X, state):
    if X.shape != state.shape: # we have a state vector
        return state.conj().T.dot(X.dot(state)).sum()
    else: # we have a density operator
        return X.multiply(state).sum()

# get polar and azimulthal angles of a vector (v_z, v_x, v_y)
def vec_theta_phi(v):
    return np.arccos(v[0]/linalg.norm(v)), np.arctan2(v[2],v[1])

# rotate a vector about an axis by a given angle
def rotate_vector(vector, axis, angle):
    g_z = np.array([ [  0,  0,  0 ],
                     [  0,  0, -1 ],
                     [  0,  1,  0 ] ])
    g_x = np.array([ [  0,  0,  1 ],
                     [  0,  0,  0 ],
                     [ -1,  0,  0 ] ])
    g_y = np.array([ [  0, -1,  0 ],
                     [  1,  0,  0 ],
                     [  0,  0,  0 ] ])
    g = ( axis[0] * g_z + axis[1] * g_x + axis[2] * g_y)  / np.linalg.norm(axis)
    return linalg.expm(angle * g) @ vector

# coherent spin state on S = N/2 Bloch sphere
def coherent_spin_state_angles(theta, phi, N = 10):
    state = np.zeros(N+1, dtype = complex)
    for m in range(N+1):
        c_theta = np.sin(theta/2)**(N-m) * np.cos(theta/2)**m
        c_phi = np.exp(1j * (N/2-m) * phi)
        state[m] = np.sqrt(binomial(N,m)) * c_theta * c_phi
    return state
def coherent_spin_state(vec, N = 10):
    theta, phi = vec_theta_phi(vec)
    return coherent_spin_state_angles(theta, phi, N)

# take expectation values of spin operator and matrix
def spin_vec_mat_vals(state, S_op_vec, SS_op_mat):
    S_vec = np.array([ np.real(val(X,state)) for X in S_op_vec ])
    SS_mat = np.array([ [ np.real(val(X,state)) for X in XS ] for XS in SS_op_mat ])
    return S_vec, SS_mat

# variance of spin state about an axis
def spin_variance(axis, S_vec, SS_mat, state = None):
    if state != None: S_vec, SS_mat = spin_vec_mat_vals(state, S_vec, SS_mat)
    return np.real(axis @ SS_mat @ axis - abs(S_vec @ axis)**2)

# return (\xi^2, axis), where:
#   "axis" is the axis of minimal spin variance in the plane orthogonal to <\vec S>
#   \xi^2 = (<S_axis^2> - <S_axis>^2) / (S/2) is the spin squeezing parameter
def spin_squeezing(state, S_op_vec, SS_op_mat, N):
    S_vec, SS_mat = spin_vec_mat_vals(state, S_op_vec, SS_op_mat)

    if S_vec[1] == 0 and S_vec[2] == 0:
        perp_vec = [0,1,0]
    else:
        rot_axis = np.cross([1,0,0], S_vec)
        perp_vec = rotate_vector(S_vec, rot_axis, np.pi/2) / linalg.norm(S_vec)

    def squeezing_axis(eta): # eta = angle in plane orthogonal to the spin vector
        return rotate_vector(perp_vec, S_vec, eta)
    def variance(eta):
        axis = squeezing_axis(eta)
        return np.real(axis @ SS_mat @ axis)

    optimum = optimize.minimize_scalar(variance, method = "bounded", bounds = (0, np.pi))
    if not optimum.success: sys.exit("squeezing optimization failed")

    optimal_phi = optimum.x
    minimal_variance = optimum.fun
    squeezing_parameter = minimal_variance * N / linalg.norm(S_vec)**2

    return squeezing_parameter, squeezing_axis(optimal_phi)

# time evolution of a state
def evolve(state, time, hamiltonian):
    new_state = sparse.linalg.expm_multiply(-1j * time * hamiltonian, state)
    if state.shape == hamiltonian.shape: # i.e. if the state is a density operator
        new_state = sparse.linalg.expm_multiply(-1j * time * hamiltonian,
                                                new_state.conj().T).conj().T
    return new_state

# squeezing parameter after orthogonal-state one-axis twisting
def squeezing_OAT(chi_t, N):
    A = 1 - np.cos(2*chi_t)**(N-2)
    B = 4 * np.sin(chi_t) * np.cos(chi_t)**(N-2)
    mag = np.cos(chi_t)**(N-1)
    return ( (1 + 1/4*(N-1)*A) - 1/4*(N-1)*np.sqrt(A*A+B*B) ) / mag**2


##########################################################################################
# plotting a state on the S = N/2 Bloch sphere
##########################################################################################

def plot_state(state, grid_size = 51, single_sphere = False):
    N = state.size-1
    if single_sphere:
        fig = plt.figure(figsize=plt.figaspect(1))
    else:
        fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = Axes3D(fig)

    theta, phi = np.mgrid[0:np.pi:(grid_size*1j), 0:2*np.pi:(grid_size*1j)]
    z_vals = np.cos(theta)
    x_vals = np.sin(theta) * np.cos(phi)
    y_vals = np.sin(theta) * np.sin(phi)

    def color_val(theta, phi):
        angle_state = coherent_spin_state_angles(theta, phi, N)
        return abs(angle_state.conjugate() @ state)**2
    color_vals = np.vectorize(color_val)(theta, phi)
    norm = colors.Normalize(vmin = np.min(color_vals),
                            vmax = np.max(color_vals), clip = False)
    color_map = cm.inferno(norm(color_vals))

    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    if single_sphere:
        ax.plot_surface(x_vals, y_vals, z_vals, rstride = 1, cstride = 1,
                        facecolors = color_map, shade = False)
    else:
        ax.plot_surface(x_vals, y_vals-1.5, z_vals, rstride = 1, cstride = 1,
                        facecolors = color_map, shade = False)
        ax.plot_surface(-x_vals, y_vals+1.5, z_vals, rstride = 1, cstride = 1,
                        facecolors = color_map, shade = False)
        ax.set_ylim(-2,2)

    ax.set_axis_off()
    ax.view_init(elev = 0, azim = 0)
    return fig, ax
