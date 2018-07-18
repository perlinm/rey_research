#!/usr/bin/env python3

import numpy as np
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

def spin_op_z(N): return np.diag(np.arange(N+1)-N/2)

def spin_op_m(N):
    S = N/2
    m_z = np.arange(N) - N/2
    return np.diag(np.sqrt((S-m_z)*(S+m_z+1)),1)

def spin_op_p(N): return spin_op_m(N).transpose()

def spin_op_x(N):
    Sp = spin_op_p(N)
    return 1/2 * ( Sp + Sp.transpose() )

def spin_op_y(N):
    Sp = spin_op_p(N)
    return -1j/2 * ( Sp - Sp.transpose() )

def spin_op_vec(N):
    Sz = spin_op_z(N)
    Sp = spin_op_p(N)
    Sx = 1/2 * ( Sp + Sp.transpose() )
    Sy = -1j/2 * ( Sp - Sp.transpose() )
    return np.array([ Sz, Sx, Sy ])

def spin_op_n(vec, N):
    return np.einsum("i,iAB->AB", vec, spin_op_vec(N)) / np.linalg.norm(vec)


##########################################################################################
# coherent states and squeezing
##########################################################################################

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

# get spin vector <\vec S> for a state
def spin_vec(state):
    N = state.size-1
    def val(X): return state.conj() @ X @ state
    return np.real(np.array([ val(spin_op_z(N)), val(spin_op_x(N)), val(spin_op_y(N)) ]))

# get normalized spin vector <\hat S> for a state
def spin_axis(state): return spin_vec(state) / ( (state.size-1) / 2)

# variance of spin state about an axis
def spin_variance(state, axis):
    N = state.size - 1
    Sn = spin_op_n(axis, N)
    def val(X): return state.conj() @ X @ state
    return abs(val(Sn @ Sn) - abs(val(Sn))**2)

# return (\xi^2, axis), where:
#   "axis" is the axis of minimal spin variance in the plane orthogonal to <\vec S>
#   \xi^2 = (<S_axis^2> - <S_axis>^2) / (S/2) is the spin squeezing parameter
def spin_squeezing(state):
    N = state.size-1
    S = N/2
    state_axis = spin_axis(state)
    if state_axis[1] == 0 and state_axis[2] == 0:
        perp_vec = [0,1,0]
    else:
        rot_axis = np.cross([1,0,0], state_axis)
        perp_vec = rotate_vector(state_axis, rot_axis, np.pi/2)

    def squeezing_axis(eta): # eta is the angle in the plane orthogonal to the spin vector
        return rotate_vector(perp_vec, state_axis, eta)
    def variance(eta):
        return spin_variance(state, squeezing_axis(eta))

    optimum = optimize.minimize_scalar(variance, method = "bounded", bounds = (0, np.pi))
    if not optimum.success:
        print("squeezing optimization failed")
        exit()
    optimal_phi = optimum.x
    minimal_variance = optimum.fun

    S_vec = np.einsum("i,Aij,j->A", state.conj(), spin_op_vec(N), state, optimize = True)
    squeezing_parameter = minimal_variance * N / abs(S_vec @ S_vec)

    return squeezing_parameter, squeezing_axis(optimal_phi)

# propagator for z-axis twisting
def squeezing_OAT_propagator(chi_t, N):
    Sz = spin_op_z(N)
    return linalg.expm(-1j * chi_t * Sz @ Sz)

# squeezing parameter after orthogonal-state one-axis twisting
def squeezing_OAT(chi_t, N):
    A = 1 - np.cos(2*chi_t)**(N-2)
    B = 4 * np.sin(chi_t) * np.cos(chi_t)**(N-2)
    mag = np.cos(chi_t)**(N-1)
    return ( (1 + 1/4*(N-1)*A) - 1/4*(N-1)*np.sqrt(A*A+B*B) ) / mag**2

# propagator for z-y two-axis twisting
def squeezing_TAT_propagator(chi_t, N):
    Sz = spin_op_z(N)
    Sy = spin_op_y(N)
    return linalg.expm(-1j * chi_t * ( Sz @ Sz - Sy @ Sy ))

# squeezing parameter after orthogonal-state two-axis twisting
def squeezing_TAT(chi_t, N, return_propagator = False):
    U_zy = squeezing_TAT_propagator(chi_t, N)
    state_x = coherent_spin_state_angles(np.pi/2, 0, N)
    return spin_squeezing(U_zy @ state_x)[0]


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
