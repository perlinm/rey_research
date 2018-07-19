#!/usr/bin/env python3

import numpy as np
import scipy.sparse as sparse
import scipy.linalg as linalg
import scipy.optimize as optimize
import matplotlib.pyplot as plt

from squeezing_methods import spin_vec_mat_vals

from scipy.special import binom as binomial

# spin operators for N particles in the S = N/2 Dicke manifold
def spin_op_z_dicke(N):
    Sz = sparse.lil_matrix((N+1,N+1))
    Sz.setdiag(np.arange(N+1)-N/2)
    return Sz.tocsr()

def spin_op_m_dicke(N):
    S = N/2
    m_z = np.arange(N) - N/2
    Sm = sparse.lil_matrix((N+1,N+1))
    Sm.setdiag(np.sqrt((S-m_z)*(S+m_z+1)),1)
    return Sm.tocsr()

def spin_op_x_dicke(N):
    Sp = spin_op_m_dicke(N)
    return ( Sm + Sm.T ) / 2

def spin_op_y_dicke(N):
    Sp = spin_op_m_dicke(N)
    return ( Sm - Sm.T ) * 1j / 2

def spin_op_vec_mat_dicke(N):
    Sz = spin_op_z_dicke(N)
    Sm = spin_op_m_dicke(N)
    Sx = ( Sm + Sm.T ) / 2
    Sy = ( Sm - Sm.T ) * 1j / 2

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

# get polar and azimulthal angles of a vector (v_z, v_x, v_y)
def vec_theta_phi(v):
    return np.arccos(v[0]/linalg.norm(v)), np.arctan2(v[2],v[1])

# use stirling's approximation to compute ln(n!)
def ln_factorial(n):
    if n == 0: return 0
    return ( n * np.log(n) - n + 1/2 * np.log(np.pi) +
             1/6 * np.log( 8*n**3 + 4*n**2 + n + 1/30 ) )

# return logarithm of binomial coefficient, using an approximation if necessary
def ln_binomial(N,m):
    binomial_coeff = binomial(N,m)
    if binomial_coeff != np.inf: return np.log(binomial_coeff)
    else: return ln_factorial(N) - ln_factorial(m) - ln_factorial(N-m)

# coherent spin state on S = N/2 Bloch sphere
def coherent_spin_state_angles(theta, phi, N = 10):
    state = np.zeros(N+1, dtype = complex)
    for m in range(N+1):
        ln_c_theta = (N-m) * np.log(np.sin(theta/2)) + m * np.log(np.cos(theta/2))
        ln_c_phi = 1j * (N/2-m) * phi
        ln_state_m = 1/2 * ln_binomial(N,m) + ln_c_theta + ln_c_phi
        state[m] = np.exp(ln_state_m)
    return state
def coherent_spin_state(vec, N = 10):
    theta, phi = vec_theta_phi(vec)
    return coherent_spin_state_angles(theta, phi, N)

# squeezing parameter after orthogonal-state one-axis twisting
def squeezing_OAT(chi_t, N):
    A = 1 - np.cos(2*chi_t)**(N-2)
    B = 4 * np.sin(chi_t) * np.cos(chi_t)**(N-2)
    mag = np.cos(chi_t)**(N-1)
    return ( (1 + 1/4*(N-1)*A) - 1/4*(N-1)*np.sqrt(A*A+B*B) ) / mag**2

# plot a state on the S = N/2 Bloch sphere
def plot_dicke_state(state, grid_size = 51, single_sphere = False):
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
