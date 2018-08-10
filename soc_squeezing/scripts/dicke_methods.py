#!/usr/bin/env python3

# FILE CONTENTS: methods for simulating dynamics in the Dicke manifold

import numpy as np
import scipy.sparse as sparse
import scipy.linalg as linalg
import scipy.optimize as optimize
import matplotlib.pyplot as plt

from scipy.special import binom, factorial
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors

from squeezing_methods import minimal_orthogonal_variance

# spin operators for N particles in the S = N/2 Dicke manifold
def spin_op_z_dicke(N):
    return sparse.diags(np.arange(N+1)-N/2, format = "csr")

def spin_op_m_dicke(N):
    S = N/2
    m_z = np.arange(N) - N/2
    diag_vals = np.sqrt((S-m_z)*(S+m_z+1))
    return sparse.diags(diag_vals, 1, format = "csr")

def spin_op_p_dicke(N):
    return spin_op_m_dicke(N).T

def spin_op_x_dicke(N):
    Sm = spin_op_m_dicke(N)
    return ( Sm + Sm.T ) / 2

def spin_op_y_dicke(N):
    Sm = spin_op_m_dicke(N)
    return ( Sm - Sm.T ) * 1j / 2

def spin_op_vec_mat_dicke(N):
    Sz = spin_op_z_dicke(N)
    Sm = spin_op_m_dicke(N)
    Sx = ( Sm + Sm.T ) / 2
    Sy = ( Sm - Sm.T ) * 1j / 2

    Szz = Sz.dot(Sz)
    Sxx = Sx.dot(Sx)
    Syy = Sy.dot(Sy)
    Sxy = Sx.dot(Sy)
    Syz = Sy.dot(Sz)
    Szx = Sz.dot(Sx)

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
def ln_binom(N,m):
    binomial_coeff = binom(N,m)
    if binomial_coeff != np.inf: return np.log(binomial_coeff)
    else: return ln_factorial(N) - ln_factorial(m) - ln_factorial(N-m)

# coherent spin state on S = N/2 Bloch sphere
def coherent_spin_state_angles(theta, phi, N = 10):
    if theta == 0:
        state = np.zeros(N+1, dtype=complex)
        state[-1] = 1
        return state
    if theta == np.pi:
        state = np.zeros(N+1, dtype=complex)
        state[0] = 1
        return state
    theta -= int(theta/np.pi) * np.pi
    return np.exp(np.array([ 1/2 * ln_binom(N,m)
                             + (N-m) * np.log(np.sin(theta/2))
                             + m * np.log(np.cos(theta/2))
                             + 1j * (N/2-m) * phi
                             for m in range(N+1) ]))
def coherent_spin_state(vec, N = 10):
    theta, phi = vec_theta_phi(vec)
    return coherent_spin_state_angles(theta, phi, N)


##########################################################################################
# plotting states on the S = N/2 Bloch sphere
##########################################################################################

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
    color_map = plt.cm.inferno(norm(color_vals))

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


##########################################################################################
# analytical solutions to special cases
##########################################################################################

# squeezing parameter from one-axis twisting, accounting for e --> g decay
def squeezing_OAT(chi_t, N, decay_rate_over_chi = 0):
    g = decay_rate_over_chi # shorthand for decay rate in units with \chi = 1
    t = chi_t # shorthand for time in units with \chi = 1

    # if there is no spin decay, we have simple analytical formulas
    if g == 0:

        S = N/2
        var_Sy = S/2 + S/2 * (S-1/2) * ( 1 - np.cos(2*chi_t)**(N-2) )
        var_Sz = S/2

        A = var_Sy - var_Sz
        C = var_Sy + var_Sz
        B = 2 * S * (S-1/2) * np.sin(chi_t) * np.cos(chi_t)**(N-2)

        var_min = 1/2 * np.real( C - np.sqrt(A**2 + B**2) )
        Sx = S * np.cos(chi_t)**(N-1)

        return var_min * N / Sx**2

    # otherwise, we compute the spin correlators derived in foss-feig2013nonequilibrium

    def s(J): return J + 1j*g/2
    def Phi(J): return np.exp(-g*t/2) * ( np.cos(s(J)*t) + g*t/2 * np.sinc(s(J)*t/np.pi) )
    def Psi(J): return np.exp(-g*t/2) * (1j*s(J)-g/2) * t * np.sinc(s(J)*t/np.pi)

    # spin magnitude and variance along z is determined classically
    #   by uncorrelated decay events
    Sz = -N/2 * (1-np.exp(-g*t))
    Sz_Sz = N/4 * np.exp(-2*g*t) + Sz**2
    var_Sz = Sz_Sz - Sz**2

    # mean spin values
    Sp = N/2 * np.exp(-g*t/2) * Phi(1)**(N-1)
    Sm = np.conj(Sp)
    Sx = np.real(Sp)
    Sy = np.imag(Sp)

    # two-spin correlators
    Sp_Sz = -1/2 * Sp + 1/4 * N * (N-1) * np.exp(-g*t/2) * Psi(1) * Phi(1)**(N-2)
    Sm_Sz = 1/2 * Sm + 1/4 * N * (N-1) * np.exp(-g*t/2) * Psi(-1) * Phi(1)**(N-2)
    Sp_Sp = 1/4 * N * (N-1) * np.exp(-g*t) * Phi(2)**(N-2)
    Sm_Sm = 1/4 * N * (N-1) * np.exp(-g*t) * Phi(-2)**(N-2)
    Sp_Sm = N/2 + Sz + 1/4 * N * (N-1) * np.exp(-g*t) * Phi(0)**(N-2)
    Sm_Sp = N/2 - Sz + 1/4 * N * (N-1) * np.exp(-g*t) * Phi(0)**(N-2)

    Sx_Sz = 1/2 * ( Sp_Sz + Sm_Sz )
    Sy_Sz = -1j/2 * ( Sp_Sz - Sm_Sz )
    Sz_Sx = np.conj(Sx_Sz)
    Sz_Sy = np.conj(Sy_Sz)

    Sx_Sx = 1/4 * ( Sp_Sp + Sm_Sm + Sp_Sm + Sm_Sp )
    Sy_Sy = -1/4 * ( Sp_Sp + Sm_Sm - Sp_Sm - Sm_Sp )
    Sx_Sy = -1j/4 * ( Sp_Sp - Sm_Sm + Sp_Sm - Sm_Sp )
    Sy_Sx = np.conj(Sx_Sy)

    # otherwise perform a numerical minimization of the spin variance
    S_vec = np.real(np.array([ Sz, Sx, Sy ])).T
    SS_mat = np.array([ [ Sz_Sz, Sz_Sx, Sz_Sy ],
                        [ Sx_Sz, Sx_Sx, Sx_Sy ],
                        [ Sy_Sz, Sy_Sx, Sy_Sy ] ]).T

    if type(t) == np.ndarray:
        var_min = np.array([ minimal_orthogonal_variance(S_vec[ii], SS_mat[ii])
                             for ii in range(len(t)) ])
    else:
        var_min = minimal_orthogonal_variance(S_vec, SS_mat)

    return var_min * N / np.real(Sz*Sz + Sx*Sx + Sy*Sy)
