#!/usr/bin/env python3

# FILE CONTENTS: methods for simulating dynamics in the Dicke manifold

import numpy as np
import scipy.sparse as sparse
import scipy.linalg as linalg
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors

from squeezing_methods import ln_binom, squeezing_from_correlators

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
    return np.array([ np.arccos(v[0]/linalg.norm(v)), np.arctan2(v[2],v[1]) ])

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
# analytical solutions
##########################################################################################

# exact correlators for OAT with decoherence
# derivations in foss-feig2013nonequilibrium
def correlators_OAT_exact(N, chi_t, decay_rate_over_chi):
    g = decay_rate_over_chi # shorthand for decay rate in units with \chi = 1
    t = chi_t # shorthand for time in units with \chi = 1

    Sz = N/2 * (np.exp(-g*t)-1)
    var_Sz = N/2 * (1 - np.exp(-g*t)/2) * np.exp(-g*t)
    Sz_Sz = var_Sz + Sz**2

    def s(J): return J + 1j*g/2
    def Phi(J): return np.exp(-g*t/2) * ( np.cos(s(J)*t) + g*t/2 * np.sinc(s(J)*t/np.pi) )
    def Psi(J): return np.exp(-g*t/2) * (1j*s(J)-g/2) * t * np.sinc(s(J)*t/np.pi)

    Sp = N/2 * np.exp(-g*t/2) * Phi(1)**(N-1)
    Sm = np.conj(Sp)

    Sp_Sz = -1/2 * Sp + 1/4 * N * (N-1) * np.exp(-g*t/2) * Psi( 1) * Phi( 1)**(N-2)
    Sm_Sz =  1/2 * Sm + 1/4 * N * (N-1) * np.exp(-g*t/2) * Psi(-1) * Phi(-1)**(N-2)
    Sp_Sp = 1/4 * N * (N-1) * np.exp(-g*t) * Phi( 2)**(N-2)
    Sp_Sm = N/2 + Sz + 1/4 * N * (N-1) * np.exp(-g*t) # note that Phi(0) == 1

    return Sz, Sz_Sz, Sp, Sp_Sz, Sm_Sz, Sp_Sp, Sp_Sm


# squeezing parameter from one-axis twisting, accounting for e --> g decay
def squeezing_OAT(N, chi_t, decay_rate_over_chi = 0):

    # if there is no spin decay, we have simple analytical formulas
    if decay_rate_over_chi == 0:

        S = N/2
        var_Sy = S/2 + S/2 * (S-1/2) * ( 1 - np.cos(2*chi_t)**(N-2) )
        var_Sz = S/2

        A = var_Sy - var_Sz
        C = var_Sy + var_Sz
        B = 2 * S * (S-1/2) * np.sin(chi_t) * np.cos(chi_t)**(N-2)

        var_min = 1/2 * np.real( C - np.sqrt(A**2 + B**2) )
        Sx = S * np.cos(chi_t)**(N-1)

        return var_min * N / Sx**2

    # otherwise, use more complex but exact spin correlators to compute squeezing
    correlators = correlators_OAT_exact(N, chi_t, decay_rate_over_chi)

    return squeezing_from_correlators(N, *correlators)
