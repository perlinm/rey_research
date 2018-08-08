#!/usr/bin/env python3

# FILE CONTENTS: methods for simulating dynamics in the Dicke manifold

import numpy as np
import scipy.sparse as sparse
import scipy.linalg as linalg
import scipy.optimize as optimize
import matplotlib.pyplot as plt

from squeezing_methods import minimal_orthogonal_variance

from scipy.special import binom, factorial

# spin operators for N particles in the S = N/2 Dicke manifold
def spin_op_z_dicke(N):
    Sz = sparse.lil_matrix((N+1,N+1))
    Sz.setdiag(np.arange(N+1)-N/2)
    return Sz.tocsr()

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


##########################################################################################
# analytical solutions to special cases
##########################################################################################

# squeezing parameter from one-axis twisting, accounting for e --> g decay
# spin correlators retrieved from foss-feig2013nonequilibrium
def squeezing_OAT(chi_t, N, decay_rate_over_chi = 0):
    g = decay_rate_over_chi # shorthand for decay rate in units with \chi = 1
    t = chi_t # shorthand for time in units with \chi = 1

    def s(J): return J + 1j*g/2
    def Phi(J): return np.exp(-g*t/2) * ( np.cos(s(J)*t) + g*t/2 * np.sinc(s(J)*t/np.pi) )
    def Psi(J): return np.exp(-g*t/2) * (1j*s(J)-g/2) * t * np.sinc(s(J)*t/np.pi)

    # spin magnitude and variance along z is determined classically
    #   by uncorrelated decay events
    decay_prob = 1 - np.exp(-g*t)
    Sz = -N/2 * decay_prob # < Sz >
    dSz_2 = N/4 + N/2 * decay_prob * (1-decay_prob) # < Sz^2 > - < Sz >^2
    Sz_2 = dSz_2 + Sz**2 # < Sz^2 >

    # remaining mean spin values
    Sp = N/2 * np.exp(-g*t/2) * Phi(1)**(N-1) # < S+ >
    Sx = np.real(Sp) # < Sx >
    Sy = np.imag(Sp) # < Sy >

    # symmetrized two-spin correlators
    Sp_2 = 1/4 * N * (N-1) * np.exp(-g*t) * Phi(2)**(N-2) # < S+ S+ >
    Sm_2 = 1/4 * N * (N-1) * np.exp(-g*t) * Phi(-2)**(N-2) # < S- S- >
    Sp_Sm_sym = N/2 + 1/4 * N * (N-1) * np.exp(-g*t) * Phi(0)**(N-2) # < S+ S- >
    Sp_Sz_sym = 1/4 * N * (N-1) * np.exp(-g*t/2) * Psi(1) * Phi(1)**(N-2) # < S+ Sz >

    Sx_2 = 1/4 * np.real( Sp_2 + Sm_2 + 2 * Sp_Sm_sym ) # < Sx Sx >
    Sy_2 = -1/4 * np.real( Sp_2 + Sm_2 - 2 * Sp_Sm_sym ) # < Sy Sy >
    Sx_Sz_sym = np.real(Sp_Sz_sym)  # < Sx Sz >
    Sy_Sz_sym = np.imag(Sp_Sz_sym) # < Sy Sz >
    Sx_Sy_sym = 1/4 * np.imag(Sp_2 - Sm_2) # < Sx Sy >

    # spin variances in x and y
    dSx_2 = Sx_2 - Sx**2 # < Sx^2 > - < Sx >^2
    dSy_2 = Sy_2 - Sy**2 # < Sy^2 > - < Sy >^2

    # minimal spin variance in the plane orthogonal to the mean spin vector
    if g == 0:
        # if there is no spin decay, we have simple analytical formulas
        A = dSy_2 - dSz_2
        B = 2 * Sy_Sz_sym
        C = dSy_2 + dSz_2
        var_min = 1/2 * ( C - np.sqrt(A**2 + B**2) )
    else:
        # otherwise perform a numerical minimization of the spin variance
        S_vec = np.array([ Sz, Sx, Sy ]).T
        SS_mat = np.array([ [ Sz_2,      Sx_Sz_sym, Sy_Sz_sym ],
                            [ Sx_Sz_sym, Sx_2,      Sx_Sy_sym ],
                            [ Sy_Sz_sym, Sx_Sy_sym, Sy_2      ] ]).T
        if type(t) == np.ndarray:
            var_min = np.array([ minimal_orthogonal_variance(S_vec[ii], SS_mat[ii])
                                 for ii in range(len(t)) ])
        else:
            var_min = minimal_orthogonal_variance(S_vec, SS_mat)

    # return squeezing parameter: \xi^2 = var_min \times N / |<S>|^2
    return var_min * N / (Sz*Sz + Sx*Sx + Sy*Sy)
