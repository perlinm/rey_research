#!/usr/bin/env python3

# FILE CONTENTS: methods for simulating dynamics in the Dicke manifold

import numpy as np
import scipy.sparse as sparse
import scipy.linalg as linalg

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors

from special_functions import axis_str, vec_theta_phi

# natural logarithms of factorial and binomial coefficient
from math import lgamma
def ln_factorial(n): return lgamma(n+1)
ln_factorial = np.vectorize(ln_factorial)
def ln_binom(N,m): return ln_factorial(N) - ln_factorial(m) - ln_factorial(N-m)

# spin operators for N particles in the S = N/2 Dicke manifold
def spin_op_z_dicke(N):
    return sparse.diags(np.arange(N+1)-N/2)

def spin_op_m_dicke(N):
    S = N/2
    m_z = np.arange(N) - N/2
    diag_vals = np.sqrt((S-m_z)*(S+m_z+1))
    return sparse.diags(diag_vals, 1)

def spin_op_p_dicke(N):
    return spin_op_m_dicke(N).T

def spin_op_x_dicke(N):
    Sm = spin_op_m_dicke(N)
    return ( Sm + Sm.T ) / 2

def spin_op_y_dicke(N):
    Sm = spin_op_m_dicke(N)
    return ( Sm - Sm.T ) * 1j / 2

def spin_op_vec_dicke(N):
    Sz = spin_op_z_dicke(N)
    Sm = spin_op_m_dicke(N)
    Sx = ( Sm + Sm.T ) / 2
    Sy = ( Sm - Sm.T ) * 1j / 2
    return [ Sz, Sx, Sy ]

def spin_op_vec_mat_dicke(N):
    Sz, Sx, Sy = spin_op_vec_dicke(N)

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

# coherent spin state on many-body Bloch sphere
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
    if theta < 0:
        theta = -theta
        phi += np.pi
    m_vals = np.array(range(N+1))
    ln_magnitudes = ( 1/2 * ln_binom(N,m_vals)
                      + (N-m_vals) * np.log(np.sin(theta/2))
                      + m_vals * np.log(np.cos(theta/2)) )
    phases = np.exp(1j * (N/2-m_vals) * phi)
    return np.exp(ln_magnitudes) * phases

def coherent_spin_state(vec, N = 10):
    if type(vec) is str: return coherent_spin_state(axis_str(vec), N)
    theta, phi = vec_theta_phi(vec)
    return coherent_spin_state_angles(theta, phi, N)


##########################################################################################
# plotting states on the S = N/2 Bloch sphere
##########################################################################################

try:
    import cmocean
    def sphere_cmap(color_vals):
        return cmocean.cm.amp(color_vals)
except ModuleNotFoundError:
    def sphere_cmap(color_vals):
        return plt.cm.get_cmap("inferno")(color_vals)

def plot_dicke_state(state, grid_size = 101, single_sphere = True, figsize = None,
                     rasterized = True):
    N = state.shape[0]-1
    if figsize is None:
        if single_sphere:
            figsize = plt.figaspect(1)
        else:
            figsize = plt.figaspect(0.5)
    fig = plt.figure(figsize = figsize)
    ax = Axes3D(fig)

    theta, phi = np.mgrid[0:np.pi:(grid_size*1j), 0:2*np.pi:(grid_size*1j)]
    z_vals = np.cos(theta)
    x_vals = np.sin(theta) * np.cos(phi)
    y_vals = np.sin(theta) * np.sin(phi)

    def color_val(theta, phi):
        angle_state = coherent_spin_state_angles(theta, phi, N)
        if state.ndim == 1:
            return abs(angle_state.conjugate() @ state)**2
        else:
            return np.einsum("i,ij,j->", angle_state.conjugate(), state, angle_state).real
    color_vals = np.vectorize(color_val)(theta, phi)
    norm = colors.Normalize(vmin = np.min(color_vals),
                            vmax = np.max(color_vals), clip = False)
    color_map = sphere_cmap(norm(color_vals))

    ax_lims = np.array([-1,1]) * 0.6
    ax.set_xlim(ax_lims)
    ax.set_ylim(ax_lims)
    ax.set_zlim(ax_lims)

    asymmetry = 0.025
    z_vals -= asymmetry

    if single_sphere:
        y_vals -= asymmetry
        ax.plot_surface(x_vals, y_vals, z_vals, rstride = 1, cstride = 1,
                        facecolors = color_map, rasterized = rasterized)
    else:
        y_vals -= 2*asymmetry
        y_offset = 1.1
        y_lims = 2*ax_lims
        ax.plot_surface(x_vals, y_vals-y_offset, z_vals, rstride = 1, cstride = 1,
                        facecolors = color_map, rasterized = rasterized)
        ax.plot_surface(-x_vals, y_vals+y_offset, z_vals, rstride = 1, cstride = 1,
                        facecolors = color_map, rasterized = rasterized)
        ax.set_ylim(y_lims)

    ax.set_axis_off()
    ax.view_init(elev = 0, azim = 0)
    return fig, ax
