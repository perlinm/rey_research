#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import scipy.optimize as optimize

from scipy.special import binom as binomial
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors

np.set_printoptions(linewidth = 200)

##########################################################################################
# collective spin vectors in Dicke basis
##########################################################################################

def S_z(N): return np.diag(np.arange(N+1)-N/2)

def S_p(N):
    S = N/2
    m_z = np.arange(N) - N/2
    return np.diag(np.sqrt((S-m_z)*(S+m_z+1)),-1)

def S_m(N): return S_p(N).transpose()

def S_x(N):
    Sp = S_p(N)
    return 1/2 * ( Sp + Sp.transpose() )

def S_y(N):
    Sp = S_p(N)
    return -1j/2 * ( Sp - Sp.transpose() )

def S_n(vec, N):
    return ( vec[0] * S_z(N) + vec[1] * S_x(N) + vec[2] * S_y(N) ) / np.linalg.norm(vec)


##########################################################################################
# constructing states and calculating squeezing
##########################################################################################

# get polar and azimulthal angles of a vector (v_z, v_x, v_y)
def vec_theta_phi(v):
    return np.arccos(v[0]/linalg.norm(v)), np.arctan2(v[2],v[1])

# rotate a vector about an axis by a given angle
def rotate_vector(vector, axis, angle):
    L_z = np.array([ [  0,  0,  0 ],
                     [  0,  0, -1 ],
                     [  0,  1,  0 ] ])
    L_x = np.array([ [  0,  0,  1 ],
                     [  0,  0,  0 ],
                     [ -1,  0,  0 ] ])
    L_y = np.array([ [  0, -1,  0 ],
                     [  1,  0,  0 ],
                     [  0,  0,  0 ] ])
    L = ( axis[0] * L_z + axis[1] * L_x + axis[2] * L_y)  / np.linalg.norm(axis)
    return linalg.expm(angle * L) @ vector

# coherent spin state on S = N/2 Bloch sphere
def coherent_spin_state(vec, N = 10):
    theta, phi = vec_theta_phi(vec)
    state = np.zeros(N+1, dtype = complex)
    for m in range(N+1):
        c_theta = np.sin(theta/2)**(N-m) * np.cos(theta/2)**m
        c_phi = np.exp(1j * (N/2-m) * phi)
        state[m] = np.sqrt(binomial(N,m)) * c_theta * c_phi
    return state

# get spin vector <\vec S> for a state
def spin_vec(state):
    N = state.size-1
    def val(X): return state.conjugate() @ X @ state
    return np.real(np.array([ val(S_z(N)), val(S_x(N)), val(S_y(N)) ]))

# get normalized spin vector <\hat S> for a state
def spin_axis(state): return spin_vec(state) / ( (state.size-1) / 2)

# variance of spin state about an axis
def spin_variance(state, axis):
    N = state.size - 1
    Sn = S_n(axis, N)
    def val(X): return state.conjugate() @ X @ state
    return abs(val(Sn @ Sn) - abs(val(Sn))**2)

# return (\xi^2, axis), where:
#   "axis" is the axis of minimal spin variance in the plane orthogonal to <\vec S>
#   \xi^2 = (<S_axis^2> - <S_axis>^2) / (S/2) is the spin squeezing parameter
def spin_squeezing(state):
    S = (state.size-1)/2

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

    optimum = optimize.minimize(variance, 0) # use 0 as a guess for eta
    optimal_phi = optimum.x[0]
    squeezing_parameter = optimum.fun / (S/2)

    return squeezing_parameter, squeezing_axis(optimal_phi)


##########################################################################################
# plotting a state on the S = N/2 Bloch sphere
##########################################################################################

def plot_state(state, grid_size = 30):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(111, projection = "3d")

    theta, phi = np.mgrid[0:np.pi:(grid_size*1j), 0:2*np.pi:(grid_size*1j)]
    z = np.cos(theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)

    def color_val(theta, phi):
        vec = [ np.cos(theta), np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi) ]
        angle_state = coherent_spin_state(vec, state.size-1)
        return abs(angle_state.conjugate() @ state)**2
    color_val_vect = np.vectorize(color_val)
    color_vals = color_val_vect(theta, phi)
    norm = colors.Normalize(vmin = np.min(color_vals),
                            vmax = np.max(color_vals), clip = False)
    color_map = cm.inferno(norm(color_vals))

    ax.plot_surface(x, y-1.5, z, rstride = 1, cstride = 1,
                    facecolors = color_map)
    ax.plot_surface(-x, y+1.5, z, rstride = 1, cstride = 1,
                    facecolors = color_map)

    ax.set_axis_off()
    ax.view_init(elev = 0, azim = 0)
    ax.set_xlim(-1,1)
    ax.set_ylim(-2,2)
    ax.set_zlim(-1,1)
    fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
    return fig, ax


##########################################################################################
# actual squeezing methods
##########################################################################################

# squeeze a state with the one-axis twisting Hamiltonian \chi S_z^2 for a time t
def squeeze_OAT(state, chi_t):
    N = state.size - 1
    return np.exp(-1j*chi_t*(np.arange(N+1)-N/2)**2) * state

# squeezing parameter after one-axis twisting in a direction orthogonal to the initial state
def squeezing_OAT_perp(chi_t, N):
    A = 1 - np.cos(2*chi_t)**(N-2)
    B = 4 * np.sin(chi_t) * np.cos(chi_t)**(N-2)
    return (1 + 1/4*(N-1)*A) - 1/4*(N-1)*np.sqrt(A*A+B*B)

# squeeze a state with the two-axis twisting Hamiltonian \chi (S_z^2-S_y^2) for a time t,
def squeeze_TAT(state, chi_t):
    N = state.size-1
    Sz = S_z(N)
    Sy = S_y(N)
    return linalg.expm(-1j*chi_t*(Sz@Sz - Sy@Sy)) @ state



N = 50
S = N / 2

state_z = coherent_spin_state([1,0,0], N)
state_x = coherent_spin_state([0,1,0], N)
state_y = coherent_spin_state([0,0,1], N)

initial_state = state_x
tau_vals = np.linspace(0, 3, 100)
times = tau_vals * N**(-2/3)
squeezing_OAT_vals = np.vectorize(squeezing_OAT_perp)(times,N)
squeezing_TAT_vals = np.zeros(tau_vals.size)
for ii in range(times.size):
    final_TAT_state = squeeze_TAT(initial_state, times[ii]/3)
    squeezing_TAT_vals[ii], _ = spin_squeezing(final_TAT_state)

print("tau_OAT_opt:", tau_vals[squeezing_OAT_vals.argmin()])
print("tau_TAT_opt:", tau_vals[squeezing_TAT_vals.argmin()])

plt.plot(tau_vals, np.log(squeezing_OAT_vals), label = "OAT")
plt.plot(tau_vals, np.log(squeezing_TAT_vals), label = "TAT")
plt.xlabel(r"$\tau$")
plt.ylabel(r"$\ln(\xi^2)$")
plt.legend(loc="best")
plt.tight_layout()
plt.show()
