#!/usr/bin/env python3

import sys
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

from mathieu_methods import mathieu_solution
from overlap_methods import tunneling_1D, pair_overlap_1D
from sr87_olc_constants import g_int_LU, recoil_energy_NU, recoil_energy_Hz
from squeezing_methods import squeezing_OAT, coherent_spin_state, \
    squeezing_OAT_propagator, squeezing_TAT_propagator, spin_squeezing

from fermi_hubbard_methods import get_c_op_mats, spin_op_vec_mat, polarized_states, \
    H_lat_q, H_int_q, H_lat_j, H_int_j, product
from fermi_hubbard_methods import spin_squeezing as spin_squeezing_fermi


show = "show" in sys.argv
save = "save" in sys.argv

figsize = (4,3)
fig_dir = "../figures/"
params = { "text.usetex" : True }
plt.rcParams.update(params)

N = 6 # atoms
L = 6 # lattice sites
phi = np.pi / 50 # spin-orbit coupling parameter

lattice_depth = 4 # shallow (tunneling) axis lattice depth

max_tau = 2 # for simulation: chi * max_time = max_tau * N **(-2/3)
time_steps = 100 # time steps in simulation


# lattice bands and site number: only matters for calculations of lattice parameters
bands = 5
site_number = 100
c_lattice_depth = 60 # deep (confining) axis lattice depth
eta = N / product(L) # total filling fraction

c_momenta, c_fourier_vecs, _ = mathieu_solution(c_lattice_depth, bands, site_number)
J_T = tunneling_1D(c_lattice_depth, c_momenta, c_fourier_vecs)
K_T = pair_overlap_1D(c_momenta, c_fourier_vecs)
momenta, fourier_vecs, _ = mathieu_solution(lattice_depth, bands, site_number)
K_0 = pair_overlap_1D(momenta, fourier_vecs)
J_0 = tunneling_1D(lattice_depth, momenta, fourier_vecs)

print("J_T (2\pi Hz):", J_T * recoil_energy_Hz)
print()

if type(L) == int:
    U = g_int_LU[1] * K_0 * K_T**2
    chi = 8 * J_0**2 * np.sin(phi/2)**2 / ((N-1) * eta * U)
elif len(L) == 2:
    U = g_int_LU[1] * K_0**2 * K_T
    chi = 16 * J_0**2 * np.sin(phi/2)**2 / ((N-1) * eta * U)

print("J_0 (2\pi Hz):", J_0 * recoil_energy_Hz)
print("U (2\pi Hz):", U * recoil_energy_Hz)
print("U/L (2\pi Hz):", U / product(L) * recoil_energy_Hz)
print("chi (2\pi Hz):", chi * recoil_energy_Hz)
print("omega (2\pi Hz):", N * np.sqrt(chi*U/product(L)) * recoil_energy_Hz)
print()

tau_vals = np.linspace(0, max_tau, time_steps)
chi_times = tau_vals * N**(-2/3)
times = chi_times / chi
times_SI = times / recoil_energy_NU

squeezing_OAT_vals = np.vectorize(squeezing_OAT)(chi_times, N)
squeezing_TAT_vals = np.zeros(chi_times.size) # initialize vector of TAT squeezing values
state = coherent_spin_state([0,1,0], N) # initialize state pointing in x
d_chi_t = chi_times[1] - chi_times[0] # size of one time step
dU_TAT = squeezing_TAT_propagator(d_chi_t/3, N) # TAT propagator for one time step
for ii in range(squeezing_OAT_vals.size):
    squeezing_TAT_vals[ii] = spin_squeezing(state)[0] # compute squeezing parameter
    state = dU_TAT @ state # propagate state for one time step

tau_opt_OAT = tau_vals[squeezing_OAT_vals.argmin()]
tau_opt_TAT = tau_vals[squeezing_TAT_vals.argmin()]
t_opt_OAT = tau_opt_OAT * N**(-2/3) / chi
t_opt_TAT = tau_opt_TAT * N**(-2/3) / chi

print("t_opt_OAT (sec):", t_opt_OAT / recoil_energy_NU)
print("t_opt_TAT (sec):", t_opt_TAT / recoil_energy_NU)

if not (show or save): exit()

def to_dB(x): return 10*np.log10(x)

plt.figure(figsize = figsize)

##########################################################################################
##########################################################################################

c_op_mats = get_c_op_mats(L, N)
S_op_vec, SS_op_mat = spin_op_vec_mat(L, N, c_op_mats)
state_z, state_x, state_y = polarized_states(L, N)

H = H_lat_q(L, N, J_0, phi, c_op_mats) + H_int_q(L, N, U, c_op_mats)

state = state_x
using_state_vectors = state.shape != H.shape

dt = times[-1] / time_steps
squeezing_fermi_vals = np.zeros(time_steps)
for ii in range(time_steps):
    squeezing_fermi_vals[ii], _ = spin_squeezing_fermi(state, S_op_vec, SS_op_mat, N)
    state = sparse.linalg.expm_multiply(-1j*dt*H, state)
    if not using_state_vectors:
        state = sparse.linalg.expm_multiply(-1j*dt*H, state.conj().T).conj().T

##########################################################################################
##########################################################################################

plt.plot(times_SI, -to_dB(squeezing_OAT_vals), label = "OAT")
plt.plot(times_SI, -to_dB(squeezing_TAT_vals), label = "TAT")
plt.plot(times_SI, -to_dB(squeezing_fermi_vals), label = "FH")
plt.xlim(0,times_SI[-1])
plt.xlabel(r"Time (seconds)")
plt.ylabel(r"Squeezing: $-10\log_{10}(\xi^2)$")
plt.legend(loc="best")
plt.tight_layout()

if save: plt.savefig(fig_dir + "squeezing_comparison.pdf")
if show: plt.show()
