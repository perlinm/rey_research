#!/usr/bin/env python3

import sys
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

from scipy.special import binom

from mathieu_methods import mathieu_solution
from overlap_methods import tunneling_1D, pair_overlap_1D
from sr87_olc_constants import g_int_LU, recoil_energy_NU, recoil_energy_Hz
from squeezing_methods import squeezing_OAT, coherent_spin_state, spin_squeezing
from squeezing_methods import spin_op_vec_mat as spin_op_vec_mat_dicke
from squeezing_methods import evolve as evolve_dicke

from fermi_hubbard_methods import get_c_op_mats, spin_op_vec_mat, polarized_states, \
    H_lat_q, H_int_q, spin_squeezing_FH, evolve, product, spacial_basis


show = "show" in sys.argv
save = "save" in sys.argv

figsize = (4,3)
fig_dir = "../figures/"
params = { "text.usetex" : True }
plt.rcParams.update(params)

L = 100 # lattice sites
N = int(product(L) / 2) # atoms
phi = np.pi / 50 # spin-orbit coupling parameter
fermi_N_limit = 8 # maximum number of atoms for which to run Fermi Hubbard calculations
periodic = True # use periodic boundary conditions?

lattice_depth = 4 # shallow (tunneling) axis lattice depth

max_tau = 2 # for simulation: chi * max_time = max_tau * N **(-2/3)
time_steps = 1000 # time steps in simulation

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
elif len(L) == 2:
    U = g_int_LU[1] * K_0**2 * K_T

L = np.array(L, ndmin = 1)
soc_field_vals = -4 * J_0 * np.sin(phi/2) * np.array([ np.sin(2*np.pi*q[ii]/L[ii])
                                                       for ii in range(len(L)) if L[ii] > 2
                                                       for q in spacial_basis(L) ])
soc_field_mean = np.mean(soc_field_vals)
soc_field_variance = np.mean( (soc_field_vals - soc_field_mean)**2 )
chi = soc_field_variance / ( (N-1) * eta * U )
if chi < 1e-10: sys.exit("there is no spin squeezing with the given parameters!")

print("J_0 (2\pi Hz):", J_0 * recoil_energy_Hz)
print("U (2\pi Hz):", U * recoil_energy_Hz)
print("U/L (2\pi Hz):", U / product(L) * recoil_energy_Hz)
print("chi (2\pi Hz):", chi * recoil_energy_Hz)
print("omega (2\pi Hz):", N * np.sqrt(abs(chi*U/product(L))) * recoil_energy_Hz)
print()

tau_vals = np.linspace(0, max_tau, time_steps)
chi_times = tau_vals * N**(-2/3)
times = chi_times / chi
times_SI = times / recoil_energy_NU

S_op_vec, SS_op_mat = spin_op_vec_mat_dicke(N)
H = 1/3 * ( SS_op_mat[0][0] - SS_op_mat[2][2] )
state = coherent_spin_state([0,1,0], N)

squeezing_OAT_vals = np.vectorize(squeezing_OAT)(chi_times, N)
squeezing_TAT_vals = np.zeros(time_steps) # initialize vector of TAT squeezing values
d_chi_t = chi_times[1] - chi_times[0] # size of one time step
for ii in range(time_steps):
    squeezing_TAT_vals[ii], _ = spin_squeezing(state, S_op_vec, SS_op_mat, N)
    state = evolve_dicke(state, d_chi_t, H)

tau_opt_OAT = tau_vals[squeezing_OAT_vals.argmin()]
tau_opt_TAT = tau_vals[squeezing_TAT_vals.argmin()]
t_opt_OAT = tau_opt_OAT * N**(-2/3) / chi
t_opt_TAT = tau_opt_TAT * N**(-2/3) / chi

print("t_opt_OAT (sec):", t_opt_OAT / recoil_energy_NU)
print("t_opt_TAT (sec):", t_opt_TAT / recoil_energy_NU)

if not (show or save): exit()

def to_dB(x): return 10*np.log10(x)

plt.figure(figsize = figsize)

plt.plot(times_SI, -to_dB(squeezing_OAT_vals), label = "OAT")
plt.plot(times_SI, -to_dB(squeezing_TAT_vals), label = "TAT")

if N <= fermi_N_limit:
    print()
    print("Fermi-Hubbard hilbert space dimension:", int(binom(2*product(L),N)))
    c_op_mats = get_c_op_mats(L, N, depth = 2)
    S_op_vec, SS_op_mat = spin_op_vec_mat(L, N, c_op_mats)
    state_z, state_x, state_y = polarized_states(L, N)

    state_free = state_x
    state_drive = state_x
    H_free = H_int_q(L, N, U, c_op_mats) + H_lat_q(L, N, J_0, phi, c_op_mats, periodic)

    omega = N * np.sqrt(abs(chi*U/product(L)))
    beta = 0.90572
    def H_laser(t):
        return -beta * omega * np.cos(omega * t) * S_op_vec[2]

    dt = times[-1] / time_steps
    squeezing_free_vals = np.zeros(time_steps)
    squeezing_drive_vals = np.zeros(time_steps)
    for ii in range(time_steps):
        print("{}/{}".format(ii,time_steps))
        H_drive = H_free + H_laser(times[ii]+dt/2)
        squeezing_free_vals[ii], _ = spin_squeezing_FH(state_free, S_op_vec, SS_op_mat, N)
        squeezing_drive_vals[ii], _ = spin_squeezing_FH(state_drive, S_op_vec, SS_op_mat, N)
        state_free = evolve(state_free, dt, H_free)
        state_drive = evolve(state_drive, dt, H_drive)

    plt.plot(times_SI, -to_dB(squeezing_free_vals), label = "FH (free)")
    plt.plot(times_SI, -to_dB(squeezing_drive_vals), label = "FH (driven)")

plt.xlim(0,times_SI[-1])
plt.xlabel(r"Time (seconds)")
plt.ylabel(r"Squeezing: $-10\log_{10}(\xi^2)$")
plt.legend(loc="best")
plt.tight_layout()

if save: plt.savefig(fig_dir + "squeezing_comparison.pdf")
if show: plt.show()
