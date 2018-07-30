#!/usr/bin/env python3

import sys, scipy
import numpy as np
import matplotlib.pyplot as plt

from mathieu_methods import mathieu_solution
from overlap_methods import tunneling_1D, pair_overlap_1D
from sr87_olc_constants import g_int_LU, recoil_energy_NU, recoil_energy_Hz

from dicke_methods import spin_op_vec_mat_dicke, coherent_spin_state, squeezing_OAT
from fermi_hubbard_methods import prod, get_simulation_parameters, spatial_basis, \
    get_c_op_mats, spin_op_vec_mat_FH, polarized_states_FH, gauged_energy, H_full
from squeezing_methods import spin_vec_mat_vals, spin_squeezing, evolve

compute_TAT = "tat" in sys.argv
show = "show" in sys.argv
save = "save" in sys.argv

figsize = (4,3)
fig_dir = "../figures/"
params = { "text.usetex" : True }
plt.rcParams.update(params)

L = 100 # lattice sites
phi = np.pi/25 # spin-orbit coupling parameter

lattice_depth = 5 # shallow (tunneling) axis lattice depth
confining_depth = 60 # lattice depth along confining axes

fermi_N_cap = 8 # maximum number of atoms for which to run Fermi Hubbard calculations
use_hubbard = False # use the hubbard model?

max_tau = 2 # for simulation: chi * max_time = max_tau * N **(-2/3)
time_steps = 1000 # time steps in simulation

N = prod(L)
L, J_0, phi, K_0, momenta, fourier_vecs, energies, J_T, K_T = \
    get_simulation_parameters(L, phi, lattice_depth, confining_depth)

U_int = g_int_LU[1] * K_T**(3-L.size) * prod(K_0)
energies_or_J = J_0 if use_hubbard else energies
soc_field_variance = np.var([ ( gauged_energy(q, 1, phi, L, energies_or_J)
                                - gauged_energy(q, 0, phi, L, energies_or_J) )
                              for q in spatial_basis(L) ])
if np.sqrt(soc_field_variance)/U_int < 1e-5:
    sys.exit("there is no spin squeezing with the given parameters!")
chi = soc_field_variance / ( N * (N-1) * U_int / prod(L) )
omega = N * np.sqrt(abs(chi*U_int/prod(L)))

print(r"J_T (2\pi Hz):", J_T * recoil_energy_Hz)
print()
for ii in range(len(J_0)):
    print("J_{} (2\pi Hz):".format(ii), J_0[ii] * recoil_energy_Hz)
print(r"U_int (2\pi kHz):", U_int * recoil_energy_Hz)
print(r"chi (2\pi mHz):", chi * recoil_energy_Hz * 1e3)
print(r"omega (2\pi Hz):", omega * recoil_energy_Hz)
print()
print(r"h_std/U_int:", np.sqrt(soc_field_variance) / U_int)
print()

tau_vals = np.linspace(0, max_tau, time_steps)
chi_times = tau_vals * N**(-2/3)
times = chi_times / chi
times_SI = times / recoil_energy_NU

S_op_vec, SS_op_mat = spin_op_vec_mat_dicke(N)
H = 1/3 * ( SS_op_mat[0][0] - SS_op_mat[2][2] )
state = coherent_spin_state([0,1,0], N)

squeezing_OAT_vals = np.vectorize(squeezing_OAT)(chi_times, N)
t_opt_OAT = times[squeezing_OAT_vals.argmin()]
print("t_opt_OAT (sec):", t_opt_OAT / recoil_energy_NU)

if not (compute_TAT or show or save): exit()

squeezing_TAT_vals = np.zeros(time_steps) # initialize vector of TAT squeezing values
d_chi_t = chi_times[1] - chi_times[0] # size of one time step
for ii in range(time_steps):
    squeezing_TAT_vals[ii], _ = spin_squeezing(state, S_op_vec, SS_op_mat, N)
    state = evolve(state, H, d_chi_t)
t_opt_TAT = times[squeezing_TAT_vals.argmin()]
print("t_opt_TAT (sec):", t_opt_TAT / recoil_energy_NU)
print()

if not (show or save): exit()

def to_dB(x): return 10*np.log10(x)

plt.figure(figsize = figsize)

plt.plot(times_SI, -to_dB(squeezing_OAT_vals), label = "OAT")
plt.plot(times_SI, -to_dB(squeezing_TAT_vals), label = "TAT")

if N <= fermi_N_cap:
    hilbert_dim = int(scipy.special.binom(2*prod(L),N))
    print("Fermi-Hubbard hilbert space dimension:", hilbert_dim)

    c_op_mats = get_c_op_mats(L, N, depth = 2)
    S_op_vec, SS_op_mat = spin_op_vec_mat_FH(L, N, c_op_mats)
    state_z, _, _ = polarized_states_FH(L, N)

    H_lat, H_int, H_clock = H_full(N, L, phi, lattice_depth, confining_depth,
                                   c_op_mats, use_hubbard)
    H_free = H_lat + H_int

    beta = 0.90572
    def H_laser(t):
        return -beta * omega * np.cos(omega * t) * H_clock

    dt = times[-1] / time_steps
    squeezing_free_vals = np.zeros(time_steps)
    squeezing_drive_vals = np.zeros(time_steps)
    state_free = evolve(state_z, H_clock, -np.pi/2)
    state_drive = state_free.copy()
    counter = 0
    for ii in range(time_steps):
        if ii * 10 // time_steps >= counter:
            counter += 1
            print("{}/{}".format(ii,time_steps))
        H_drive = H_free + H_laser(times[ii]+dt/2)
        squeezing_free_vals[ii], _ = spin_squeezing(state_free, S_op_vec, SS_op_mat, N)
        squeezing_drive_vals[ii], _ = spin_squeezing(state_drive, S_op_vec, SS_op_mat, N)
        state_free = evolve(state_free, H_free, dt)
        state_drive = evolve(state_drive, H_drive, dt)

    plt.plot(times_SI, -to_dB(squeezing_free_vals), label = "FH (free)")
    plt.plot(times_SI, -to_dB(squeezing_drive_vals), label = "FH (driven)")

plt.xlim(0,times_SI[-1])
plt.ylim(0,plt.gca().get_ylim()[1])
plt.xlabel(r"Time (seconds)")
plt.ylabel(r"Squeezing: $-10\log_{10}(\xi^2)$")
plt.legend(loc="best")
plt.tight_layout()

if save: plt.savefig(fig_dir + "squeezing_comparison.pdf")
if show: plt.show()
