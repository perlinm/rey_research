#!/usr/bin/env python3

import sys, scipy
import numpy as np
import matplotlib.pyplot as plt

from mathieu_methods import mathieu_solution
from overlap_methods import tunneling_1D, pair_overlap_1D
from sr87_olc_constants import g_int_LU, recoil_energy_NU, recoil_energy_Hz

from dicke_methods import spin_op_vec_mat_dicke, coherent_spin_state, squeezing_OAT
from fermi_hubbard_methods import product, get_simulation_parameters, spatial_basis, \
    get_c_op_mats, spin_op_vec_mat_FH, polarized_states_FH, gauged_energy, H_full
from squeezing_methods import spin_vec_mat_vals, spin_squeezing, evolve

show = "show" in sys.argv
save = "save" in sys.argv

figsize = (4,3)
fig_dir = "../figures/"
params = { "text.usetex" : True }
plt.rcParams.update(params)

L = 5 # lattice sites
N = product(L) # atoms
phi = np.pi / 50 # spin-orbit coupling parameter
fermi_N_cap = 8 # maximum number of atoms for which to run Fermi Hubbard calculations
use_hubbard = True # use the hubbard model?

lattice_depth = 4 # shallow (tunneling) axis lattice depth
confining_depth = 60 # lattice depth along confining axes

max_tau = 2 # for simulation: chi * max_time = max_tau * N **(-2/3)
time_steps = 1000 # time steps in simulation

L, J_0, phi, K_0, momenta, fourier_vecs, energies, J_T, K_T = \
    get_simulation_parameters(L, phi, lattice_depth, confining_depth)

U = g_int_LU[1] * K_T**(3-L.size) * product(K_0)
energies_or_J = J_0 if use_hubbard else energies
soc_field_variance = np.var([ ( gauged_energy(q, 1, phi, L, energies_or_J)
                                - gauged_energy(q, 0, phi, L, energies_or_J) )
                              for q in spatial_basis(L) ])
if soc_field_variance/U**2 < 1e-10:
    sys.exit("there is no spin squeezing with the given parameters!")
chi = soc_field_variance / ( N * (N-1) * U / product(L) )
omega = N * np.sqrt(abs(chi*U/product(L)))

print(r"J_T (2\pi Hz):", J_T * recoil_energy_Hz)
print()
for ii in range(len(J_0)):
    print("J_{} (2\pi Hz):".format(ii), J_0[ii] * recoil_energy_Hz)
print(r"U (2\pi Hz):", U * recoil_energy_Hz)
print(r"chi (2\pi Hz):", chi * recoil_energy_Hz)
print(r"omega (2\pi Hz):", omega * recoil_energy_Hz)
print()
print(r"\tilde{h}/U:", np.sqrt(soc_field_variance) / U) # needs to be < 0.05
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

squeezing_TAT_vals = np.zeros(time_steps) # initialize vector of TAT squeezing values
d_chi_t = chi_times[1] - chi_times[0] # size of one time step
for ii in range(time_steps):
    squeezing_TAT_vals[ii], _ = spin_squeezing(state, S_op_vec, SS_op_mat, N)
    state = evolve(state, H, d_chi_t)
t_opt_TAT = times[squeezing_TAT_vals.argmin()]
print("t_opt_TAT (sec):", t_opt_TAT / recoil_energy_NU)

if not (show or save): exit()

def to_dB(x): return 10*np.log10(x)

plt.figure(figsize = figsize)

plt.plot(times_SI, -to_dB(squeezing_OAT_vals), label = "OAT")
plt.plot(times_SI, -to_dB(squeezing_TAT_vals), label = "TAT")

if N <= fermi_N_cap:
    print()
    hilbert_dim = int(scipy.special.binom(2*product(L),N))
    print("Fermi-Hubbard hilbert space dimension:", hilbert_dim)
    c_op_mats = get_c_op_mats(L, N, depth = 2)
    S_op_vec, SS_op_mat = spin_op_vec_mat_FH(L, N, c_op_mats)
    _, state_x, _ = polarized_states_FH(L, N)

    state_free = state_x
    state_drive = state_x
    H_free = H_full(N, L, phi, lattice_depth, confining_depth, c_op_mats, use_hubbard)

    beta = 0.90572
    def H_laser(t):
        return -beta * omega * np.cos(omega * t) * S_op_vec[2]

    dt = times[-1] / time_steps
    squeezing_free_vals = np.zeros(time_steps)
    squeezing_drive_vals = np.zeros(time_steps)
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
