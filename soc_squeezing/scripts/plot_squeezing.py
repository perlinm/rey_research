#!/usr/bin/env python3

import sys
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import itertools

from scipy.special import binom

from mathieu_methods import mathieu_solution
from overlap_methods import tunneling_1D, pair_overlap_1D
from sr87_olc_constants import g_int_LU, recoil_energy_NU, recoil_energy_Hz

from dicke_methods import spin_op_vec_mat_dicke, coherent_spin_state, squeezing_OAT
from fermi_hubbard_methods import product, spatial_basis, \
    get_c_op_mats, spin_op_vec_mat_FH, polarized_states_FH, H_full
from squeezing_methods import spin_vec_mat_vals, spin_squeezing, evolve, val

show = "show" in sys.argv
save = "save" in sys.argv

figsize = (4,3)
fig_dir = "../figures/"
params = { "text.usetex" : True }
plt.rcParams.update(params)

L = [ 100, 100 ] # lattice sites
N = product(L) # atoms
phi = np.pi / 30 # spin-orbit coupling parameter
fermi_N_cap = 8 # maximum number of atoms for which to run Fermi Hubbard calculations
periodic = True # use periodic boundary conditions?

lattice_depth = 4 # shallow (tunneling) axis lattice depth

max_tau = 2 # for simulation: chi * max_time = max_tau * N **(-2/3)
time_steps = 1000 # time steps in simulation

# lattice bands and site number: only matters for calculations of lattice parameters
bands = 5
site_number = 121
c_lattice_depth = 60 # lattice depth along confining axes

momenta = [ None ] * len(L)
vecs = [ None ] * len(L)
J_0 = [ None ] * len(L)
K_0 = [ None ] * len(L)

c_momenta, c_fourier_vecs, _ = mathieu_solution(c_lattice_depth, bands, site_number)
J_T = tunneling_1D(c_lattice_depth, c_momenta, c_fourier_vecs)
K_T = pair_overlap_1D(c_momenta, c_fourier_vecs)

for ii in range(len(L)):
    momenta[ii], vecs[ii], _ = mathieu_solution(lattice_depth, bands, L[ii])
    J_0[ii] = tunneling_1D(lattice_depth, momenta[ii], vecs[ii])
    K_0[ii] = pair_overlap_1D(momenta[ii], vecs[ii])

U = g_int_LU[1] * K_T**(3-len(L)) * product(K_0)

tun_sin_vals = np.array([ J_0[ii] * np.sin(2*np.pi/L[ii] * (q[ii]-(L[ii]-1)/2))
                          for ii in range(len(L)) if L[ii] > 2
                          for q in spatial_basis(L) ])
soc_field_vals = -4 * np.sin(phi/2) * tun_sin_vals
soc_field_variance = np.mean( ( soc_field_vals - np.mean(soc_field_vals) )**2 )
chi = soc_field_variance / ( N * (N-1) * U / product(L) )
if chi < 1e-10: sys.exit("there is no spin squeezing with the given parameters!")
omega = N * np.sqrt(abs(chi*U/product(L)))

print("J_T (2\pi Hz):", J_T * recoil_energy_Hz)
print()
for ii in range(len(J_0)):
    print("J_{} (2\pi Hz):".format(ii), J_0[ii] * recoil_energy_Hz)
print("U (2\pi Hz):", U * recoil_energy_Hz)
print("chi (2\pi Hz):", chi * recoil_energy_Hz)
print("omega (2\pi Hz):", omega * recoil_energy_Hz)
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
    state = evolve(state, H, d_chi_t)

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

if N <= fermi_N_cap:
    print()
    print("Fermi-Hubbard hilbert space dimension:", int(binom(2*product(L),N)))
    c_op_mats = get_c_op_mats(L, N, depth = 2)
    S_op_vec, SS_op_mat = spin_op_vec_mat_FH(L, N, c_op_mats)
    state_z, state_x, state_y = polarized_states_FH(L, N)

    state_free = state_x
    state_drive = state_x
    H_free = H_full(lattice_depth, c_lattice_depth, phi, L, N, c_op_mats)

    beta = 0.90572
    def H_laser(t):
        return -beta * omega * np.cos(omega * t) * S_op_vec[2]

    dt = times[-1] / time_steps
    squeezing_free_vals = np.zeros(time_steps)
    squeezing_drive_vals = np.zeros(time_steps)
    for ii in range(time_steps):
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
