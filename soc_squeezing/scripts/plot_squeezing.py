#!/usr/bin/env python3

import sys, scipy
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar

from mathieu_methods import mathieu_solution
from overlap_methods import pair_overlap_1D, tunneling_1D
from dicke_methods import squeezing_OAT, spin_op_vec_mat_dicke
from fermi_hubbard_methods import sum, prod, get_simulation_parameters, spatial_basis, \
    get_c_op_mats, spin_op_vec_mat_FH, polarized_states_FH, gauged_energy, hamiltonians
from squeezing_methods import spin_squeezing, evolve, val

from sr87_olc_constants import g_int_LU, recoil_energy_NU, recoil_energy_Hz

np.set_printoptions(linewidth = 200)

show = "show" in sys.argv
save = "save" in sys.argv
compute_TAT = len(sys.argv) > 1

figsize = (4,3)
fig_dir = "../figures/"
params = { "text.usetex" : True }
plt.rcParams.update(params)


##########################################################################################
# simulation options
##########################################################################################

L = 30 # lattice sites
U_J_target = 5 # target value of U_int / J_0
h_U_target = 0.05 # target value of h_std / U_int
excited_lifetime_SI = 10 # seconds; lifetime of excited state (from e --> g decay)

site_number = 100 # number of sites in lattice calculations
confining_depth = 200 # lattice depth along confining axis
lattice_depth_bounds = (1,15) # min / max lattice depths we will allow

max_tau = 2 # for simulation: chi * max_time = max_tau * N **(-2/3)
time_steps = 1000 # time steps in simulation

fermi_N_max = 8 # maximum number of atoms for which to run Fermi Hubbard calculations


##########################################################################################
# compute all experimental parameters
##########################################################################################

N = prod(L)
lattice_dim = np.array(L, ndmin = 1).size

# determine primary lattice depth which optimally satisfies our target U_int / J_0
momenta, fourier_vecs, _ = mathieu_solution(confining_depth, 1, site_number)
K_T = pair_overlap_1D(momenta, fourier_vecs)
J_T = tunneling_1D(confining_depth, momenta, fourier_vecs)
def U_J(depth):
    momenta, fourier_vecs, _ = mathieu_solution(depth, 1, site_number)
    J_0 = tunneling_1D(depth, momenta, fourier_vecs)
    K_0 = pair_overlap_1D(momenta, fourier_vecs)
    return g_int_LU[1] * K_0**lattice_dim * K_T**(3-lattice_dim) / J_0
lattice_depth = minimize_scalar(lambda x: abs(U_J(x)-U_J_target),
                                method = "bounded", bounds = lattice_depth_bounds).x

# get simulation parameters and on-site interaction energy
L, J_0, J_T, K_0, K_T, momenta, fourier_vecs, energies = \
    get_simulation_parameters(L, lattice_depth, confining_depth)
U_int = g_int_LU[1] * K_T**(3-lattice_dim) * prod(K_0)

# determine optimal SOC angle using hubbard approximation
def h_std(phi): return 2**(1+lattice_dim/2)*J_0[0]*np.sin(phi/2)
phi = minimize_scalar(lambda x: abs(h_std(x)/U_int-h_U_target),
                      method = "bounded", bounds = (0, np.pi)).x

# SOC field variance (h_std^2), OAT strength (chi), and TAT drive frequency (omega)
soc_field_variance = np.var([ ( gauged_energy(q, 1, phi, L, energies)
                                - gauged_energy(q, 0, phi, L, energies) )
                              for q in spatial_basis(L) ])
chi = soc_field_variance / ( N * (N-1) * U_int / prod(L) )
omega = N * np.sqrt(abs(chi*U_int/prod(L)))

print(r"V_0 (E_R):", lattice_depth)
print(r"V_T (E_R):", confining_depth)
print(r"J_T (2\pi Hz):", J_T * recoil_energy_Hz)
print()
for ii in range(len(J_0)):
    print("J_{} (2\pi Hz):".format(ii), J_0[ii] * recoil_energy_Hz)
print(r"U_int (2\pi Hz):", U_int * recoil_energy_Hz)
print(r"phi/pi:", phi / np.pi)
print(r"chi (2\pi Hz):", chi * recoil_energy_Hz)
print(r"omega (2\pi Hz):", omega * recoil_energy_Hz)
print()
print(r"U_int/J_0:", U_int / J_0[0])
print(r"h_std/U_int:", np.sqrt(soc_field_variance) / U_int)
print()


##########################################################################################
# compute squeezing parameters
##########################################################################################

# convert value to decibels (dB)
def to_dB(x): return 10*np.log10(x)

# determine excited-state decay rate in units of the OAT strength
decay_rate_LU = 1/excited_lifetime_SI / recoil_energy_NU
decay_rate_over_chi = decay_rate_LU / chi

# determine simulation times and the size of a single time step
tau_vals = np.linspace(0, max_tau, time_steps)
chi_times = tau_vals * N**(-2/3)
times = chi_times / chi
times_SI = times / recoil_energy_NU
d_chi_t = chi_times[-1] / time_steps
dt = times[-1] / time_steps

# compute OAT squeezing parameters both with and without decay
sqz_OAT = squeezing_OAT(N, chi_times)
t_opt_OAT = times[sqz_OAT.argmin()]
sqz_opt_OAT = -to_dB(sqz_OAT.min())
print("t_opt_OAT (sec):", t_opt_OAT / recoil_energy_NU)
print("sqz_opt_OAT:", sqz_opt_OAT)

sqz_OAT_D = squeezing_OAT(N, chi_times, decay_rate_over_chi)
t_opt_OAT_D = times[sqz_OAT_D.argmin()]
sqz_opt_OAT_D = -to_dB(sqz_OAT_D.min())
print()
print("t_opt_OAT_D (sec):", t_opt_OAT_D / recoil_energy_NU)
print("sqz_opt_OAT_D:", sqz_opt_OAT_D)

# we only need to go on if we will compute TAT squeezing parameters
if not compute_TAT: exit()
print()

# compute Hamiltonian, spin vector, spin-spin matrix, and initial states for TVF and TAT,
#   exploiting a parity symmetry in both cases to reduce the size of the Hilbert space
S_op_vec, SS_op_mat = spin_op_vec_mat_dicke(N)
S_op_vec = [ X[::2,::2] for X in S_op_vec ]
SS_op_mat = [ [ X[::2,::2] for X in XS ] for XS in SS_op_mat ]

H_TVF = SS_op_mat[1][1] - N/2 * S_op_vec[0]
H_TAT = 1/3 * ( SS_op_mat[1][2] + SS_op_mat[2][1] )

state_TVF = np.zeros(N//2+1)
state_TVF[0] = 1
state_TAT = np.copy(state_TVF)

# compute modified OAT squeezing parameters
sqz_TVF = np.zeros(time_steps)
sqz_TAT = np.zeros(time_steps)
for ii in range(time_steps):
    sqz_TVF[ii] = spin_squeezing(state_TVF, S_op_vec, SS_op_mat, N)

    XX_TAT = np.real(val(SS_op_mat[1][1],state_TAT))
    Z_TAT = np.real(val(S_op_vec[0],state_TAT))
    sqz_TAT[ii] = XX_TAT * N / Z_TAT**2

    state_TVF = evolve(state_TVF, H_TVF, d_chi_t)
    state_TAT = evolve(state_TAT, H_TAT, d_chi_t)

t_opt_TVF = times[sqz_TVF.argmin()]
t_opt_TAT = times[sqz_TAT.argmin()]
sqz_opt_TVF = -to_dB(sqz_TVF.min())
sqz_opt_TAT = -to_dB(sqz_TAT.min())
print("t_opt_TVF (sec):", t_opt_TVF / recoil_energy_NU)
print("sqz_opt_TVF:", sqz_opt_TVF)
print()
print("t_opt_TAT (sec):", t_opt_TAT / recoil_energy_NU)
print("sqz_opt_TAT:", sqz_opt_TAT)

# we do only need to go on if we will make plots
if not (show or save): exit()
print()

# if we have few enough particles, compute FH squeezing
if N <= fermi_N_max:
    hilbert_dim = int(scipy.special.binom(2*prod(L),N))
    print("Fermi-Hubbard hilbert space dimension:", hilbert_dim)

    c_op_mats = get_c_op_mats(L, N, depth = 2)
    S_op_vec, SS_op_mat = spin_op_vec_mat_FH(L, N, c_op_mats)
    state_z, state_x, state_y = polarized_states_FH(L, N)
    H_lat, H_int, H_clock = hamiltonians(N, L, phi, lattice_depth,
                                         confining_depth, c_op_mats)
    H_SS = H_lat - U_int / prod(L) * sum([ SS_op_mat[ii][ii] for ii in range(3) ])
    H_free = H_lat + H_int

    state_SS = state_y
    state_free = evolve(state_z, H_clock, np.pi/2) # state pointing in y
    state_static = evolve(state_free, S_op_vec[0], np.pi/2) # state pointing in x
    state_periodic = state_free.copy() # state pointing in y

    H_static = H_free + chi * N/2 * H_clock
    modulation_index = 0.90572
    def H_periodic(t):
        return H_free + modulation_index * omega * np.cos(omega * t) * H_clock

    counter = 0
    sqz_SS = np.zeros(time_steps)
    sqz_FH_free = np.zeros(time_steps)
    sqz_FH_static = np.zeros(time_steps)
    sqz_FH_periodic = np.zeros(time_steps)
    for ii in range(time_steps):
        if ii * 10 // time_steps >= counter:
            counter += 1
            print("{}/{}".format(ii,time_steps))
        sqz_SS[ii] = spin_squeezing(state_SS, S_op_vec, SS_op_mat, N)
        sqz_FH_free[ii] = spin_squeezing(state_free, S_op_vec, SS_op_mat, N)
        sqz_FH_static[ii] = spin_squeezing(state_static, S_op_vec, SS_op_mat, N)
        sqz_FH_periodic[ii] = spin_squeezing(state_periodic, S_op_vec, SS_op_mat, N)
        state_SS = evolve(state_SS, H_SS, dt)
        state_free = evolve(state_free, H_free, dt)
        state_static = evolve(state_static, H_static, dt)
        state_periodic = evolve(state_periodic, H_periodic(times[ii]+dt/2), dt)


##########################################################################################
# make squeezing plot
##########################################################################################

plt.figure(figsize = figsize)

if L.size == 1: L_text = str(L[0])
else: L_text = "(" + ",".join([ str(L_j) for L_j in L ]) + ")"
plt.title(r"$L={},~U/J={}$".format(L_text,U_J_target))

plt.plot(times_SI, -to_dB(sqz_OAT), label = r"OAT")
if N > fermi_N_max:
    plt.gca().set_prop_cycle(None) # reset color cycle
    plt.plot(times_SI, -to_dB(sqz_OAT_D), "--")
plt.plot(times_SI, -to_dB(sqz_TVF), label = "TVF")
plt.plot(times_SI, -to_dB(sqz_TAT), label = "TAT")

if N <= fermi_N_max:
    plt.plot(times_SI, -to_dB(sqz_SS), label = "SS")
    plt.plot(times_SI, -to_dB(sqz_FH_free), label = "FH (free)")
    plt.plot(times_SI, -to_dB(sqz_FH_static), label = "FH (static)")
    plt.plot(times_SI, -to_dB(sqz_FH_periodic), label = "FH (periodic)")

plt.xlim(0,times_SI[-1])
plt.ylim(0,plt.gca().get_ylim()[1])
plt.xlabel(r"Time (seconds)")
plt.ylabel(r"Squeezing: $-10\log_{10}(\xi^2)$")
plt.legend(loc = "best")
plt.tight_layout()

if save: plt.savefig(fig_dir + "squeezing_N{}_U{}.png".format(N, U_J_target))
if show: plt.show()
