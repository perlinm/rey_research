#!/usr/bin/env python3

# FILE CONTENTS: plots comparisons of the FH and OAT models for fixed U/J and several N

import sys, scipy
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar

from mathieu_methods import mathieu_solution
from overlap_methods import pair_overlap_1D, tunneling_1D
from dicke_methods import spin_op_vec_mat_dicke, coherent_spin_state
from fermi_hubbard_methods import get_simulation_parameters, spatial_basis, get_c_op_mats, \
    spin_op_vec_mat_FH, polarized_states_FH, gauged_energy, hamiltonians
from squeezing_methods import spin_squeezing, evolve, squeezing_OAT

from sr87_olc_constants import g_int_LU, recoil_energy_NU, recoil_energy_Hz

show = "show" in sys.argv
save = "save" in sys.argv

figsize = (4,3)
fig_dir = "../figures/breakdown/"
params = { "text.usetex" : True }
plt.rcParams.update(params)


##########################################################################################
# simulation options
##########################################################################################

U_J_target = int(sys.argv[1])

N_vals = [ 2, 3, 4 ]
free = True
static = False
periodic = False

half_phi = False

h_U_target = 0.05 # target value of h_std / U_int
excited_lifetime_SI = 10 # seconds; lifetime of excited state (from e --> g decay)

site_number = 100 # number of sites in lattice calculations
confining_depth = 200 # lattice depth along confining axis
lattice_depth_bounds = (1,15) # min / max lattice depths we will allow

max_tau = 2 # for simulation: chi * max_time = max_tau * N **(-2/3)
time_steps = 1000 # time steps in simulation

fermi_N_max = 8 # maximum number of atoms for which to run Fermi Hubbard calculations


##########################################################################################
# determine title, file name, and plot colors
##########################################################################################

assert(int(free) + int(static) + int(periodic) == 1)

title = r"U/J={},~\phi_{{\mathrm{{SOC}}}}=\phi_{{\mathrm{{opt}}}}".format(U_J_target)
if half_phi: title += "/2"

fig_name = "squeezing"
if free: fig_name += "_free"
if static: fig_name += "_static"
if periodic: fig_name += "_periodic"
if half_phi: fig_name += "_half_phi"
fig_name += "_U{}.png".format(U_J_target)

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

##########################################################################################
# compute squeezing and make plots
##########################################################################################

lattice_dim = 1
tau_vals = np.linspace(0, max_tau, time_steps)

plt.figure(figsize = figsize)
plt.title(r"${}$".format(title))

for ii, N in enumerate(N_vals):
    color = colors[ii]

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
        get_simulation_parameters(N, lattice_depth, confining_depth)
    U_int = g_int_LU[1] * K_T**(3-lattice_dim) * np.prod(K_0)

    # determine optimal SOC angle using hubbard approximation
    def h_std(phi): return 2**(1+lattice_dim/2)*J_0[0]*np.sin(phi/2)
    phi = minimize_scalar(lambda x: abs(h_std(x)/U_int-h_U_target),
                          method = "bounded", bounds = (0, np.pi)).x
    if half_phi: phi /= 2

    # SOC field variance (h_std^2), OAT strength (chi), and TAT drive frequency (omega)
    soc_field_variance = np.var([ ( gauged_energy(q, 1, phi, L, energies)
                                    - gauged_energy(q, 0, phi, L, energies) )
                                  for q in spatial_basis(L) ])
    chi = soc_field_variance / ( N * (N-1) * U_int / np.prod(L) )
    omega = N * np.sqrt(abs(chi*U_int/np.prod(L)))

    print(r"V_0 (E_R):", lattice_depth)
    print(r"V_T (E_R):", confining_depth)
    print(r"J_T (2\pi Hz):", J_T * recoil_energy_Hz)
    print()
    for ii, J_ii in enumerate(J_0):
        print("J_{} (2\pi Hz):".format(ii), J_ii * recoil_energy_Hz)
    print(r"U_int (2\pi Hz):", U_int * recoil_energy_Hz)
    print(r"phi/pi:", phi / np.pi)
    print(r"chi (2\pi Hz):", chi * recoil_energy_Hz)
    print(r"omega (2\pi Hz):", omega * recoil_energy_Hz)
    print()
    print(r"U_int/J_0:", U_int / J_0[0])
    print(r"h_std/U_int:", np.sqrt(soc_field_variance) / U_int)
    print()

    # determine simulation times and the size of a single time step
    chi_times = tau_vals * N**(-2/3)
    times = chi_times / chi
    times_SI = times / recoil_energy_NU
    d_chi_t = chi_times[-1] / time_steps
    dt = times[-1] / time_steps

    if free:
        sqz = squeezing_OAT(N, chi_times, in_dB = True)
        print("t_opt_OAT (sec):", times[sqz.argmax()] / recoil_energy_NU)
        print("sqz_opt_OAT:", sqz.max())

    else:
        S_op_vec, SS_op_mat = spin_op_vec_mat_dicke(N)
        state = coherent_spin_state([0,1,0], N)
        if static:
            H = SS_op_mat[0][0] + N/2 * S_op_vec[1]
        if periodic:
            H = 1/3 * ( SS_op_mat[0][0] - SS_op_mat[2][2] )

        sqz = np.zeros(time_steps)
        for ii in range(time_steps):
            sqz[ii] = spin_squeezing(N, state, S_op_vec, SS_op_mat, in_dB = True)
            state = evolve(state, H, d_chi_t)

        print("t_opt_driven (sec):", times[sqz.argmax()] / recoil_energy_NU)
        print("sqz_opt_driven:", sqz.max())
        print()

    plt.plot(times_SI, sqz, "--", color = color)


    hilbert_dim = int(scipy.special.binom(2*np.prod(L),N))
    print("Fermi-Hubbard hilbert space dimension:", hilbert_dim)

    c_op_mats = get_c_op_mats(L, N, depth = 2)
    S_op_vec, SS_op_mat = spin_op_vec_mat_FH(L, N, c_op_mats)
    state_z, state_X, state_y = polarized_states_FH(L, N)
    H_lat, H_int, H_clock = hamiltonians(N, L, phi, lattice_depth, confining_depth, c_op_mats)
    H_free = H_lat + H_int

    state_free = evolve(state_z, H_clock, np.pi/2) # state pointing in y
    state_static = evolve(state_free, S_op_vec[0], np.pi/2) # state pointing in x
    state_periodic = state_free.copy() # state pointing in y

    H_static = H_free + chi * N/2 * H_clock
    modulation_index = 0.90572
    def H_periodic(t):
        return H_free + modulation_index * omega * np.cos(omega * t) * H_clock

    if free:
        H = H_free
        state = state_free
    if static:
        H = H_static
        state = state_static
    if periodic:
        state = state_periodic

    counter = 0
    sqz_FH = np.zeros(time_steps)
    for ii in range(time_steps):
        if ii * 10 // time_steps >= counter:
            counter += 1
            print("{}/{}".format(ii,time_steps))
        sqz_FH[ii] = spin_squeezing(N, state, S_op_vec, SS_op_mat, in_dB = True)
        if periodic: H = H_periodic(times[ii]+dt/2)
        state = evolve(state, H, dt)


    plt.plot(times_SI, sqz_FH, "-", color = color,
             label = r"$N={}$".format(N))

plt.xlim(0,times_SI[-1])
plt.ylim(0,sqz.max()*1.2)
plt.xlabel(r"Time (seconds)")
plt.ylabel(r"Squeezing: $-10\log_{10}(\xi^2)$")

plt.plot([-1],[-1],"k-",label="FH")
plt.plot([-1],[-1],"k--",label="OAT")
plt.legend(loc = "lower right")

plt.tight_layout()

if save: plt.savefig(fig_dir + fig_name)
if show: plt.show()
