#!/usr/bin/env python3

import sys, scipy
import numpy as np

from scipy.optimize import minimize_scalar

from mathieu_methods import mathieu_solution
from overlap_methods import pair_overlap_1D, tunneling_1D
from fermi_hubbard_methods import get_simulation_parameters, gauged_energy, spatial_basis

from sr87_olc_constants import g_int_LU, recoil_energy_NU, recoil_energy_Hz

np.set_printoptions(linewidth = 200)


L = [10]*2 # lattice sites
U_J_target = 2 # target value of U_int / J_0

site_number = 1000 # number of sites in lattice calculations
lattice_depth_bounds = (1,15) # min / max lattice depths we will allow
confining_depth = 60 # lattice depth along confining axis

h_U_target = 0.05 # target value of h_std / U_int

excited_lifetime_SI = 10 # seconds; lifetime of excited state (from e --> g decay)


##########################################################################################
# compute all experimental parameters
##########################################################################################

N = np.product(L)
eta = N / np.product(L)
lattice_dim = np.array(L, ndmin = 1).size

print("D:", np.size(L))
print("N:", N)
print()

# determine primary lattice depth which optimally satisfies our target U_int / J_0
momenta, fourier_vecs, soc_energies = mathieu_solution(confining_depth, 1, site_number)
K_T = pair_overlap_1D(momenta, fourier_vecs)
J_T = tunneling_1D(confining_depth, momenta, fourier_vecs)
def U_J(depth):
    momenta, fourier_vecs, _ = mathieu_solution(depth, 1, site_number)
    J_0 = tunneling_1D(depth, momenta, fourier_vecs)
    K_0 = pair_overlap_1D(momenta, fourier_vecs)
    U = g_int_LU[1] * K_0**lattice_dim * K_T**(3-lattice_dim)
    return U / J_0
lattice_depth = minimize_scalar(lambda x: abs(U_J(x)-U_J_target),
                                method = "bounded", bounds = lattice_depth_bounds).x

print("V_0 (E_R):", lattice_depth)

# get simulation parameters and on-site interaction energy
L, J_0, J_T, K_0, K_T, momenta, fourier_vecs, energies = \
    get_simulation_parameters(L, lattice_depth, confining_depth, site_number)
U_int = g_int_LU[1] * K_T**(3-lattice_dim) * np.product(K_0)

print("V_T (E_R):", confining_depth)
print("J_T (2\pi Hz):", J_T * recoil_energy_Hz)
print()
for ii in range(len(J_0)):
    print(f"J_{ii} (2\pi Hz):", J_0[ii] * recoil_energy_Hz)
print("U_int (2\pi Hz):", U_int * recoil_energy_Hz)


# determine optimal SOC angle using hubbard approximation
def h_std(phi): return 2**(1+lattice_dim/2)*J_0[0]*np.sin(phi/2)
phi = minimize_scalar(lambda x: abs(h_std(x)/U_int-h_U_target),
                      method = "bounded", bounds = (0, np.pi)).x

print("phi/pi:", phi / np.pi)

# SOC field variance (h_std^2), OAT strength (chi), and TAT drive frequency (omega)
soc_field_variance = np.var([ ( gauged_energy(q, 1, phi, L, energies)
                                - gauged_energy(q, 0, phi, L, energies) )
                              for q in spatial_basis(L) ])
chi = soc_field_variance / ( (N-1) * eta * U_int )
omega = np.sqrt(abs(eta*U_int*chi*N))

chi_NU = chi * recoil_energy_NU

print("chi (2\pi Hz):", chi * recoil_energy_Hz)
print("omega (2\pi Hz):", omega * recoil_energy_Hz)
print()
print("U_int/J_0:", U_int / J_0[0])
print("h_std/U_int:", np.sqrt(soc_field_variance) / U_int)

# get decay rate info
decay_rate_LU = 1/excited_lifetime_SI / recoil_energy_NU
decay_rate = decay_rate_LU / chi

print()
print("decay_rate/chi:", decay_rate)

