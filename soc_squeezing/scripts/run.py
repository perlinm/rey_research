#!/usr/bin/env python3

import sys
from pylab import *

from mathieu_methods import mathieu_solution
from overlap_methods import tunneling_1D, pair_overlap_1D
from sr87_olc_constants import g_int_LU, recoil_energy_NU, recoil_energy_Hz
from squeezing_methods import squeezing_OAT, coherent_spin_state, \
    squeezing_TAT_propagator, spin_squeezing

show = "show" in sys.argv
save = "save" in sys.argv

# lattice bands and site number: only matters for calculations of lattice parameters
bands = 5
site_number = 100

tunneling_dims = 1 # number of dimensions in which to tunnel
lattice_depth = 4 # shallow (tunneling) axis lattice depth
c_lattice_depth = 60 # deep (confining) axis lattice depth
phi = pi/50 # spin-orbit coupling parameter

N = 50 # total number of atoms
L = 100 # total number of lattice sites
eta = N/L # total filling fraction

max_tau = 3 # for simulation: chi * max_time = max_tau * N **(-2/3)
time_steps = 100 # time steps in simulation

c_momenta, c_fourier_vecs, _ = mathieu_solution(c_lattice_depth, bands, site_number)
J_T = tunneling_1D(c_lattice_depth, c_momenta, c_fourier_vecs)
K_T = pair_overlap_1D(c_momenta, c_fourier_vecs)
momenta, fourier_vecs, _ = mathieu_solution(lattice_depth, bands, site_number)
K_0 = pair_overlap_1D(momenta, fourier_vecs)
J_0 = tunneling_1D(lattice_depth, momenta, fourier_vecs)

print("J_T (2\pi Hz):", J_T * recoil_energy_Hz)
print()

if tunneling_dims == 1:
    U = g_int_LU[1] * K_0 * K_T**2
    chi = 8 * J_0**2 * sin(phi/2)**2 / ((N-1) * eta * U)

elif tunneling_dims == 2:
    U = g_int_LU[1] * K_0**2 * K_T
    chi = 16 * J_0**2 * sin(phi/2)**2 / ((N-1) * eta * U)

print("J_0 (2\pi Hz):", J_0 * recoil_energy_Hz)
print("U (2\pi Hz):", U * recoil_energy_Hz)
print("xi (2\pi Hz):", U/L * recoil_energy_Hz)
print("chi (2\pi Hz):", chi * recoil_energy_Hz)
print("omega (2\pi Hz):", N*sqrt(U/L*chi) * recoil_energy_Hz)
print()

tau_vals = np.linspace(0, max_tau, time_steps)
chi_times = tau_vals * N**(-2/3)
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
times_SI = chi_times / chi / recoil_energy_NU
squeezing_OAT_dB = -10*np.log(squeezing_OAT_vals.min())/np.log(10)
squeezing_TAT_dB = -10*np.log(squeezing_TAT_vals.min())/np.log(10)

print("t_opt_OAT (sec):", t_opt_OAT / recoil_energy_NU)
print("t_opt_TAT (sec):", t_opt_TAT / recoil_energy_NU)
print()
print("OAT squeezing (dB):", squeezing_OAT_dB)
print("TAT squeezing (dB):", squeezing_TAT_dB)


if not show or save: exit()

def to_dB(x): return 10*np.log10(x)

fig_dir = "../figures/"
figsize = (3,2)
params = { "text.usetex" : True }
plt.rcParams.update(params)

plt.figure(figsize = figsize)
plt.plot(times_SI, to_dB(squeezing_OAT_vals), label = "OAT")
plt.plot(times_SI, to_dB(squeezing_TAT_vals), label = "TAT")
plt.xlim(0,times_SI[-1])
plt.xlabel(r"Time (seconds)")
plt.ylabel(r"Squeezing: $10\log_{10}(\xi^2)$")
plt.legend(loc="best")
plt.tight_layout()

if save: plt.savefig(fig_dir + "squeezing_comparison.pdf")
if show: plt.show()
