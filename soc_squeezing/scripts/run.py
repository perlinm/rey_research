#!/usr/bin/env python3

from pylab import *
import scipy.linalg as linalg

from mathieu_methods import mathieu_solution
from overlap_methods import tunneling_1D, pair_overlap_1D
from sr87_olc_constants import g_int_LU, recoil_energy_NU, recoil_energy_Hz
from squeezing_methods import squeezing_OAT, squeezing_TAT

c_lattice_depth = 60
lattice_depth = 4
bands = 5
site_number = 100


c_momenta, c_fourier_vecs, c_energies = mathieu_solution(c_lattice_depth, bands, site_number)
J_T = tunneling_1D(c_lattice_depth, c_momenta, c_fourier_vecs)
K_T = pair_overlap_1D(c_momenta, c_fourier_vecs)
print("J_T (2\pi Hz):", J_T * recoil_energy_Hz)
print()


depth = 4
site_number = 100 # per axis
eta = 1/2 # filling fraction

N = int(round(site_number * eta))
L = site_number

phi = pi/50

momenta, fourier_vecs, _ = mathieu_solution(depth, bands, site_number)

K_0 = pair_overlap_1D(momenta, fourier_vecs)
J_0 = tunneling_1D(depth, momenta, fourier_vecs)
U = g_int_LU[1] * K_0 * K_T**2

chi = 8 * J_0**2 * sin(phi/2)**2 / ((N-1) * eta * U)

tau_vals = np.linspace(0, 3, 100)
chi_times = tau_vals * N**(-2/3)
squeezing_OAT_vals = np.vectorize(squeezing_OAT)(chi_times, N)
squeezing_TAT_vals = np.vectorize(squeezing_TAT)(chi_times, N)

tau_opt_OAT = tau_vals[squeezing_OAT_vals.argmin()]
tau_opt_TAT = tau_vals[squeezing_TAT_vals.argmin()]

t_opt_OAT = tau_opt_OAT * N**(-2/3) / chi
t_opt_TAT = tau_opt_TAT * N**(-2/3) / chi
times_SI = chi_times / chi / recoil_energy_NU
squeezing_OAT_dB = -10*np.log(squeezing_OAT_vals.min())/np.log(10)
squeezing_TAT_dB = -10*np.log(squeezing_TAT_vals.min())/np.log(10)

print("J_0 (2\pi Hz):", J_0 * recoil_energy_Hz)
print("U (2\pi Hz):", U * recoil_energy_Hz)
print("xi (2\pi Hz):", U/L * recoil_energy_Hz)
print("chi (2\pi Hz):", chi * recoil_energy_Hz)
print("t_opt_OAT (sec):", t_opt_OAT / recoil_energy_NU)
print("t_opt_TAT (sec):", t_opt_TAT / recoil_energy_NU)
print()
print("OAT squeezing (dB):", squeezing_OAT_dB)
print("TAT squeezing (dB):", squeezing_TAT_dB)

exit()

fig_dir = "../figures/"
figsize = (3,2)
params = { "text.usetex" : True }
plt.rcParams.update(params)

plt.figure(figsize = figsize)
plt.plot(times_SI, -10*np.log(squeezing_OAT_vals)/np.log(10), label = "OAT")
plt.plot(times_SI, -10*np.log(squeezing_TAT_vals)/np.log(10), label = "TAT")
plt.xlim(0,times_SI[-1])
plt.xlabel(r"Time (seconds)")
plt.ylabel(r"Squeezing: $-10\log_{10}(\xi^2)$")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(fig_dir + "squeezing_comparison.pdf")
