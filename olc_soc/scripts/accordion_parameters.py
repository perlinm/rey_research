#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import itertools

from scipy.integrate import quad, solve_ivp

from mathieu_methods import mathieu_solution
from overlap_methods import pair_overlap_1D, tunneling_1D, lattice_wavefunction
from sr87_olc_constants import g_int_LU, recoil_energy_Hz

from fermion_methods import f_op, sum_ops, mul_ops, rank_comb, restrict_matrix

from time import time
start_time = time()

np.set_printoptions(linewidth = 200)

V_P = 2.5 # "primary" lattice depth in units of "regular" recoil energy
V_T = 50 # "transverse" lattice depths in units of "regular" recoil energy
stretch_P = 1.7 # "stretch factor" for accordion lattice along primary axis
stretch_T = 1.4 # "stretch factor" for accordion lattice along transverse axis
tilt = 10e3 # 2\pi Hz per lattice site
magnetic_field = 500 # Gauss; for nuclear spin splitting
rabi_frequency = 1e3 # 2\pi Hz; clock laser

relevance_cutoff = 1e-2 # for reducing "relevant" Hilbert space

# number of lattice bands and lattice sites to use in mathieu equation solver
bands, site_number = 5, 100


nuclear_splitting_per_Gauss = 110 # 2\pi Hz / Gauss / nuclear spin
nuclear_splitting = magnetic_field * nuclear_splitting_per_Gauss



# numerical integral over the entire lattice
def lattice_integral(integrand, site_number, subinterval_limit = 500):
    lattice_length = np.pi * site_number

    def real_integrand(z): return np.real(integrand(z))
    def imag_integrand(z): return np.imag(integrand(z))

    real_half = quad(real_integrand, -lattice_length/2, lattice_length/2,
                     limit = subinterval_limit)
    imag_half = quad(imag_integrand, -lattice_length/2, lattice_length/2,
                     limit = subinterval_limit)
    return real_half[0] + 1j * imag_half[0]

# compute laser-induced overlap of localized wavefunctions
def laser_overlap(site_phase, momenta, fourier_vecs, site_shift = 0):
    def integrand(z):
        return ( np.conj(lattice_wavefunction(z, 0, momenta, fourier_vecs))
                 * np.exp(1j * site_phase/np.pi * z)
                 * lattice_wavefunction(z, 0, momenta, fourier_vecs, site_shift) )
    return lattice_integral(integrand, len(momenta))


# get bare and laser-induced tunneling rates
def get_J_O(depth, stretch):
    depth_S = depth * stretch**2
    momenta_S, fourier_vecs_S, _ = mathieu_solution(depth_S, bands, site_number)
    tunneling_rate_S = tunneling_1D(depth_S, momenta_S, fourier_vecs_S)
    tunneling_rate = tunneling_rate_S / stretch**2
    tunneling_rate *= recoil_energy_Hz

    rabi_ratio = abs( laser_overlap(np.pi, momenta_S, fourier_vecs_S, 1) /
                      laser_overlap(np.pi, momenta_S, fourier_vecs_S) )

    return tunneling_rate, rabi_ratio * rabi_frequency


depths = np.arange(2,7.01,0.25)
stretches = np.arange(1,3.01,0.25)
for depth, stretch in itertools.product(depths, stretches):
    if depth != 2: continue
    print(depth,stretch,*get_J_O(depth,stretch))
