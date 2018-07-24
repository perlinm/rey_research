#!/usr/bin/env python3

# FILE CONTENTS: (numerical) methods relating to interaction Hamiltonians on a 3-D lattice

import numpy as np
import sympy as sym
from scipy.special import zeta
from itertools import product as cartesian_product
from itertools import combinations, permutations
from scipy.stats.mstats import gmean

from mathieu_methods import mathieu_solution
from overlap_methods import pair_overlap_1D, tunneling_1D, momentum_coupling_overlap_3D
from sr87_olc_constants import k_lattice_AU, m_SR87_AU, m_SR87_LU, C6_AU

# return real and positive roots of a null equation of one variable
def real_positive_roots(null_expression, variable, precision):
    return [ float(root.as_real_imag()[0])
             for root in sym.solve(null_expression, variable)
             if abs(root.as_real_imag()[1]) < precision
             if root.as_real_imag()[0] > 0 ]

# compute energy correction coefficients at given order in perturbation theory (pt_order)
#   all quantities in "lattice units" with lattice wavenumber and recoil energy set to 1
def energy_correction_coefficients(lattice_depths, site_number,
                                   pt_order = 3, bands = 13):
    assert(pt_order >= 1)

    momenta_x, fourier_vecs_x, _ = mathieu_solution(lattice_depths[0], bands, site_number)
    momenta_y, fourier_vecs_y, _ = mathieu_solution(lattice_depths[1], bands, site_number)
    momenta_z, fourier_vecs_z, _ = mathieu_solution(lattice_depths[2], bands, site_number)
    a_2_1 = ( pair_overlap_1D(momenta_x, fourier_vecs_x) *
              pair_overlap_1D(momenta_y, fourier_vecs_y) *
              pair_overlap_1D(momenta_z, fourier_vecs_z) )

    momenta_list = [ momenta_x, momenta_y, momenta_z ]
    fourier_vecs_list = [ fourier_vecs_x, fourier_vecs_y, fourier_vecs_z ]
    a_prime_2_1 = momentum_coupling_overlap_3D(momenta_list, fourier_vecs_list)

    if pt_order == 1:
        return [ a_2_1, a_prime_2_1 ]

    x, y, z = 0, 1, 2 # axis indices
    lattice_depths = np.sort(lattice_depths)

    # compute all 1-D band energies, spatial wavefunction overlaps, and tunneling rates
    band_energies = np.zeros((3,bands))
    K_1D = np.zeros((3,bands,bands))
    if pt_order > 2:
        K_t_1D = np.zeros((3,bands,bands))
        t_1D = np.zeros((3,bands))
    for axis in range(3):
        # if any of the lattice depths are the same, we can recycle our calculations
        if axis > 0 and lattice_depths[axis] == lattice_depths[axis-1]:
            band_energies[axis,:] = band_energies[axis-1,:]
            K_1D[axis,:,:] = K_1D[axis-1,:,:]
            if pt_order > 2:
                K_t_1D[axis,:,:] = K_t_1D[axis-1,:,:]
                t_1D[axis,:] = t_1D[axis-1,:]
            continue

        # otherwise compute everything for this lattice depth
        momenta, fourier_vecs, energies = mathieu_solution(lattice_depths[axis],
                                                           bands, site_number)
        band_energies[axis,:] = np.mean(energies,0)
        band_energies[axis,:] -= band_energies[axis,0]

        for nn in range(bands):
            if pt_order > 2:
                t_1D[axis,nn] = tunneling_1D(lattice_depths[axis],
                                             momenta, fourier_vecs, nn)
            for mm in range(nn,bands):
                K_1D[axis,nn,mm] = pair_overlap_1D(momenta, fourier_vecs, nn, mm)
                K_1D[axis,mm,nn] = K_1D[axis,nn,mm]
                if pt_order > 2:
                    K_t_1D[axis,nn,mm] = pair_overlap_1D(momenta, fourier_vecs, nn, mm,
                                                         neighbors = 1)
                    K_t_1D[axis,mm,nn] = K_t_1D[axis,nn,mm]

    K = prod(K_1D[:,0,0])

    if pt_order == 2:

        # collect all (even) band indices to loop over
        even_bands = [ (n_x, n_y, n_z)
                       for n_z in range(0,bands,2)
                       for n_y in range(n_z,bands,2)
                       for n_x in range(n_y,bands,2) ]

        # second order spatial overlap factor
        a_3_2 = 0
        for n_x, n_y, n_z in even_bands[1:]:
            K_n = K_1D[x,0,n_x] * K_1D[y,0,n_y] * K_1D[z,0,n_z]
            E_n = band_energies[x,n_x] + band_energies[y,n_y] + band_energies[z,n_z]
            a_3_2 += K_n**2 / E_n * len(set(permutations([n_x,n_y,n_z])))

            return a_2_1, a_prime_2_1, a_3_2

    # collect all band indices to loop over
    all_bands = ( (n_x, n_y, n_z)
                  for n_z in range(bands)
                  for n_y in range(n_z,bands)
                  for n_x in range(n_y,bands) )

    # second and third order spatial overlap factors
    a_3_2, a_3_3, a_4_3_1, a_4_3_2, a_4_3_3, a_5_3, g_2_2, g_3_1_2, g_3_2_2 = np.zeros(9)
    for n_x, n_y, n_z in all_bands:
        n_permutations = len(set(permutations([n_x,n_y,n_z])))
        E_n = band_energies[x,n_x] + band_energies[y,n_y] + band_energies[z,n_z]
        K_n = K_1D[x,0,n_x] * K_1D[y,0,n_y] * K_1D[z,0,n_z]

        parity_x = n_x % 2
        parity_y = n_y % 2
        parity_z = n_z % 2

        if E_n > 0 and (parity_x + parity_y + parity_z == 0):
            K_t_n = K_n * (K_t_1D[x,0,n_x] / K_1D[x,0,n_x] +
                           K_t_1D[y,0,n_y] / K_1D[y,0,n_y] +
                           K_t_1D[z,0,n_z] / K_1D[z,0,n_z]) / 3
            t_n = (t_1D[x,n_x] + t_1D[y,n_y] + t_1D[z,n_z]) / 3

            a_3_2 += K_n**2 / E_n * n_permutations
            a_5_3 += K_n**2 / E_n**2 * n_permutations # missing factor of K
            g_3_1_2 += K_n * K_t_n / E_n * n_permutations
            g_3_2_2 += K_n**2 * t_n / E_n**2 * n_permutations


        for m_x, m_y, m_z in cartesian_product(range(parity_x,bands,2),
                                               range(parity_y,bands,2),
                                               range(parity_z,bands,2)):

            E_m = band_energies[x,m_x] + band_energies[y,m_y] + band_energies[z,m_z]
            if E_n + E_m == 0: continue

            K_m = K_1D[x,0,m_x] * K_1D[y,0,m_y] * K_1D[z,0,m_z]
            K_n_m = K_1D[x,n_x,m_x] * K_1D[y,n_y,m_y] * K_1D[z,n_z,m_z]
            K_t_n_m = K_n_m * (K_t_1D[x,n_x,m_x] / K_1D[x,n_x,m_x] +
                               K_t_1D[x,n_y,m_y] / K_1D[x,n_y,m_y] +
                               K_t_1D[x,n_z,m_z] / K_1D[x,n_z,m_z]) / 3

            if E_m > 0:
                a_4_3_1 += K_n_m * K_n * K_m / ( (E_n+E_m) * E_m ) * n_permutations

                if E_n > 0:
                    a_4_3_2 += K_n_m * K_n * K_m / (E_n * E_m) * n_permutations

            a_4_3_3 += K_n_m**2 / (E_n+E_m)**2 * n_permutations # missing factor of K
            g_2_2 += K_n_m * K_t_n_m / (E_n+E_m) * n_permutations

            for l_x, l_y, l_z in cartesian_product(range(parity_x,bands,2),
                                                   range(parity_y,bands,2),
                                                   range(m_z,bands,2)):

                E_l = band_energies[x,l_x] + band_energies[y,l_y] + band_energies[z,l_z]
                if E_n + E_l == 0: continue

                K_n_l = K_1D[x,n_x,l_x] * K_1D[y,n_y,l_y] * K_1D[z,n_z,l_z]
                K_m_l = K_1D[x,m_x,l_x] * K_1D[y,m_y,l_y] * K_1D[z,m_z,l_z]

                prefactor = (-1)**(l_x+l_y+l_z) * n_permutations
                if l_z != m_z: prefactor *= 2
                a_3_3 += prefactor * K_n_m * K_m_l * K_n_l / ( (E_n+E_m) * (E_n+E_l) )

    a_4_3_3 *= K
    a_5_3 *= K

    return a_2_1, a_prime_2_1, a_3_2, a_3_3, a_4_3_1, a_4_3_2, a_4_3_3, \
        a_5_3, g_2_2, g_3_1_2, g_3_2_2

##########################################################################################
# methods related to renormalization of coupling constants
##########################################################################################

# renormalize coupling constant for two atoms using analytical results from busch1998two
# specifically, we use Eq. 16 with both sides of equation divided by sqrt(2/pi)
# calculation modified to use two-body overlap integral in a lattice
# if backwards == True, convert a renormalized scattering length into a free-space one
def renormalized_coupling(coupling, lattice_depths,
                          bands = 15, site_number = 121,
                          precision = 10, backwards = False, harmonic = False):

    mean_depth = gmean(lattice_depths) # geometric mean of lattice depths
    w_eff = np.sqrt(2 * mean_depth / m_SR87_LU) # effective angular harmonic trap frequency

    # two-atom ground-state overlap integral
    if not harmonic:
        overlap_factor = 1
        for depth in lattice_depths:
            momenta, fourier_vecs, _ = mathieu_solution(depth, bands, site_number)
            overlap_factor *= pair_overlap_1D(momenta, fourier_vecs)
    else:
        # determine ground-state two-body overlap integral in a harmonic oscillator
        HO_length = mean_depth**(-1/4) # harmonic oscillator length in lattice units
        overlap_factor = np.sqrt(2/np.pi) / (4*np.pi) / HO_length**3

    unknown_coupling = sym.symbols("G")
    if not backwards: # compute an in-trap scattering length from a harmonic one
        E_free = coupling * overlap_factor / w_eff
        E_lattice = unknown_coupling * overlap_factor / w_eff
    else: # compute harmonic scattering length from an in-trap one
        E_free = unknown_coupling * overlap_factor / w_eff
        E_lattice = coupling * overlap_factor / w_eff

    # coefficients of series expansion of the right hand side of our expression
    c_0 = 1
    c_1 = 1 - np.log(2)
    c_2 = -np.pi**2/24 - np.log(2) + 1/2 * np.log(2)**2
    series = c_0 + c_1 * E_lattice + c_2 * E_lattice**2
    null_expression = 1/E_free - 1/E_lattice * series

    return min(real_positive_roots(null_expression, unknown_coupling, precision),
               key = lambda x: abs(x/coupling-1))

# effective coupling constant for momentum-dependent interactions
def momentum_coupling(coupling_LU, C6_AU):
    a_LU = coupling_LU / (4*np.pi/m_SR87_LU) # scattering length
    a_AU = a_LU / k_lattice_AU # scattering length in atomic units

    gamma_ratio = sym.gamma(3/4) / sym.gamma(1/4)
    chi = gamma_ratio * (4 * m_SR87_AU * C6_AU)**(1/4) / a_AU

    r_eff_LU = 1/3 * gamma_ratio**(-2) * chi * (1 - 2*chi + 2*chi**2) * a_LU
    return (4*np.pi/m_SR87_LU) * (1/2 * r_eff_LU * a_LU**2)

# extract two-body excited-state coupling constant from excitation energy
def excited_state_coupling(excitation_energy, a_2_1, a_prime_2_1, coupling_gg,
                           precision = 10):

    unknown_coupling = sym.symbols("G")

    # primed coupling constants
    coupling_prime_gg = momentum_coupling(coupling_gg, C6_AU[0])
    unknown_coupling_prime = momentum_coupling(unknown_coupling, C6_AU[1])

    # solve expression for excitation energy
    null_expression = ( a_2_1 * (unknown_coupling - coupling_gg)
                        + a_prime_2_1 * (unknown_coupling_prime - coupling_prime_gg)
                        - excitation_energy )
    return real_positive_roots(null_expression, unknown_coupling, precision)[0]
