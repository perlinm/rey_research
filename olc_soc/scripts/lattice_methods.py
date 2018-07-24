#!/usr/bin/env python3

import numpy as np
from numpy.matlib import repmat
import scipy.integrate

from mathieu_methods import mathieu_solution
from overlap_methods import pair_overlap_1D
from sr87_olc_constants import k_clock_LU


# compute a numerical integral over the entire lattice
# integrant must obey symmetries of momentum-space wavefunctions
def symmetric_integral(integrand, site_number, subinterval_limit = 500):
    lattice_length = np.pi * site_number

    def real_integrand(z): return np.real(integrand(z))
    def imag_integrand(z): return np.imag(integrand(z))

    real_half = scipy.integrate.quad(real_integrand, 0, lattice_length/2,
                                     limit = subinterval_limit)
    imag_half = scipy.integrate.quad(imag_integrand, -lattice_length/4, lattice_length/4,
                                     limit = subinterval_limit)

    return 2 * (real_half[0] + 1j * imag_half[0])


##########################################################################################
# lattice commensurability methods
##########################################################################################

# shift crystal momentum q (in units of k_lattice) to first brillouin zone
def shift_q(q):
    return q - round(q/2) * 2

# discrete index corresponding to the momentum closest to q (in the first brillouin zone)
def q_index(q, site_number, symmetric = True):
    shift = -1/site_number if symmetric else 0
    return int(round((q+1+shift)/2 * site_number)) % site_number

# make the momentum q commensurate with the lattice
def lattice_q(q, momenta):
    q_brl = momenta[q_index(q, len(momenta))]
    return q_brl + round(q - q_brl)

# shift fourier vectors appropriately for quasi-momenta outside the first brillouin zone
def shift_vecs(q, vecs):
    while q >= 1:
        vecs = np.roll(vecs, -1) # "rotate" vectors to the left
        vecs[-1] = 0 # clear rightmost entry
        q -= 2
    while q < -1:
        vecs = np.roll(vecs, 1) # "rotate" vectors to the right
        vecs[0] = 0 # clear leftmost entry
        q += 2
    return vecs

# get fourier vectors corresponing to quasimomentum q and band n
def qn_vectors(q, n, momenta, fourier_vecs):
    q = lattice_q(q,momenta)
    qi = q_index(q,len(momenta))
    n_vecs = fourier_vecs[qi,n,:]
    return shift_vecs(q, n_vecs)


##########################################################################################
# lattice energies
##########################################################################################

def qn_energy(q, n, energies, detuning_frequency = 0):
    if detuning_frequency == 0:
        return energies[q_index(q,np.shape(energies)[0]),n]
    else:
        base_energy = energies[q_index(0,np.shape(energies)[0]),0]
        energy = energies[q_index(q,np.shape(energies)[0]),n]
        k = round((energy-base_energy) / detuning_frequency)
        return energy - k * detuning_frequency

def qns_energy(q, n, s, energies, detuning_frequency = 0):
    return qn_energy(q+s*k_clock_LU/2, n, energies, detuning_frequency)

def qns_energy_gap(q, n, s, energies):
    return qns_energy(q, n+1, -s, energies) - qns_energy(q, n, s, energies)


##########################################################################################
# lattice wavefunctions
##########################################################################################

def qn_state_z(q, n, z, momenta, fourier_vecs):
    qi = q_index(q,np.shape(fourier_vecs)[0])
    phase = np.exp(1j * lattice_q(shift_q(q), momenta) * z)

    k_max = len(fourier_vecs[qi,n,:]) // 2
    exponentials = np.exp(2 * 1j * np.arange(-k_max,k_max+1) * z)
    magnitude = sum(fourier_vecs[qi,n,:] * exponentials) / np.sqrt(np.pi * len(momenta))

    return phase * magnitude

def jn_state_z(j, n, z, momenta, fourier_vecs):
    site_number = len(momenta)
    fourier_terms = len(fourier_vecs[0,0,:])
    z_j = np.pi * j

    k_offset = int(fourier_terms) // 2
    k_values = 2 * (np.arange(fourier_terms) - k_offset)
    q_phases = repmat(np.exp(1j * momenta * (z-z_j)), fourier_terms, 1).T
    k_phases = repmat(np.exp(1j * k_values * (z-z_j)), site_number, 1)

    normalization = np.sqrt(np.pi) * site_number
    return np.sum(fourier_vecs[:,n,:] * q_phases * k_phases) / normalization


##########################################################################################
# sanity checks: single-particle wavefunction and laser-induced overlap integrals
##########################################################################################

# numerical wavefunction overlap integral: < qn | gm >
def numerical_inner_product(q, n, g, m, momenta, fourier_vecs):
    def integrand(z):
        return ( np.conj(qn_state_z(q, n, z, momenta, fourier_vecs))
                 * qn_state_z(g, m, z, momenta, fourier_vecs) )
    return symmetric_integral(integrand, np.shape(fourier_vecs)[0])

# numerical laser overlap integral: < gm\bar s | e^{-iskz} | qns >
def numerical_laser_overlap(q, n, s, g, m, momenta, fourier_vecs):
    sk = s*lattice_q(k_clock_LU, momenta)
    def integrand(z):
        return ( np.conj(qn_state_z(q, n, z, momenta, fourier_vecs))
                 * np.exp(-1j * sk * z)
                 * qn_state_z(g, m, z, momenta, fourier_vecs) )
    return symmetric_integral(integrand, len(momenta))


##########################################################################################
# single-particle vector overlap methods
##########################################################################################

# overlap between bloch functions for which the overall phase has been cancelled
def vector_overlap(q, n, g, m, momenta, fourier_vecs):
    return ( qn_vectors(q, n, momenta, fourier_vecs)
             @ qn_vectors(g, m, momenta, fourier_vecs) )

# \Omega^{qs}_{nm}: < qm\bar s | e^{-isk} | qns >
# for **gauge transformed** quasimomenum q
def laser_overlap(q, s, n, m, momenta, fourier_vecs):
    return vector_overlap(q+s*k_clock_LU/2, n, q-s*k_clock_LU/2, m, momenta, fourier_vecs)

# lattice modulation overlap integral: (1/2) * < qms | cos(2*z) | qns >
# for **gauge transformed** quasimomenum q
def lattice_overlap(q, s, n, m, momenta, fourier_vecs):
    q += s*k_clock_LU/2
    return 1/4 * ( vector_overlap(q, n, q+2, m, momenta, fourier_vecs) +
                   vector_overlap(q, n, q-2, m, momenta, fourier_vecs) )


##########################################################################################
# two-body overlap methods
##########################################################################################

# two-body overlap quasi-momentum overlap in shallow lattice axis
# overlap for (kk,aa) + (ll,bb) <--> (pp,cc) + (qq,dd) coupling,
# where kk, ll, pp, qq are quasi-momenta, and aa, bb, cc, dd are band indices
def full_pair_overlap(momenta, fourier_vecs, kk, ll, pp, qq,
                      aa = 0, bb = 0, cc = 0, dd = 0, subinterval_limit = 500):
    if ( lattice_q(kk,momenta) + lattice_q(ll,momenta)
         - lattice_q(pp,momenta) - lattice_q(qq,momenta) ) != 0:
        return 0 # enforce conservation of momentum
    if (aa + bb + cc + dd) % 2:
        return 0 # enforce conservation of parity

    site_number = len(momenta)
    kk = lattice_q(kk, momenta)
    ll = lattice_q(ll, momenta)
    pp = lattice_q(pp, momenta)
    qq = lattice_q(qq, momenta)

    ka_vecs = qn_vectors(kk, aa, momenta, fourier_vecs)
    lb_vecs = qn_vectors(ll, bb, momenta, fourier_vecs)
    pc_vecs = qn_vectors(pp, cc, momenta, fourier_vecs)
    qd_vecs = qn_vectors(qq, dd, momenta, fourier_vecs)

    fourier_terms = len(fourier_vecs[0,0,:])
    k_max = fourier_terms // 2
    k_values = 2 * (np.arange(fourier_terms) - k_max)
    def integrand(z):
        q_phases = np.exp(1j * (pp + qq - kk - ll) * z)
        k_phases = np.exp(1j * k_values * z)

        phi_ka = ka_vecs @ k_phases
        phi_lb = lb_vecs @ k_phases
        phi_pc = pc_vecs @ k_phases
        phi_qd = qd_vecs @ k_phases

        return q_phases * ( np.conj(phi_ka * phi_lb) * phi_pc * phi_qd )

    # make sure that the integral is real (as it must be)
    overlap = symmetric_integral(integrand, site_number)
    assert(np.imag(overlap) < 1e-15)

    normalization = (np.pi * site_number)**2
    return np.real(overlap) / normalization
