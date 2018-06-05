#!/usr/bin/env python3

# FILE CONTENTS: (numerical) methods to compute overlap integrals

from numpy import *
from numpy.matlib import repmat # to construct a matrix from repeated copies of an array
from numpy.polynomial import hermite # hermite polynomial
from scipy.integrate import quad # numerical integration
from collections import Counter as counter # count unique objects in list

from mathieu_methods import mathieu_solution

##########################################################################################
# on-site wavefunctions
##########################################################################################

def harmonic_wavefunction(z, n, lattic_depth):
    # lattice potential V sin^2(k z) <--> harmonic oscillator 1/2 m w^2 x^2
    # so V k^2 z^2 = 1/2 m w^2 z^2 --> m*w = sqrt(2*V*m) k
    # in lattice units, k = 1 and m = 1/2, so m*w = sqrt(V)
    mw = sqrt(lattice_depth)
    n_vec = zeros(n+1)
    n_vec[-1] = 1
    normalization = 1/sqrt(2**n * math.factorial(n)) * (mw/pi)**(1/4)
    return normalization * exp(-mw*z**2/2) * hermite.hermval(sqrt(mw)*z, n_vec)

def lattice_wavefunction(z, n, momenta, fourier_vecs, neighbor = False):

    i = complex(0,1)
    site_number = len(fourier_vecs[:,0,0])
    fourier_terms = len(fourier_vecs[0,0,:])
    k_offset = int(fourier_terms)//2
    k_values = 2 * (arange(fourier_terms) - k_offset)

    q_phases = repmat(exp(i * momenta * z), fourier_terms, 1).T
    k_phases = repmat(exp(i * k_values * z), site_number, 1)
    phases = q_phases * k_phases

    if neighbor:
        site_shift = repmat(exp(i * momenta * pi), fourier_terms, 1).T
        phases *= site_shift

    normalization = pi * site_number**2
    return sum(fourier_vecs[:,n,:] * phases) / sqrt(normalization)


##########################################################################################
# single-particle overlap integrals
##########################################################################################

# 1-D single-particle wannier orbital kinetic overlap
def kinetic_overlap_1D(momenta, fourier_vecs, nn = 0, mm = None, neighbor_site = False):
    if mm == None: mm = nn

    site_number = len(fourier_vecs[:,0,0])
    fourier_terms = len(fourier_vecs[0,0,:])
    k_offset = int(fourier_terms)//2
    k_values = 2 * (arange(fourier_terms) - k_offset)

    qk_mat = repmat(momenta, fourier_terms, 1).T + repmat(k_values, site_number, 1)

    if not neighbor_site:
        return sum(fourier_vecs[:,nn,:] * fourier_vecs[:,mm,:] * qk_mat**2) / site_number

    else:
        site_shift = repmat(exp(complex(0,1) * momenta * pi), fourier_terms, 1).T
        return sum(fourier_vecs[:,nn,:] * fourier_vecs[:,mm,:] *
                   qk_mat**2 * site_shift) / site_number

# 1-D single-particle wannier orbital lattice overlap
def lattice_overlap_1D(momenta, fourier_vecs, nn = 0, mm = None, neighbor_site = False):
    if mm == None: mm = nn

    site_number = len(fourier_vecs[:,0,0])
    fourier_terms = len(fourier_vecs[0,0,:])
    k_offset = int(fourier_terms)//2
    k_values = 2 * (arange(fourier_terms) - k_offset)

    if not neighbor_site:
        direct = sum(fourier_vecs[:,nn,:] * fourier_vecs[:,mm,:])
        above = sum(fourier_vecs[:,nn,:-1] * fourier_vecs[:,mm,1:])
        below = sum(fourier_vecs[:,nn,1:] * fourier_vecs[:,mm,:-1])

    else:
        site_shift = repmat(exp(complex(0,1) * momenta * pi), fourier_terms, 1).T
        direct = sum(fourier_vecs[:,nn,:] * fourier_vecs[:,mm,:] * site_shift)
        above = sum(fourier_vecs[:,nn,:-1] * fourier_vecs[:,mm,1:] * site_shift[:,1:])
        below = sum(fourier_vecs[:,nn,1:] * fourier_vecs[:,mm,:-1] * site_shift[:,:-1])

    return ( 1/2 * direct - 1/4 * (above + below) ) / site_number

# 1-D nearest-neighbor tunneling rate: - J c_j^\dag c_{j+1} + h.c.
def tunneling_1D(lattice_depth, momenta, fourier_vecs, nn = 0, mm = None):
    kinetic_overlap = kinetic_overlap_1D(momenta, fourier_vecs, nn, mm, True)
    lattice_overlap = lattice_overlap_1D(momenta, fourier_vecs, nn, mm, True)
    return -real(kinetic_overlap + lattice_depth * lattice_overlap)


##########################################################################################
# two-particle overlap integrals
##########################################################################################

# 1-D two-particle wavefunction overlap integral K^{nn mm}_{ll kk}
# if neighbor_site is True, overlap will be between two atoms in neighboring lattice sites
def pair_overlap_1D(momenta, fourier_vecs, nn = 0, mm = 0, ll = 0, kk = 0,
                    neighbors = 0, subinterval_limit = 500):
    if (nn + mm + ll + kk) % 2 != 0: return 0 # odd integrals vanish
    assert(neighbors in [0,1,2]) # 3 and 4 are the same as 1 and 0
    if nn < mm: nn, mm = mm, nn # enforce nn >= mm
    if ll < kk: ll, kk = kk, ll # enforce ll >= kk

    # the wannier orbitals are
    #   \phi_nn(z) = \sum_{q,k} c_{qk}^{(nn)} e^{i(q+2k)z} = c_nn \cdot E
    # where c_nn and E are Q x K matrices (for Q quasimomenta and K fourier components)
    #   - c_nn contains the fourier coefficients c_{qk}^{(nn)}
    #   - and E contains the phases e^{i(q+2k)z}
    # and \cdot denotes the scalar product, much like (a,b) \cdot (x,y) = ax + by
    i = complex(0,1)
    site_number = len(momenta)
    fourier_terms = len(fourier_vecs[0,0,:])
    k_max = fourier_terms // 2
    k_values = 2 * (arange(fourier_terms) - k_max)
    def integrand(z):
        q_phases = repmat(exp(i * momenta * z), fourier_terms, 1).T
        k_phases = repmat(exp(i * k_values * z), site_number, 1)
        phases = q_phases * k_phases
        phases_kk = phases
        phases_ll = phases
        if neighbors != 0:
            site_shift = repmat(exp(i * momenta * pi), fourier_terms, 1).T
            neighbor_phases = phases * site_shift
            phases_kk = neighbor_phases
            phases_ll = neighbor_phases if neighbors == 2 else phases
        return real(conj(sum(fourier_vecs[:,nn,:] * phases) *
                         sum(fourier_vecs[:,mm,:] * phases)) *
                        sum(fourier_vecs[:,ll,:] * phases_ll) *
                        sum(fourier_vecs[:,kk,:] * phases_kk))

    half_length = pi * site_number / 2
    shift = 0 if (neighbors == 0) else -pi/2
    normalization = pi * site_number**2
    integral = quad(integrand, -half_length-shift, half_length-shift,
                    limit = subinterval_limit)[0]
    return integral / normalization**2

def harmonic_pair_overlap_1D(lattice_depth, nn = 0, mm = 0, ll = 0, kk = 0,
                             neighbors = 0, subinterval_limit = 500):
    if (nn + mm + ll + kk) % 2!= 0: return 0 # odd integrals vanish
    assert(neighbors in [0,1,2]) # 3 and 4 are the same as 1 and 0

    site_shift = 0 if not neighbors else pi
    def integrand(z):
        shift_kk = site_shift if neighbors > 0 else 0
        shift_ll = site_shift if neighbors > 1 else 0
        return ( harmonic_wavefunction(z, nn, lattice_depth) *
                 harmonic_wavefunction(z, mm, lattice_depth) *
                 harmonic_wavefunction(z + shift_ll, ll, lattice_depth) *
                 harmonic_wavefunction(z + shift_kk, kk, lattice_depth) )

    # mass * trap frequency in lattice units (explained in "harmonic_wavefunction" method)
    mw = sqrt(lattice_depth)
    # all wavefunctions decay exponentially past the position
    #   z_n at which 1/2 m w^2 z_n^2 = E_n = w*(n+1/2),
    # where n = max(nn,mm,ll,kk)
    # working it out, we get z_n = sqrt( (2*n+1) / (m*w) )
    # we integrate two lattice sites (lattice constant = pi) past this position
    z_max = sqrt( (2*max(nn,mm,ll,kk) + 1) / mw ) + 2*pi

    return quad(integrand, -z_max, z_max, limit = subinterval_limit)[0]

# ground-state finite-range overlap integral
#   i.e. Eq. 28 in johnson2012effective, but without the factor of 4*pi
def momentum_coupling_overlap_3D(momenta, fourier_vecs, subinterval_limit = 500):
    i = complex(0,1)
    site_number = len(momenta)
    fourier_terms = len(fourier_vecs[0,0,:])
    k_max = fourier_terms // 2
    k_values = 2 * (arange(fourier_terms) - k_max)
    def integrand(z):
        q_phases = repmat(exp(i * momenta * z), fourier_terms, 1).T
        k_phases = repmat(exp(i * k_values * z), site_number, 1)
        phases = q_phases * k_phases
        qk_mat = repmat(momenta, fourier_terms, 1).T + repmat(k_values, site_number, 1)
        phi_z = real(sum(fourier_vecs[:,0,:] * phases))
        d_phi_z = -imag(sum(fourier_vecs[:,0,:] * phases * qk_mat))
        dd_phi_z = -real(sum(fourier_vecs[:,0,:] * phases * qk_mat**2))
        return phi_z**2 * ( d_phi_z**2 - phi_z * dd_phi_z )

    half_length = pi * site_number / 2
    normalization = pi * site_number**2
    integral = 2 * quad(integrand, 0, half_length, limit = subinterval_limit)[0]
    overlap_1D = pair_overlap_1D(momenta, fourier_vecs)
    return 3/2 * overlap_1D**2 * integral / normalization**2


##########################################################################################
# three-particle overlap integrals
##########################################################################################

# 1-D three-particle overlap with lattice for a given depth
def triplet_ground_overlap_1D(lattice_depth, bands, site_number, subinterval_limit = 500):
    momenta, fourier_vecs, _ = mathieu_solution(lattice_depth, bands, site_number)

    i = complex(0,1)
    site_number = len(momenta)
    fourier_terms = len(fourier_vecs[0,0,:])
    k_max = fourier_terms // 2
    k_values = 2 * (arange(fourier_terms) - k_max)
    normalized_fourier_vecs = fourier_vecs
    def integrand(z):
        q_phases = repmat(exp(i * momenta * z), fourier_terms, 1).T
        k_phases = repmat(exp(i * k_values * z), site_number, 1)
        phases = real(q_phases * k_phases)
        return sum(normalized_fourier_vecs[:,0,:] * phases)**6

    half_length = pi * site_number / 2
    normalization = pi * site_number**2
    integral = quad(integrand, -half_length, half_length, limit = subinterval_limit)[0]
    return integral / normalization**3
