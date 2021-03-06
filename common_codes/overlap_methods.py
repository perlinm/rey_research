#!/usr/bin/env python3

# FILE CONTENTS: (numerical) methods to compute overlap integrals

import numpy as np
from numpy.matlib import repmat # to construct a matrix from repeated copies of an array
from numpy.polynomial import hermite # hermite polynomial
from scipy.integrate import quad # numerical integration
from collections import Counter as counter # count unique objects in list

import math


##########################################################################################
# on-site wavefunctions
##########################################################################################

def harmonic_wavefunction(z, n, lattice_depth):
    # lattice potential V sin^2(k z) <--> harmonic oscillator 1/2 m w^2 x^2
    # so V k^2 z^2 = 1/2 m w^2 z^2 --> m*w = sqrt(2*V*m) k
    # in lattice units, k = 1 and m = 1/2, so m*w = sqrt(V)
    mw = np.sqrt(lattice_depth)
    n_vec = np.zeros(n+1)
    n_vec[-1] = 1
    normalization = 1/np.sqrt(2**n * math.factorial(n)) * (mw/np.pi)**(1/4)
    return normalization * np.exp(-mw*z**2/2) * hermite.hermval(np.sqrt(mw)*z, n_vec)

def lattice_wavefunction(z, n, momenta, fourier_vecs, site_shift = 0):
    site_number = len(momenta)
    fourier_terms = len(fourier_vecs[0,0,:])
    k_offset = int(fourier_terms)//2
    k_values = 2 * (np.arange(fourier_terms) - k_offset)

    z_j = site_shift * np.pi
    q_phases = repmat(np.exp(1j * momenta * (z-z_j)), fourier_terms, 1).T
    k_phases = repmat(np.exp(1j * k_values * (z-z_j)), site_number, 1)
    phases = q_phases * k_phases

    normalization = np.pi * site_number**2
    return np.sum(fourier_vecs[:,n,:] * q_phases * k_phases) / np.sqrt(normalization)


##########################################################################################
# single-particle overlap integrals
# nn and mm are and band indices
# site_shift is the number of lattice sites between the initial/final states
##########################################################################################

# 1-D single-particle wannier orbital kinetic overlap integral
def kinetic_overlap_1D(momenta, fourier_vecs, nn = 0, mm = None, site_shift = 0):
    if mm == None: mm = nn

    site_number = len(fourier_vecs[:,0,0])
    fourier_terms = len(fourier_vecs[0,0,:])
    k_offset = int(fourier_terms)//2
    k_values = 2 * (np.arange(fourier_terms) - k_offset)

    qk_mat = repmat(momenta, fourier_terms, 1).T + repmat(k_values, site_number, 1)
    site_phases = repmat(np.exp(-1j * momenta * np.pi * site_shift), fourier_terms, 1).T
    return np.sum(fourier_vecs[:,nn,:] * fourier_vecs[:,mm,:] *
                  qk_mat**2 * site_phases) / site_number

# 1-D single-particle wannier orbital lattice overlap integral
def lattice_overlap_1D(momenta, fourier_vecs, nn = 0, mm = None, site_shift = 0):
    if mm == None: mm = nn

    site_number = len(fourier_vecs[:,0,0])
    fourier_terms = len(fourier_vecs[0,0,:])
    k_offset = int(fourier_terms)//2
    k_values = 2 * (np.arange(fourier_terms) - k_offset)

    site_phases = repmat(np.exp(-1j * momenta * np.pi * site_shift), fourier_terms, 1).T
    direct = np.sum(fourier_vecs[:,nn,:] * fourier_vecs[:,mm,:] * site_phases)
    above = np.sum(fourier_vecs[:,nn,:-1] * fourier_vecs[:,mm,1:] * site_phases[:,1:])
    below = np.sum(fourier_vecs[:,nn,1:] * fourier_vecs[:,mm,:-1] * site_phases[:,:-1])

    return ( 1/2 * direct - 1/4 * (above + below) ) / site_number

# 1-D nearest-neighbor tunneling rate: - J \sum_j c_j^\dag c_{j+1} + h.c.
def tunneling_1D(lattice_depth, momenta, fourier_vecs, nn = 0, mm = None, site_shift = 1):
    kinetic_overlap = kinetic_overlap_1D(momenta, fourier_vecs, nn, mm, site_shift)
    lattice_overlap = lattice_overlap_1D(momenta, fourier_vecs, nn, mm, site_shift)
    return -np.real(kinetic_overlap + lattice_depth * lattice_overlap)


##########################################################################################
# two-particle on-site overlap integrals
# kk, ll, mm, nn are band indices
##########################################################################################

# 1-D two-body on-site wavefunction overlap integral
#   K^{kk ll}_{mm nn} = \int \d z \phi_{kk}^* \phi_{ll}^* \phi_{mm} \phi_{nn}
# if "neighbors" is 1, \phi_{nn} is evaluated at an adjacent lattice site
# if "neighbors" is 2, both \phi_{nn} and \phi_{mm} are evaluated at an adjacent site
def pair_overlap_1D(momenta, fourier_vecs, kk = 0, ll = 0, mm = 0, nn = 0,
                    neighbors = 0, padding_sites = None, subinterval_limit = 1000):
    if (kk + ll + mm + nn) % 2 != 0: return 0 # odd integrals vanish
    assert(neighbors in [0,1,2]) # 3 and 4 are the same as 1 and 0
    if padding_sites == None: padding_sites = 2 + max(kk,ll,mm,nn)

    # the wannier orbitals are
    #   \phi_n(z) = \sum_{q,k} c_{qk}^{(n)} e^{i(q+2k)z} = c_n \cdot E
    # where c_n and E are Q x K matrices (for Q quasimomenta and K fourier components)
    #   - c_n contains the fourier coefficients c_{qk}^{(n)}
    #   - and E contains the phases e^{i(q+2k)z}
    # and \cdot denotes the scalar product, much like (a,b) \cdot (x,y) = ax + by
    site_number = len(momenta)
    fourier_terms = len(fourier_vecs[0,0,:])
    k_max = fourier_terms // 2
    k_values = 2 * (np.arange(fourier_terms) - k_max)
    def integrand(z):
        q_phases = repmat(np.exp(1j * momenta * z), fourier_terms, 1).T
        k_phases = repmat(np.exp(1j * k_values * z), site_number, 1)
        phases = q_phases * k_phases
        phases_nn = phases
        phases_mm = phases
        if neighbors != 0:
            site_shift = repmat(np.exp(-1j * momenta * np.pi), fourier_terms, 1).T
            neighbor_phases = phases * site_shift
            phases_nn = neighbor_phases
            phases_mm = neighbor_phases if neighbors == 2 else phases
        return np.real(np.conj(np.sum(fourier_vecs[:,kk,:] * phases) *
                               np.sum(fourier_vecs[:,ll,:] * phases)) *
                       np.sum(fourier_vecs[:,mm,:] * phases_mm) *
                       np.sum(fourier_vecs[:,nn,:] * phases_nn))

    interval = np.pi * padding_sites
    shift = 0 if (neighbors == 0) else np.pi/2
    normalization = np.pi * site_number**2
    if neighbors % 2 == 0: # integral is symmetric about the middle site
        integral = 2 * quad(integrand, shift, shift + interval,
                            limit = subinterval_limit)[0]
    else: # integral is not symmetric about the middle site
        integral = quad(integrand, shift-interval, shift+interval,
                        limit = subinterval_limit)[0]
    return integral / normalization**2

# like pair_overlap_1D, but with harmonic oscillator wavefunctions
def harmonic_pair_overlap_1D(lattice_depth, kk = 0, ll = 0, mm = 0, nn = 0,
                             neighbors = 0, subinterval_limit = 1000):
    if (kk + ll + mm + nn) % 2!= 0: return 0 # odd integrals vanish
    assert(neighbors in [0,1,2]) # 3 and 4 are the same as 1 and 0

    site_shift = 0 if not neighbors else np.pi
    def integrand(z):
        shift_nn = site_shift if neighbors > 0 else 0
        shift_mm = site_shift if neighbors > 1 else 0
        return ( harmonic_wavefunction(z, kk, lattice_depth) *
                 harmonic_wavefunction(z, ll, lattice_depth) *
                 harmonic_wavefunction(z - shift_mm, mm, lattice_depth) *
                 harmonic_wavefunction(z - shift_nn, nn, lattice_depth) )

    # mass * trap frequency in lattice units (explained in "harmonic_wavefunction" method)
    mw = np.sqrt(lattice_depth)
    # all wavefunctions decay exponentially past the position
    #   z_n at which 1/2 m w^2 z_n^2 = E_n = w*(n+1/2),
    # where n = max(kk,ll,mm,nn)
    # working it out, we get z_n = sqrt( (2*n+1) / (m*w) )
    # we integrate two lattice sites (lattice constant = pi) past this position
    z_max = np.sqrt( (2*max(kk,ll,mm,nn) + 1) / mw ) + 2*np.pi

    return quad(integrand, -z_max, z_max, limit = subinterval_limit)[0]

# two-body 1-D ground-state nearest-neighbor p-wave overlap integral
#   \int \d z | \phi_0 \phi_1' - \phi_0' \phi_1 |^2
def pair_overlap_pwave_1D(momenta, fourier_vecs, kk = 0, ll = 0, mm = 0, nn = 0,
                          padding_sites = None, subinterval_limit = 1000):
    if padding_sites == None: padding_sites = 2

    site_number = len(momenta)
    fourier_terms = len(fourier_vecs[0,0,:])
    k_max = fourier_terms // 2
    k_values = 2 * (np.arange(fourier_terms) - k_max)
    def integrand(z):
        q_phases = repmat(np.exp(1j * momenta * z), fourier_terms, 1).T
        k_phases = repmat(np.exp(1j * k_values * z), site_number, 1)
        phases = q_phases * k_phases
        site_shift = repmat(np.exp(-1j * momenta * np.pi), fourier_terms, 1).T
        neighbor_phases = phases * site_shift
        qk_mat = repmat(momenta, fourier_terms, 1).T + repmat(k_values, site_number, 1)

        phi_0 = np.real(np.sum(fourier_vecs[:,0,:] * phases))
        phi_1 = np.real(np.sum(fourier_vecs[:,0,:] * neighbor_phases))
        d_phi_0 = np.imag(np.sum(fourier_vecs[:,0,:] * phases * qk_mat))
        d_phi_1 = np.imag(np.sum(fourier_vecs[:,0,:] * neighbor_phases * qk_mat))
        return abs( phi_0 * d_phi_1 - d_phi_0 * phi_1 )**2

    shift = np.pi/2
    interval = np.pi * padding_sites
    normalization = np.pi * site_number**2
    integral = 2 * quad(integrand, shift, shift+interval, limit = subinterval_limit)[0]
    return integral / normalization**2

# ground-state momentum-dependent coupling overlap integral
#   i.e. Eq. 28 in johnson2012effective, but without the factor of 4*pi
def momentum_coupling_overlap_3D(momenta_list, fourier_vecs_list,
                                 padding_sites = None, subinterval_limit = 1000):
    assert(type(momenta_list) is list or type(momenta_list) is np.ndarray)
    if padding_sites == None: padding_sites = 2

    if type(momenta_list) is list:
        assert(len(momenta_list) == 3)
        assert(len(fourier_vecs_list) == 3)
        momenta_x, fourier_vecs_x = momenta_list[0], fourier_vecs_list[0]
        momenta_y, fourier_vecs_y = momenta_list[1], fourier_vecs_list[1]
        momenta_z, fourier_vecs_z = momenta_list[2], fourier_vecs_list[2]
        assert(momenta_x.size == momenta_y.size)
        assert(momenta_y.size == momenta_z.size)
    else:
        # if we specified only one set of momenta and fourier_vecs,
        #   assume they are the same along amm axes
        momenta_x, fourier_vecs_x = momenta_list, fourier_vecs_list
        momenta_y, fourier_vecs_y = momenta_list, fourier_vecs_list
        momenta_z, fourier_vecs_z = momenta_list, fourier_vecs_list

    site_number = len(momenta_x)
    fourier_terms = len(fourier_vecs_x[0,0,:])
    k_max = fourier_terms // 2
    k_values = 2 * (np.arange(fourier_terms) - k_max)
    def integrand(momenta, fourier_vecs, z):
        q_phases = repmat(np.exp(1j * momenta * z), fourier_terms, 1).T
        k_phases = repmat(np.exp(1j * k_values * z), site_number, 1)
        phases = q_phases * k_phases
        qk_mat = repmat(momenta, fourier_terms, 1).T + repmat(k_values, site_number, 1)
        phi_z = np.real(np.sum(fourier_vecs[:,0,:] * phases))
        d_phi_z = -np.imag(np.sum(fourier_vecs[:,0,:] * phases * qk_mat))
        dd_phi_z = -np.real(np.sum(fourier_vecs[:,0,:] * phases * qk_mat**2))
        return phi_z**2 * ( d_phi_z**2 - phi_z * dd_phi_z )

    interval = np.pi * padding_sites
    normalization = np.pi * site_number**2

    def integrand_x(x): return integrand(momenta_x, fourier_vecs_x, x)
    def integrand_y(y): return integrand(momenta_y, fourier_vecs_y, y)
    def integrand_z(z): return integrand(momenta_z, fourier_vecs_z, z)

    integral_x = 2 * quad(integrand_x, 0, interval, limit = subinterval_limit)[0]
    integral_y = 2 * quad(integrand_y, 0, interval, limit = subinterval_limit)[0]
    integral_z = 2 * quad(integrand_z, 0, interval, limit = subinterval_limit)[0]
    overlap_1D_x = pair_overlap_1D(momenta_x, fourier_vecs_x)
    overlap_1D_y = pair_overlap_1D(momenta_y, fourier_vecs_y)
    overlap_1D_z = pair_overlap_1D(momenta_z, fourier_vecs_z)

    overlaps = ( overlap_1D_x * overlap_1D_y * integral_z +
                 overlap_1D_y * overlap_1D_z * integral_x +
                 overlap_1D_z * overlap_1D_x * integral_y)
    return 1/2 * overlaps / normalization**2


##########################################################################################
# momentum-space overlap integrals
# pp, qq, rr, ss index quasi-momentum
# kk, ll, mm, nn index band
##########################################################################################

# get fourier vectors corresponding to given momentum / band indices,
#   shift fourier vectors left / right appropriately to account
#   for momenta outside the first brillouin zone
def state_fourier_vec(qq, nn, fourier_vecs):
    site_number = fourier_vecs[:,0,0].size
    vecs = fourier_vecs[qq % site_number, nn, :]
    while qq >= site_number:
        vecs = np.roll(vecs, -1) # "rotate" vectors to the left
        vecs[-1] = 0 # clear rightmost entry
        qq -= site_number
    while qq < 0:
        vecs = np.roll(vecs, 1) # "rotate" vectors to the right
        vecs[0] = 0 # clear leftmost entry
        qq += site_number
    return vecs

# overlap between two states after their quasi-momentum phases have been cancelled out
#   (e.g. by the clock laser)
def state_vector_overlap_1D(qq, nn, gg, mm, fourier_vecs):
    return ( state_fourier_vec(qq, nn, fourier_vecs)
             @ state_fourier_vec(gg, mm, fourier_vecs) )

# two-body overlap for (pp,kk) + (qq,ll) <--> (rr,mm) + (ss,nn) coupling,
# where pp, qq, rr, ss are quasi-momentum indices and kk, ll, mm, nn are band indices
def momentum_pair_overlap_1D(momenta, fourier_vecs,
                             pp = None, qq = None, rr = None, ss = None,
                             kk = 0, ll = 0, mm = 0, nn = 0,
                             subinterval_limit = 1000):
    site_number = len(momenta)

    # set default index values for momenta at / near 0
    if pp == None: pp = site_number // 2
    if qq == None: qq = ( site_number - 1) // 2
    if rr == None: rr = site_number // 2
    if ss == None: ss = ( site_number - 1) // 2

    # enforce conservation of momentum and parity
    if ( pp + qq - rr - ss ) % site_number != 0: return 0
    if ( kk + ll + mm + nn ) % 2 != 0: return 0

    pk_vec = state_fourier_vec(pp, kk, fourier_vecs)
    ql_vec = state_fourier_vec(qq, ll, fourier_vecs)
    rm_vec = state_fourier_vec(rr, mm, fourier_vecs)
    sn_vec = state_fourier_vec(ss, nn, fourier_vecs)

    fourier_terms = len(fourier_vecs[0,0,:])
    k_max = fourier_terms // 2
    k_values = 2 * (np.arange(fourier_terms) - k_max)
    # determine integrand at position z
    def integrand(z):
        k_phases = np.exp(1j * k_values * z)
        # individual wavefunctions at position z
        phi_pk = pk_vec @ k_phases
        phi_ql = ql_vec @ k_phases
        phi_rm = rm_vec @ k_phases
        phi_sn = sn_vec @ k_phases
        # due to the choice of gauge in mathieu_solution and conservation of parity,
        #   the integrand should always be real
        return np.real(np.conj(phi_pk * phi_ql) * phi_rm * phi_sn)

    # the integral is even about z = 0, so only compute half of it
    lattice_length = np.pi * site_number
    overlap = 2 * quad(integrand, 0, lattice_length/2, limit = subinterval_limit)[0]
    normalization = (np.pi * site_number)**2
    return overlap / normalization
