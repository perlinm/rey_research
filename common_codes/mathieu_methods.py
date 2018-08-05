#!/usr/bin/env python3

# FILE CONTENTS: (numerical) methods to solve the mathieu equation

import numpy, scipy.linalg

# this method computes eigenvalues and corresponding eigenstates of the Mathieu equation.
# solutions to (i.e. eigenfunctions of) the Mathieu equation take the form
#   \phi_{qn}(z) = e^{iqz} \sum_k c_{qn}^{(k)} e^{i2kz},
#   where q is a quasi-momentum and n is a band index.
# the Mathieu equation is solved numerically using a method outlined in
#   Coisson, Roberto, Graziano Vernizzi, and Xiaoke Yang (OSSC, 2009).
# the constants c_{qn}^{(k)} are collected into an object called "q_fourier_vecs",
#   which is indexed respectively by n such that q_fourier_vecs[n,:]
#   is a fourier-space vector representation of \phi_{qn}
# "q_energies" is indexed by n, and contains the energies corresponding to \phi_{qn}
def single_mathieu_solution(q, lattice_depth, bands, fourier_order):
    mathieu_q = -lattice_depth / 4

    # determine diagonal and off-diagonal bands of Mathieu operator
    diagonal = (2 * numpy.arange(-fourier_order, fourier_order+1) + q)**2
    off_diagonal = mathieu_q * numpy.ones(len(diagonal)-1)

    # sort solutions by increasing eigenvalue
    eig_vals, eig_vecs = scipy.linalg.eigh_tridiagonal(diagonal, off_diagonal)
    sort_order = eig_vals.argsort()
    q_energies = numpy.real(eig_vals[sort_order][:bands]) + lattice_depth / 2
    q_fourier_vecs = numpy.real(eig_vecs[:,sort_order][:,:bands]).T

    return q_fourier_vecs, q_energies

# this method computes momenta, fourier_vecs, and energies of all single-particle
#   eigenstates on a lattice.
# "momenta[q]" is a quasi-momentum in units with the lattice wavenumber equal to 1
# "fourier_vecs[q,n,:]" is a fourier-space vector representing the function \phi_{qn}.
#   \phi_{qn}(z) = e^{i q z} \sum_k fourier_vecs[q,n,k] * e^{i 2 (k - k_mean)}
# "energies[q,n]" is the single-particle energy corresponding to \phi_{qn}
# "symmetric" controls whether quasi-momenta are distributed symetrically about 0.
# "fourier_order" is a number passed so single_mathieu_solution (see method above);
#   it is generally advised to leave this variable alone.
def mathieu_solution(lattice_depth, bands, site_number,
                     symmetric = True, fourier_order = None):

    if fourier_order == None: fourier_order = max(2*bands, 10)

    # allowed quasi-momenta with a given number of lattice sites
    #   (in units of the lattice wavenumber)
    dq = 2 / site_number
    momenta = numpy.array([ qi * dq - 1 for qi in range(site_number) ])
    if symmetric: momenta += dq / 2

    # compute eigenenergies and fourier components of eigenstates
    energies = numpy.zeros((len(momenta),bands))
    fourier_vecs = numpy.zeros((len(momenta),bands,2*fourier_order+1))
    for qq in range(site_number):
        fourier_vecs[qq,:,:], energies[qq,:] \
            = single_mathieu_solution(momenta[qq], lattice_depth, bands, fourier_order)

    # enforce a fixed gauge for the momentum-space eigenstates
    for nn in range(bands):
        # make sure that the phase within each band is "smoothly varying"
        #   (in a discrete sense) with quasi-momentum
        for qi in range(1, site_number):
            inner_product = numpy.dot(fourier_vecs[qi-1,nn,:], fourier_vecs[qi,nn,:])
            if inner_product < 0:
                fourier_vecs[qi,nn,:] *= -1

        # enforce that within each band n, \phi_{jn}\p{z=z_j} "looks like"
        #    i^n*cos(z) (even n) or i^n*sin(z) (odd n)
        # positive-exponent fourier vectors for q = 0
        positive_zero_vecs = fourier_vecs[site_number//2,nn,fourier_order:]
        # sign of dominant positive-exponent fourier component
        dominant_sign = numpy.sign(max(positive_zero_vecs.min(),
                                       positive_zero_vecs.max(),
                                       key = abs))
        # the above sign should flip every two bands
        if dominant_sign != (-1)**(nn//2):
            fourier_vecs[:,nn,:] *= -1

    return momenta, fourier_vecs, energies
