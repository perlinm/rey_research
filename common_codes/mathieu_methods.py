#!/usr/bin/env python3

# FILE CONTENTS: (numerical) methods to solve the mathieu equation

import numpy, scipy.linalg

# this method computes eigenvalues and corresponding eigenstates of the Mathieu equation.
# solutions to (i.e. eigenfunctions of) the Mathieu equation take the form
#   \phi_{qn}(z) = e^{iqz} \sum_k c_{qn}^{(k)} e^{i2kz},
#   where q is a quasi-momentum and n is a band index.
# we solve the Mathieu equation numerically using a method outlined in:
#   Coisson, Roberto, Graziano Vernizzi, and Xiaoke Yang (OSSC, 2009).
# the constants c_{qn}^{(k)} are collected into an object called "fourier_vecs",
#   which is indexed respectively by q, n, and k, such that fourier_vecs[q,n,:]
#   is a fourier-space vector representation of \phi_{qn}
# "energies" is indexed by q and n, and contains the energies corresponding to \phi_{qn}
def single_mathieu_solution(q, lattice_depth, bands, fourier_order):
    mathieu_q = -lattice_depth / 4

    # construct Mathieu operator
    diagonal = (2 * numpy.arange(-fourier_order, fourier_order+1) + q)**2
    off_diagonal = mathieu_q * numpy.ones(len(diagonal)-1)
    mathieu_operator = ( numpy.diag(diagonal) +
                         numpy.diag(off_diagonal, -1) +
                         numpy.diag(off_diagonal, 1) )

    # sort solutions by increasing eigenvalue
    eig_vals, fourier_vecs = scipy.linalg.eig(mathieu_operator)
    sort_order = eig_vals.argsort()
    energies = numpy.real(eig_vals[sort_order][:bands]) + lattice_depth / 2
    fourier_vecs = numpy.real(fourier_vecs[:,sort_order][:,:bands])

    return fourier_vecs, energies

# this method computes momenta, fourier_vecs, and energies of all single-particle
#   eigenstates on a lattice.
# "momenta[q]" is a quasi-momentum in units with the lattice wavenumber equal to 1
# "fourier_vecs[q,n,:]" is a fourier-space vector representing the eigenfunction \phi_{qn}
# "energies[q,n]" is the single-particle energy corresponding to \phi_{qn}
# "symmetric" controls whether quasi-momenta are distributed symetrically about 0.
# "fourier_order" is a number passed so single_mathieu_solution (see method above);
#   it is generally advised to leave this variable alone.
def mathieu_solution(lattice_depth, bands, site_number,
                     symmetric = True, fourier_order = None):
    energies = []
    fourier_vecs = []
    if fourier_order == None: fourier_order = 2 * bands

    # allowed quasi-momenta with a given number of lattice sites
    #   (in units of the lattice wavenumber)
    dq = 2 / site_number
    momenta = numpy.array([ qi * dq - 1 for qi in range(site_number) ])
    if symmetric: momenta += dq / 2

    # compute eigenenergies and fourier components (i.e eigenstates)
    for q in momenta:
        q_fourier_vecs, q_energies \
            = single_mathieu_solution(q, lattice_depth, bands, fourier_order)
        fourier_vecs.append(q_fourier_vecs.T)
        energies.append(q_energies)
    fourier_vecs = numpy.array(fourier_vecs)
    energies = numpy.array(energies)

    # enforce a fixed gauge for the momentum-space eigenstates
    for ni in range(bands):
        # make sure that the phase within each band is "smoothly varying"
        #   (in a discrete sense) with quasi-momentum
        for qi in range(1, site_number):
            inner_product = numpy.dot(fourier_vecs[qi-1,ni,:], fourier_vecs[qi,ni,:])
            if inner_product < 0:
                fourier_vecs[qi,ni,:] *= -1

        # enforce that within each band n, \phi_{jn}\p{z=z_j} "looks like"
        #    i^n*cos(z) (even n) or i^n*sin(z) (odd n)
        # positive-exponent fourier vectors for q = 0
        positive_zero_vecs = fourier_vecs[site_number//2,ni,fourier_order:]
        # sign of dominant positive-exponent fourier component
        dominant_sign = numpy.sign(max(positive_zero_vecs.min(),
                                       positive_zero_vecs.max(),
                                       key = abs))
        # the above sign should flip every two bands
        if dominant_sign != (-1)**(ni//2):
            fourier_vecs[:,ni,:] *= -1

    return momenta, fourier_vecs, energies
