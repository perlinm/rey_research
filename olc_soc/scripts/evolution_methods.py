#!/usr/bin/env python3

import os, sys
import numpy as np
import scipy.linalg
import scipy.integrate

from mathieu_methods import mathieu_solution
from lattice_methods import qns_energy, laser_overlap
from sr87_olc_constants import spins

##########################################################################################
# eigenvectors and static Hamiltonians
##########################################################################################

# index corresponding to band n and spin s
def ns_index(n, s):
    return int(round(2*n + (s+1)/2))

# state vector
def ns_vec(n, s, bands):
    vec = np.zeros(2*bands)
    vec[ns_index(n,s)] = 1
    return vec

# Hamiltonian containing only state energies at no detuning
def energy_hamiltonian(q, energies, bands):
    H = np.zeros((2*bands,2*bands))
    for n in range(bands):
        for s in spins:
            H[ns_index(n,s),ns_index(n,s)] = qns_energy(q, n, s, energies)
    return H

# contribution to Hamiltonian from detuning
def detuning_hamiltonian(detuning, bands):
    H_diag = np.zeros(2*bands)
    for s in spins:
        H_diag[ns_index(0,s)::2] = -s*detuning/2
    return np.diag(H_diag)

# off-diagonal terms of the Hamiltonian
def coupling_hamiltonian(q, momenta, fourier_vecs, rabi_coupling, bands):
    H = np.zeros((2*bands,2*bands))
    coupling_coefficient = -rabi_coupling / 2
    for n in range(bands):
        for s in spins:
            for m in range(bands):
                H[ns_index(m,-s),ns_index(n,s)] \
                    = coupling_coefficient * laser_overlap(q,s,n,m,momenta,fourier_vecs)
    return H

# complete Hamiltonian with energies, detuning, and coupling terms
def full_hamiltonian(q, momenta, energies, fourier_vecs,
                     rabi_coupling, detuning, bands):
    return (energy_hamiltonian(q, energies, bands)
            + detuning_hamiltonian(detuning, bands)
            + coupling_hamiltonian(q, momenta, fourier_vecs, rabi_coupling, bands))

# compute effective driving Hamiltonian
def effective_driving_hamiltonian(q, momenta, energies, fourier_vecs,
                                  rabi_coupling, mod_type, mod_freq, mod_index,
                                  detuning_mean, bands):
    assert(mod_type in ["freq","amp"])

    H_coupling = coupling_hamiltonian(q, momenta, fourier_vecs, rabi_coupling, bands)
    H_E = ( energy_hamiltonian(q, energies, bands)
            + detuning_hamiltonian(detuning_mean, bands) )

    D_vals = np.zeros((2*bands,2*bands))
    k_vals = np.zeros((2*bands,2*bands))
    H_E_shift = np.zeros((2*bands,2*bands))
    for n in range(bands):
        for s in spins:
            ns = ns_index(n,s)
            for m in range(bands):
                for r in spins:
                    mr = ns_index(m,r)
                    # energy gap between |qmr> and |qns>
                    D_vals[mr,ns] = H_E[mr,mr] - H_E[ns,ns]
                    # integer "k" which minimizes | D_vals[ns,mr] - k*mod_freq |
                    k_vals[mr,ns] = round(D_vals[mr,ns] / mod_freq)

                    if H_coupling[mr,ns] == 0: continue

                    # additional coupling factor between |qns> and |qmr>
                    if mod_type == "freq":
                        H_coupling[mr,ns] *= scipy.special.jv(k_vals[mr,ns], s*mod_index)

                    else: # if mod_type == "amp"
                        if abs(k_vals[mr,ns]) == 1:
                            H_coupling[mr,ns] *= 0.5
                        else:
                            H_coupling[mr,ns] = 0

            H_E_shift[ns,ns] = -k_vals[ns,0] * mod_freq

    H_I = H_E + H_E_shift + H_coupling

    return H_I


##########################################################################################
# extract populations
##########################################################################################

# convert time series of real states to a time series of populations
def states_to_populations(states):
    dim = np.shape(np.array(states))[0] // 2
    time_num = np.shape(states)[1]
    populations = np.zeros((dim, time_num))
    for ii in range(dim):
        populations[ii,:] = states[2*ii,:]**2 + states[2*ii+1,:]**2
    return populations

# convert time series of state populations to a time series of subspace populations
def select_populations(populations, population_type = "all"):
    assert(population_type in ["all","band","spin"])
    bands = np.shape(np.array(populations))[0] // 2
    new_populations = np.copy(populations)
    if population_type == "all":
        return new_populations
    elif population_type == "band":
        for n in range(bands):
            new_populations[2*n,:] += new_populations[2*n+1,:]
        return new_populations[::2,:]
    elif population_type == "spin":
        for n in range(1,bands):
            new_populations[0,:] += new_populations[2*n,:]
            new_populations[1,:] += new_populations[2*n+1,:]
        return new_populations[:2,:]


##########################################################################################
# convert between real- and complex-valued objects
##########################################################################################

# convert a complex-valued state to a real-valued state
def to_real_state(state):
    dim = len(state)
    output_state = np.zeros(2*dim)
    for ii in range(dim):
        output_state[2*ii] = np.real(state[ii])
        output_state[2*ii+1] = np.imag(state[ii])
    return output_state

# contert a real-valued state to a complex-valued one
def to_complex_state(state):
    dim = len(state)//2
    output_state = np.zeros(dim, dtype = complex)
    for jj in range(dim):
        output_state[jj] = state[2*jj] + 1j*state[2*jj+1]
    return output_state

# convert a complex-valued Hamiltonian to a real-valued differential operator
def to_real_hamiltonian(H):
    dim = np.shape(H)[0]
    output_H = np.zeros((2*dim,2*dim))
    for ii in range(dim):
        for jj in range(dim):
            re_H = np.real(H[ii,jj])
            im_H = np.imag(H[ii,jj])
            output_H[2*ii,2*jj] = im_H
            output_H[2*ii,2*jj+1] = re_H
            output_H[2*ii+1,2*jj] = -re_H
            output_H[2*ii+1,2*jj+1] = im_H
    return output_H

# convert a real-valued differential operator to a complex-valued Hamiltonian
def to_complex_hamiltonian(H):
    dim = np.shape(H)[0]//2
    output_H = np.zeros((dim,dim), dtype = complex)
    for ii in range(dim):
        for jj in range(dim):
            output_H[ii,jj] = H[2*ii,2*jj+1] + 1j*H[2*ii,2*jj]
    return output_H


##########################################################################################
# time evolution methods
##########################################################################################

# propagate a state with a time dependent hamiltonian
def propagate(initial_state, hamiltonian, times):
    def psi_dot(t, psi):
        return scipy.linalg.blas.dgemv(1, hamiltonian(t), psi)
    solver = scipy.integrate.ode(psi_dot)

    time_steps = len(times)
    end_time = times[-1]

    if len(initial_state) == np.shape(hamiltonian(0))[0]:
        actual_initial_state = initial_state
    else:
        actual_initial_state = to_real_state(initial_state)

    states = np.zeros((len(actual_initial_state),time_steps))
    states[:,0] = actual_initial_state

    solver.set_initial_value(states[:,0], times[0])
    tt = 1
    while solver.successful():
        solver.integrate(end_time, step = True)
        if solver.t >= times[tt]:
            states[:,tt] = solver.y
            tt += 1
            if tt >= time_steps: break
    return states

# use direct integration with full time-dependent Hamiltonian to get populations over time
def time_integrated_populations(q, n, s, momenta, energies, fourier_vecs,
                                rabi_coupling, mod_type, mod_freq, mod_index,
                                detuning_mean, bands, times):
    assert(mod_type in ["freq","amp"])

    if mod_type == "freq":
        mean_hamiltonian = full_hamiltonian(q, momenta, energies, fourier_vecs,
                                            rabi_coupling, detuning_mean, bands)
        mod_hamiltonian = detuning_hamiltonian(mod_index * mod_freq, bands)

    else: # if mod_type == "amp"
        mean_hamiltonian = (energy_hamiltonian(q, energies, bands)
                            + detuning_hamiltonian(detuning_mean, bands))
        mod_hamiltonian = coupling_hamiltonian(q, momenta, fourier_vecs,
                                               rabi_coupling, bands)

    mean_hamiltonian = to_real_hamiltonian(mean_hamiltonian)
    mod_hamiltonian = to_real_hamiltonian(mod_hamiltonian)

    def hamiltonian(time):
        return mean_hamiltonian - mod_hamiltonian * np.cos(mod_freq * time)

    initial_state = ns_vec(n, s, bands)
    states = propagate(initial_state, hamiltonian, times)
    populations = states_to_populations(states)
    for tt in range(len(times)):
        populations[:,tt] /= sum(populations[:,tt])

    return populations

# use effective drive hamiltonian to get populations over time
def effective_driving_populations(q, n, s, momenta, energies, fourier_vecs,
                                  rabi_coupling, mod_type, mod_freq, mod_index,
                                  detuning_mean, bands, times):
    H_eff = effective_driving_hamiltonian(q, momenta, energies, fourier_vecs,
                                          rabi_coupling, mod_type, mod_freq, mod_index,
                                          detuning_mean, bands)
    def U_eff(t): return scipy.linalg.expm(-1j * H_eff * t)

    initial_state = ns_vec(n, s, bands)
    populations = np.zeros((len(initial_state),len(times)))
    for tt in range(len(times)):
        populations[:,tt] = abs(U_eff(times[tt]) @ initial_state)**2
        populations[:,tt] /= sum(populations[:,tt])

    return populations
