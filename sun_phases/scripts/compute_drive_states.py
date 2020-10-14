#!/usr/bin/env python3

import os, sys, functools, scipy, scipy.integrate
import numpy as np
import matplotlib.pyplot as plt

from dicke_methods import coherent_spin_state_angles, \
    spin_op_vec_mat_dicke, plot_dicke_state

np.set_printoptions(linewidth = 200)

dim = int(sys.argv[1])

data_dir = "../data/drive_states/"

##################################################
# define objects

total_spin = (dim-1)/2
spin_vals = np.arange(total_spin, -total_spin-1, -1)

def polarized_state(theta, phi):
    return coherent_spin_state_angles(theta, phi, dim-1)

init_state = polarized_state(np.pi,0)

S_vec, SS_mat = spin_op_vec_mat_dicke(dim-1)
Sz = S_vec[0].todense()
Sx = S_vec[1].todense()
Sy = S_vec[2].todense()

H_drive = Sx - Sx @ Sx

##################################################
# simulation methods

def time_deriv(state, hamiltonian):
    return -1j * hamiltonian.dot(state.T)
def ivp_deriv(hamiltonian):
    return lambda time, state : time_deriv(state, hamiltonian)
def get_states(times, hamiltonian, ivp_tolerance = 1e-10):
    solve_ivp = scipy.integrate.solve_ivp
    return solve_ivp(ivp_deriv(hamiltonian), (0,times[-1]), init_state,
                     t_eval = times, rtol = ivp_tolerance, atol = ivp_tolerance).y.T

##################################################
# simulate and save states

def get_energies(phi, qq):
    return -2 * np.array([ np.cos(qq + mu*phi) for mu in spin_vals ])
def H_SOC(phi, qq):
    return np.diag(get_energies(qq,phi))
def H_total(theta, phi, qq):
    return weight * H_drive + (1-weight) * H_SOC(phi,qq)

def correlation(ff, gg, domain):
    def _mean(hh):
        return scipy.integrate.simps(hh.T, domain) / (domain[-1] - domain[0])
    ff_dev = ff - _mean(ff) # deviation from the mean
    gg_dev = gg - _mean(gg)
    ff_var = _mean(ff_dev**2) # variance
    gg_var = _mean(gg_dev**2)

    cov = _mean(ff_dev * gg_dev) # covariance
    norm = np.sqrt(ff_var * gg_var) # normalization factor

    # correct for numerical zeroes and return
    cov[np.isclose(cov,0)] = 0
    norm[cov == 0] = 1
    return cov / norm

times = np.linspace(0, 10*np.pi, 1001)
phi_vals = np.linspace(0, 2*np.pi/total_spin, 101)
weight_vals = np.linspace(0, 1, 101)
momenta = np.linspace(-np.pi, np.pi, 121)

for phi_idx, phi in enumerate(phi_vals):
    print(f"{phi_idx}/{phi_vals.size}")

    energies = np.array([ get_energies(phi,qq) for qq in momenta ])

    phi_tag = f"d{dim:02d}_p{phi_idx:03d}"
    np.savetxt(data_dir + f"energies_{phi_tag}.txt", energies)

    for weight_idx, weight in enumerate(weight_vals):
        print(f" {weight_idx}/{weight_vals.size}")

        states = np.array([ get_states(times, H_total(weight, phi, qq))
                            for qq in momenta ])
        correlations = np.array([ correlation(abs(states[:,tt,:])**2, energies, momenta)
                                  for tt in range(times.size) ])

        suffix = f"{phi_tag}_w{weight_idx:03d}"
        states.shape = (states.shape[0], -1)
        np.savetxt(data_dir + f"states_{suffix}.txt", states)
        np.savetxt(data_dir + f"correlations_{suffix}.txt", correlations)

