#!/usr/bin/env python3

import os, sys, functools, scipy, scipy.integrate
import numpy as np
import matplotlib.pyplot as plt

from dicke_methods import spin_op_vec_dicke

np.set_printoptions(linewidth = 200)

dim = int(sys.argv[1])
if len(sys.argv) > 2:
    drive = [ int(val) for val in sys.argv[2:] ]
else:
    drive = [ 1, 0, 1 ] # default drive

data_dir = "../data/drive_states/drive_"
for val in drive:
    if val == 0: data_dir += "z"
    elif val == +1: data_dir += "p"
    elif val == -1: data_dir += "m"
    else: assert(False)
data_dir += "/"
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

##################################################
# define objects

total_spin = (dim-1)/2
Sz, Sx, Sy = [ op.todense() for op in spin_op_vec_dicke(dim-1) ]
spin_vals = np.diag(Sz)

# initial state pointing in -Z
init_state = np.zeros(dim, dtype = complex)
init_state[spin_vals.argmin()] = 1

# build drive Hamiltonian
def build_drive(drive_strengths):
    zz, pp, mm = drive_strengths
    fst = pp*mm * Sz + zz*mm * Sx + zz*pp * (Sz @ Sx + Sx @ Sz)
    snd = zz**2 * Sz @ Sz + pp**2 * Sx @ Sx + mm**2 * Sy @ Sy
    drive = fst - snd
    return drive - np.trace(drive)/dim * np.eye(dim)

##################################################
# time evolution methods

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

H_drive = build_drive(drive)

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
phi_frac_vals = np.arange(0.1, 1, 0.1)
weight_vals = np.arange(0.1, 1, 0.1)
momenta = np.linspace(-np.pi, np.pi, 201)

for phi_idx, phi_frac in enumerate(phi_frac_vals):
    print(f"{phi_idx}/{phi_frac_vals.size}")

    phi = phi_frac * 2*np.pi/total_spin
    energies = np.array([ get_energies(phi,qq) for qq in momenta ])

    phi_tag = f"d{dim:02d}_p{phi_frac:.2f}"
    np.savetxt(data_dir + f"energies_{phi_tag}.txt", energies)

    for weight_idx, weight in enumerate(weight_vals):
        print(f" {weight_idx}/{weight_vals.size}")
        sys.stdout.flush()

        states = np.array([ get_states(times, H_total(weight, phi, qq))
                            for qq in momenta ])
        correlations = np.array([ correlation(abs(states[:,tt,:])**2, energies, momenta)
                                  for tt in range(times.size) ])

        suffix = f"{phi_tag}_w{weight:.2f}"
        states.shape = (states.shape[0], -1)
        np.savetxt(data_dir + f"states_{suffix}.txt", states)
        np.savetxt(data_dir + f"correlations_{suffix}.txt", correlations)
