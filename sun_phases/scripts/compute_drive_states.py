#!/usr/bin/env python3

import os, sys, functools, scipy, scipy.integrate
import numpy as np
import matplotlib.pyplot as plt

from dicke_methods import coherent_spin_state_angles, \
    spin_op_vec_mat_dicke, plot_dicke_state

np.set_printoptions(linewidth = 200)

dim = int(sys.argv[1])

fig_dir = "../figures/drive_states/"

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

##################################################
# simulation and plotting methods

def time_deriv(state, hamiltonian):
    return -1j * hamiltonian.dot(state.T)
def ivp_deriv(hamiltonian):
    return lambda time, state : time_deriv(state, hamiltonian)

def get_states(times, hamiltonian, ivp_tolerance = 1e-10):
    return scipy.integrate.solve_ivp(ivp_deriv(hamiltonian), (0,times[-1]),
                                     init_state, t_eval = times,
                                     rtol = ivp_tolerance, atol = ivp_tolerance).y.T

def plot_states(name, times, states, angle = 0):
    R = scipy.linalg.expm(1j*Sy*angle)
    digits = int(np.floor(np.log10(time_steps-1)))+1

    sim_tag = f"{name}_{dim}"
    sim_dir = fig_dir + sim_tag + "/"
    if not os.path.isdir(sim_dir): os.makedirs(sim_dir)

    for step, ( time, state ) in enumerate(zip(times, states)):
        print(f"{step} / {time_steps}")
        step_str = str(step).zfill(digits)
        plot_dicke_state(R @ state, single_sphere = False)
        plt.savefig(sim_dir + f"{sim_tag}_{step_str}.png")
        plt.close("all")

##################################################
# simulate and save states

def energies(phi, qq):
    return -2 * np.array([ np.cos(qq + mu*phi) for mu in spin_vals ])
def H_SOC(phi, qq):
    return np.diag(energies(qq,phi))

H_drive = Sx - Sx @ Sx

def H_total(theta, phi, qq):
    return np.cos(theta) * H_drive + np.sin(theta) * H_SOC(phi,qq)

##################################################
def correlation(ff, gg, min, max):
    _mean = lambda hh : scipy.integrate.quad_vec(hh,min,max)[0] / (max-min)
    ff_mean = _mean(ff)
    gg_mean = _mean(gg)
    ff_dev = lambda qq : ff(qq) - ff_mean
    gg_dev = lambda qq : gg(qq) - gg_mean
    ff_var = _mean(lambda qq : ff_dev(qq)**2)
    gg_var = _mean(lambda qq : gg_dev(qq)**2)
    norm = np.sqrt(ff_var * gg_var)
    cov = _mean(lambda qq : ff_dev(qq) * gg_dev(qq))

    if type(cov) is np.ndarray:
        cov[np.isclose(cov,0)] = 0
        norm[cov == 0] = 1
    else:
        if np.isclose(cov,0): return 0
    return cov / norm
##################################################

times = np.linspace(0, 10*np.pi, 1001)
phi_vals = np.linspace(0, np.pi, 101)
theta_vals = np.linspace(0, np.pi, 101)
momenta = np.linspace(-np.pi, np.pi, 11)

for phi_idx, phi in enumerate(phi_vals):
    phi_energies = np.array([ energies(phi,qq) for qq in momenta ])
    for theta_idx, theta in enumerate(theta_vals):
        states = np.array([ get_states(times, H_total(theta, phi, qq))
                            for qq in momenta ])

        populations = abs(states)**2
        print(populations.shape)
        print(phi_energies.shape)
        exit()

