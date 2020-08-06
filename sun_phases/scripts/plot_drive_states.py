#!/usr/bin/env python3

import os, sys, scipy
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from dicke_methods import coherent_spin_state_angles, \
    spin_op_vec_mat_dicke, plot_dicke_state

np.set_printoptions(linewidth = 200)

dim = int(sys.argv[1])

fig_dir = "../figures/drive_states/"

##################################################
# define objects

def polarized_state(theta, phi):
    return coherent_spin_state_angles(theta, phi, dim-1)

init_state = polarized_state(np.pi,0)

S_vec, SS_mat = spin_op_vec_mat_dicke(dim-1)
Sz = S_vec[0].todense()
Sx = S_vec[1].todense()
Sy = S_vec[2].todense()

H_raman_p = np.sqrt(2) * Sx + Sx @ Sx - Sy @ Sy
H_raman_m = np.sqrt(2) * ( Sz @ Sx + Sx @ Sz ) - ( Sx @ Sx - Sy @ Sy )

##################################################
# plotting methods

def time_deriv(state, H):
    return -1j * H.dot(state.T)
def ivp_deriv(H):
    return lambda time, state : time_deriv(state, H)

def save_states(name, hamiltonian, tau_final, angle = 0, time_steps = 1000,
                ivp_tolerance = 1e-10):
    times = tau_final * 2*np.pi * np.array(range(time_steps)) / time_steps
    states = solve_ivp(ivp_deriv(hamiltonian), (0,times[-1]), init_state,
                       t_eval = times, rtol = ivp_tolerance, atol = ivp_tolerance).y

    R = scipy.linalg.expm(1j*Sy*angle)
    digits = int(np.floor(np.log10(time_steps-1)))+1

    sim_tag = f"{name}_{dim}"
    sim_dir = fig_dir + sim_tag + "/"
    if not os.path.isdir(sim_dir): os.makedirs(sim_dir)

    for step, ( time, state ) in enumerate(zip(times, states.T)):
        print(f"{step} / {time_steps}")
        step_str = str(step).zfill(digits)
        plot_dicke_state(R @ state, single_sphere = False)
        plt.savefig(sim_dir + f"{sim_tag}_{step_str}.png")
        plt.close("all")

# save_states("state_m", H_raman_m, tau_final = 1/6)
# save_states("state_p", H_raman_p, tau_final = 1/6)

##################################################

tau = 1/4
period = 2*np.pi/3

state = scipy.linalg.expm(1j * H_raman_m * period * tau) @ init_state

def infidelity(guess):
    return 1 - abs(np.dot(state.conj(), guess)**2) / sum(abs(guess)**2)

if any( np.isclose(tau % 1, [0,1/2,1]) ):
    if np.isclose(tau % 1, 1/2) and dim % 2 == 1:
        theta = np.pi/2 - np.arcsin(1/3)
    else:
        theta = np.pi

    guess = polarized_state(theta, 0)

if any( np.isclose(tau % 1, [1/4,3/4]) ):
    if dim % 2 == 0:
        alpha_1 = np.pi/2 + np.arcsin(1/3)
        alpha_2 = np.pi/2 + np.arcsin(1/3)
        beta = np.pi/3
        gamma = -(dim-1)/3 * np.pi

    else:
        alpha_1 = np.pi
        alpha_2 = np.pi/2 - np.arcsin(1/3)
        beta = 0
        gamma = (dim-2)/4 * np.pi
        if np.isclose(tau,3/4): gamma *= -1

    guess = ( polarized_state(alpha_1, +beta) * np.exp(+1j*gamma) +
              polarized_state(alpha_2, -beta) * np.exp(-1j*gamma) )

badness = infidelity(guess)

print(badness)
if not np.isclose(badness,0):
    plot_dicke_state(state)
    plot_dicke_state(guess)
    plt.show()
