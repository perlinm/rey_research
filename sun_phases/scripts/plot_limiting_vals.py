#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.special

from dicke_methods import coherent_spin_state_angles

figsize = (4,1.5)

params = { "font.size" : 10,
           "text.usetex" : True,
           "text.latex.preamble" : r"\usepackage{physics}" }
plt.rcParams.update(params)

##############################

n_vals = np.arange(2,21,2)
n_ticks = np.arange(4,21,4)

I_vals = (n_vals-1)/2
def spin_vals(n):
    I = (n-1)/2
    return np.linspace(-I,I,n)

alpha = np.pi/2 + np.arcsin(1/3)
beta = np.pi/3
def gamma(n): return -(n-1)/3 * np.pi

states = [ "X", "DS" ]
def state_vec(n, state):
    if state == "X":
        return coherent_spin_state_angles(np.pi/2, 0, n-1)
    elif state == "DS":
        state_p = coherent_spin_state_angles(alpha, +beta, n-1) * np.exp(+1j*gamma(n))
        state_m = coherent_spin_state_angles(alpha, -beta, n-1) * np.exp(-1j*gamma(n))
        return ( state_p + state_m ) / np.sqrt(2)

##############################

def r_val(n, state):
    return sum( abs(state_vec(n, state))**4 )
def var_Z(n, state):
    _state_vec = state_vec(n, state)
    I = (n-1)/2
    Z = spin_vals(n) / I
    val_Z_Z = _state_vec.conj() @ ( Z**2 * _state_vec )
    val_Z = _state_vec.conj() @ ( Z * _state_vec )
    return ( val_Z_Z - val_Z**2 ).real

def r_vals(state):
    return np.array([ r_val(n,state) for n in n_vals ])
def var_Z_vals(state):
    return np.array([ var_Z(n,state) for n in n_vals ])

figure, axes = plt.subplots(1, 2, figsize = figsize, sharex = True)

for state in states:
    label = r"$\mathrm{" + state + "}$"
    axes[0].plot(n_vals, np.sqrt(2*I_vals) * r_vals(state), ".", label = label)
    axes[1].plot(n_vals, 2*I_vals * var_Z_vals(state), ".", label = label)

axes[0].set_xticks(n_ticks)
axes[0].set_xlabel(r"$n$")
axes[1].set_xlabel(r"$n$")

axes[0].set_ylim(bottom = 0)
axes[0].set_ylabel(r"$r_\psi\times\sqrt{2I}$")
axes[1].set_ylabel(r"$\mathrm{var}_\psi(Z)\times 2I$")
axes[1].legend(loc = "best")

plt.tight_layout(pad = 0.1, w_pad = 1)
plt.savefig("../figures/oscillations/limiting_vals.pdf")
