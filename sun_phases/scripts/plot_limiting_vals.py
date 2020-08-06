#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.special

from dicke_methods import coherent_spin_state_angles

figsize = (3,2)

params = { "font.size" : 12,
           "text.usetex" : True,
           "text.latex.preamble" : [ r"\usepackage{physics}" ]}
plt.rcParams.update(params)

##############################

n_vals = np.arange(2,21,2)
def spin_vals(n):
    S = (n-1)/2
    return np.linspace(-S,S,n)

theta = np.pi/2 + np.arcsin(1/3)
alpha = np.pi/3
def beta(n):
    return -(n-1)/3 * np.pi
def state(n):
    return ( coherent_spin_state_angles(theta, +alpha, n-1) * np.exp(+1j*beta(n)) +
             coherent_spin_state_angles(theta, -alpha, n-1) * np.exp(-1j*beta(n)) ) / np.sqrt(2)

##############################

def chi(m,n):
    S = (n-1)/2
    return ( np.sqrt(scipy.special.binom(2*S,S+m)) *
             np.cos(theta/2)**(S+m) * np.sin(theta/2)**(S-m) )
def proj(m,n):
    return chi(m,n) * np.sqrt(2) * np.cos(m*alpha-beta(n))

for n in n_vals:
    assert(np.allclose( state(n), [ proj(m,n) for m in spin_vals(n) ] ))

##############################

def r_val(n):
    return sum( chi(m,n)**4 * 4 * np.cos(m*alpha-beta(n))**4
                for m in spin_vals(n) )

r_vals = [ sum(abs(state(n))**4) for n in n_vals ]
r_vals_alt = [ r_val(n) for n in n_vals ]
assert(np.allclose(r_vals, r_vals_alt))

plt.figure(figsize = figsize)
plt.plot(n_vals, r_vals, "k.")
plt.gca().set_ylim(bottom = 0)
plt.xlabel("$n$")
plt.ylabel("$r$")
plt.tight_layout()
plt.savefig("../figures/oscillations/const_vals.pdf")

##############################

def var_Z(n):
    _state = state(n)
    S = (n-1)/2
    Z = spin_vals(n) / S
    val_Z_Z = _state.conj() @ ( Z**2 * _state )
    val_Z = _state.conj() @ ( Z * _state )
    return ( val_Z_Z - val_Z**2 ).real
var_Z_vals = np.array([ var_Z(n) for n in n_vals ])

plt.figure(figsize = figsize)
plt.plot(n_vals, var_Z_vals, "k.")
plt.xlabel("$n$")
plt.ylabel(r"$\mathrm{var}(Z)$")
plt.tight_layout()
plt.savefig("../figures/oscillations/var_z_vals.pdf")


plt.figure(figsize = figsize)
plt.plot(n_vals, np.sqrt(var_Z_vals), "k.")
plt.xlabel("$n$")
plt.ylabel(r"$\mathrm{std}(Z)$")
plt.tight_layout()
plt.savefig("../figures/oscillations/std_z_vals.pdf")
