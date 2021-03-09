#!/usr/bin/env python3

import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt

from dicke_methods import spin_op_x_dicke

init_state_str = sys.argv[1]
spin_dim = int(sys.argv[2])
spin_num = int(sys.argv[3])

spin = (spin_dim-1)/2

assert( init_state_str in [ "X" ] )

data_dir = "../data/spin_bosons/"
fig_dir = "../figures/spin_bosons/"
sys_tag = f"{init_state_str}_d{spin_dim}_N{spin_num}"

if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

fontsize = 10
preamble = r"""
\usepackage{braket}
"""
params = { "font.family" : "serif",
           "font.serif" : "Computer Modern",
           "font.size" : fontsize,
           "axes.titlesize" : fontsize,
           "axes.labelsize" : fontsize,
           "text.usetex" : True,
           "text.latex.preamble" : preamble }
plt.rcParams.update(params)

# convert upper triangle of density matrix to full density matrix
def mean_vals_to_states(mean_vals):
    mean_states = np.empty(( mean_vals.shape[0], spin_dim, spin_dim ), dtype = complex)
    for idx, ( mu, nu ) in enumerate(zip(*np.triu_indices(spin_dim))):
        mean_states[:,mu,nu] = mean_vals[:,idx]
        if mu != nu:
            mean_states[:,nu,mu] = mean_vals[:,idx].conj()
    return mean_states

X_op = spin_op_x_dicke(spin_dim-1).todense() / spin
mean_X_vals = {}
for file in glob.glob(data_dir + f"states_{sys_tag}*"):
    data = np.loadtxt(file, dtype = complex)
    times, vals = data[:,0], data[:,1:]
    states = mean_vals_to_states(vals)

    log10_field = float(file.split("_")[-1][1:-4])
    mean_X_vals[log10_field] = np.einsum("tij,ji->", states, X_op).real / times.size

plt.figure(figsize = (3,2))
plt.plot(mean_X_vals.keys(), mean_X_vals.values(), "ko")
plt.xlabel(r"$\log_{10}(J\phi)$")
plt.ylabel(r"$\braket{\bar\sigma_{\mathrm{x}}}$")
plt.tight_layout()
plt.show()
