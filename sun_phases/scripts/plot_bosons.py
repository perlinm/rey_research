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
def vals_to_states(vals):
    if vals.ndim == 1: vals = np.array([vals])
    states = np.empty(( vals.shape[0], spin_dim, spin_dim ), dtype = complex)
    for idx, ( mu, nu ) in enumerate(zip(*np.triu_indices(spin_dim))):
        states[:,mu,nu] = vals[:,idx]
        if mu != nu:
            states[:,nu,mu] = vals[:,idx].conj()
    return states

X_op_vec = np.array(spin_op_x_dicke(spin_dim-1).todense()).ravel() / spin
mean_X = {}
for file in glob.glob(data_dir + f"mean_state_{sys_tag}*"):
    log10_field = float(file.split("_")[-1][1:-4])
    state = vals_to_states(np.loadtxt(file, dtype = complex))
    mean_X[log10_field] = ( state.ravel().conj() @ X_op_vec ).real

results = sorted(zip(mean_X.keys(), mean_X.values()), key = lambda x : x[0])
for result in results:
    print(10**result[0], result[1])

plt.figure(figsize = (3,2))
plt.plot(mean_X.keys(), mean_X.values(), "ko")
plt.xlabel(r"$\log_{10}(J\phi)$")
plt.ylabel(r"$\braket{\bar\sigma_{\mathrm{x}}}$")
plt.tight_layout()
plt.show()
