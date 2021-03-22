#!/usr/bin/env python3

import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt

from dicke_methods import spin_op_x_dicke

init_state_str = sys.argv[1]
spin_num = int(sys.argv[2])

assert( init_state_str in [ "X" ] )

figsize = (3,2)

data_dir = "../data/spin_bosons/"
fig_dir = "../figures/spin_bosons/"
def sys_tag(spin_dim):
    return f"{init_state_str}_d{spin_dim:02d}_N{spin_num}"

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

def get_info(file):
    dim = int(file.split("_")[-3][1:])
    log10_field = float(file.split("_")[-1][1:-4])
    return { "dim" : dim, "log10_field" : log10_field }

# convert upper triangle of density matrix to full density matrix
def vals_to_states(vals, dim):
    if vals.ndim == 1: vals = np.array([vals])
    states = np.empty(( vals.shape[0], dim, dim ), dtype = complex)
    for idx, ( mu, nu ) in enumerate(zip(*np.triu_indices(dim))):
        states[:,mu,nu] = vals[:,idx]
        if mu != nu:
            states[:,nu,mu] = vals[:,idx].conj()
    return states

plt.figure(figsize = figsize)

files = glob.glob(data_dir + f"mean_state_{sys_tag('*')}*")
dims = sorted(set([ get_info(file)["dim"] for file in files ]))

for dim in dims:
    dim_files = [ file for file in files if get_info(file)["dim"] == dim ]

    spin = (dim-1)/2
    X_op_vec = np.array(spin_op_x_dicke(dim-1).todense()).ravel() / spin
    mean_X = {}
    for file in dim_files:
        log10_field = float(file.split("_")[-1][1:-4])
        state = vals_to_states(np.loadtxt(file, dtype = complex), dim)
        mean_X[log10_field] = ( state.ravel().conj() @ X_op_vec ).real

    plt.plot(mean_X.keys(), mean_X.values(), "o", label = dim)

plt.xlabel(r"$\log_{10}(J\phi)$")
plt.ylabel(r"$\braket{\bar\sigma_{\mathrm{x}}}$")
plt.legend(loc = "best")
plt.tight_layout()
plt.show()
