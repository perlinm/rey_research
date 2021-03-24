#!/usr/bin/env python3

import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt

import scipy, scipy.optimize

from dicke_methods import spin_op_x_dicke

init_state_str = sys.argv[1]
spin_num = int(sys.argv[2])

assert( init_state_str in [ "X" ] )

figsize = (3,2)
big_figsize = (3,3.5)

data_dir = "../data/spin_bosons/"
fig_dir = "../figures/spin_bosons/"
def sys_tag(spin_dim):
    return f"{init_state_str}_d{spin_dim}_N{spin_num}"

if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

fontsize = 10
preamble = r"""
\usepackage{braket,bm}
"""
params = { "font.family" : "serif",
           "font.serif" : "Computer Modern",
           "font.size" : fontsize,
           "axes.titlesize" : fontsize,
           "axes.labelsize" : fontsize,
           "text.usetex" : True,
           "text.latex.preamble" : preamble }
plt.rcParams.update(params)

def op_text(text):
    return r"$\braket{\!\braket{" + text + r"}\!}_{\mathrm{MFT}}$"

##########################################################################################

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

figure, axes = plt.subplots(2, figsize = big_figsize, sharex = True)

files = glob.glob(data_dir + f"mean_state_{sys_tag('*')}*")
dims = np.array(sorted(set([ get_info(file)["dim"] for file in files ])))
log10_crits = np.zeros(dims.size)
log10_errs = np.zeros(dims.size)

for dim_idx, dim in enumerate(dims):
    log10_fields, dim_files \
        = zip(*sorted([ ( get_info(file)["log10_field"], file )
                        for file in files if get_info(file)["dim"] == dim ]))
    log10_fields = np.array(log10_fields)

    spin = (dim-1)/2
    x_op_vec = np.array(spin_op_x_dicke(dim-1).todense()).ravel() / spin
    mean_x = np.zeros(log10_fields.size)
    mean_ss = np.zeros(log10_fields.size)
    for idx, file in enumerate(dim_files):
        state = vals_to_states(np.loadtxt(file, dtype = complex), dim)
        state_vec = state.ravel()
        mean_x[idx] = ( state_vec.conj() @ x_op_vec ).real
        mean_ss[idx] =  ( state_vec.conj() @ state_vec ).real

    axes[0].plot(log10_fields, mean_x, "o", label = dim, zorder = -dim)
    axes[1].plot(log10_fields, mean_ss, "o", label = dim, zorder = -dim)

    # locate when <<sx>> first hits "zero"
    dx = mean_x[1:] - mean_x[:-1]
    dh = log10_fields[1:] - log10_fields[:-1]
    slope_peak = abs(dx/dh).argmax()
    max_zero = max(abs(mean_x[slope_peak+2:]))
    zero_start = np.argmax(mean_x < 2*max_zero)

    # find the critical field at which <<sx>> = 0 by fitting to a polynomial
    indices = slice(zero_start-3, zero_start)
    fit = np.polyfit(log10_fields[indices], mean_x[indices], 2)
    log10_crits[dim_idx] = min(np.roots(fit)[-1], log10_fields[zero_start])
    log10_errs[dim_idx] = log10_fields[zero_start] - log10_fields[zero_start-1]

axes[0].set_ylabel(op_text(r"\bar\sigma_{\mathrm{x}}"))
axes[1].set_ylabel(op_text(r"\bar{\bm s}\cdot\bar{\bm s}"))
axes[1].set_xlabel(r"$\log_{10}(J\phi/U)$")
for axis in axes: axis.set_ylim(-0.05, 1.05)
axes[0].legend(loc = "best", framealpha = 1)
plt.tight_layout()
plt.savefig(fig_dir + "mean_x_ss.pdf")

##################################################

crits = 10**log10_crits
crits_min_max = 10**np.array([ log10_crits - log10_errs,
                               log10_crits + log10_errs ])
crit_errs = abs(crits - crits_min_max)

def fun(nn, aa): return (nn/2)**aa
popt, pcov = scipy.optimize.curve_fit(fun, dims, crits)

plt.figure(figsize = figsize)
plt.errorbar(dims, crits, yerr = crit_errs, fmt = "ko", label = "MFT", zorder = -1)
plt.plot(dims, fun(dims, *popt), "r.", label = r"fit: $(n/2)^\alpha$")
plt.xlabel(r"spin dimension $n$")
plt.ylabel(r"$(J\phi/U)_{\mathrm{crit}}$")

handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], loc = "best")
plt.tight_layout()
plt.savefig(fig_dir + "crit_fields.pdf")
