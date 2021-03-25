#!/usr/bin/env python3

import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt

import scipy, scipy.optimize, scipy.special
import mpl_toolkits
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from dicke_methods import spin_op_x_dicke

init_state_str = sys.argv[1]
spin_num = int(sys.argv[2])

assert( init_state_str in [ "X" ] )

data_dir = "../data/spin_bosons/"
fig_dir = "../spin_model_paper/figures/"
def sys_tag(spin_dim):
    return f"{init_state_str}_d{spin_dim}_N{spin_num}"

fontsize = 9
preamble = r"""
\usepackage{braket,bm}
\newcommand{\MFT}{\mathrm{MFT}}
\newcommand{\bbk}[1]{\langle\!\langle #1 \rangle\!\rangle}
"""
params = { "font.family" : "serif",
           "font.serif" : "Computer Modern",
           "font.size" : fontsize,
           "axes.titlesize" : fontsize,
           "axes.labelsize" : fontsize,
           "text.usetex" : True,
           "text.latex.preamble" : preamble }
plt.rcParams.update(params)
markers = [ "o", "h", "p", "s", "^" ]

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

inset = mpl_toolkits.axes_grid1.inset_locator.inset_axes
figure, axes = plt.subplots(2, figsize = (2.6,3), sharex = True, sharey = True)
sub_axes = [ inset(axes[0], "35%", "35%", loc = "upper right"),
             inset(axes[1], "35%", "35%", loc = "upper right") ]

files = glob.glob(data_dir + f"mean_state_{sys_tag('*')}*")
dims = np.array(sorted(set([ get_info(file)["dim"] for file in files ])))
log10_crits = np.zeros(dims.size)
mean_x_max = np.zeros(dims.size)

for dim_idx, dim in enumerate(dims):
    log10_fields, dim_files \
        = zip(*sorted([ ( get_info(file)["log10_field"], file )
                        for file in files if get_info(file)["dim"] == dim ]))
    log10_fields = np.array(log10_fields)
    log10_fields_reduced = log10_fields + np.log10((dim/2)**(1/3))

    # collect long-time averages
    spin = (dim-1)/2
    x_op_vec = np.array(spin_op_x_dicke(dim-1).todense()).ravel() / spin
    mean_x = np.zeros(log10_fields.size)
    mean_ss = np.zeros(log10_fields.size)
    for idx, file in enumerate(dim_files):
        state = vals_to_states(np.loadtxt(file, dtype = complex), dim)
        state_vec = state.ravel()
        mean_x[idx] = ( state_vec.conj() @ x_op_vec ).real
        mean_ss[idx] =  ( state_vec.conj() @ state_vec ).real

    # locate when <<sx>> first hits zero
    dx = mean_x[1:] - mean_x[:-1]
    dh = log10_fields[1:] - log10_fields[:-1]
    slope_peak = abs(dx/dh).argmax()
    max_zero = max(abs(mean_x[slope_peak+2:]))
    zero_start = np.argmax(mean_x < 2*max_zero)

    # find the critical field at which <<sx>> = 0 by fitting to a polynomial
    indices = slice(zero_start-3, zero_start)
    fit = np.polyfit(log10_fields[indices], mean_x[indices], 2)
    log10_crits[dim_idx] = min(np.roots(fit)[-1], log10_fields[zero_start])

    # determine \lim_{h->0} <<sx>>
    if dim == 2:
        log10_fields_2 = log10_fields
        mean_x_2 = mean_x
    equiv_mean_x_2 = np.interp(log10_fields_reduced[0], log10_fields_2, mean_x_2)
    mean_x_max[dim_idx] = mean_x[0] / equiv_mean_x_2

    # plot main data
    axes[0].plot(log10_fields, mean_x, markers[dim_idx], label = dim, zorder = -dim)
    axes[1].plot(log10_fields, mean_ss, markers[dim_idx], label = dim, zorder = -dim)

    # plot insets
    fac_x = (dim-1)**0.6
    fac_ss = np.sqrt(np.pi) * scipy.special.gamma(dim) / scipy.special.gamma(dim-1/2)
    sub_axes[0].plot(log10_fields_reduced, mean_x*fac_x, ".", label = dim, zorder = -dim)
    sub_axes[1].plot(log10_fields_reduced, mean_ss*fac_ss-1, ".", label = dim, zorder = -dim)

# label axes and set axis ticks
axes[0].set_ylabel(r"$\bbk{\bar\sigma_{\mathrm{x}}}_\MFT$")
axes[1].set_ylabel(r"$\bbk{\bar{\bm s}\cdot\bar{\bm s}}_\MFT$")
axes[1].set_xlabel(r"$\log_{10}(J\phi/U)$")
for axis in sub_axes:
    axis.set_xticks([])
    axis.set_yticks([0,1])

# make legend and save figure
handles, labels = axes[0].get_legend_handles_labels()
legend = axes[0].legend(handles, labels, loc = "lower center",
                        bbox_to_anchor = (0.5,0.98),
                        handletextpad = 0.1,
                        ncol = dims.size, columnspacing = 0.4)
plt.subplots_adjust(hspace = 0.08)
kwargs = dict( bbox_extra_artists = (legend,), bbox_inches = "tight" )
plt.savefig(fig_dir + "mean_x_ss.pdf", **kwargs)

##################################################

figure, axes = plt.subplots(2, figsize = (2.4,2.8), sharex = True)

crits = 10**log10_crits
def fun(dim, aa): return (dim/2)**(-aa)
popt, pcov = scipy.optimize.curve_fit(fun, dims, crits)
print(popt, np.sqrt(pcov))
axes[0].plot(dims, crits, "ko", label = "MFT")
axes[0].plot(dims, fun(dims, *popt), "r.", label = r"fit:~$(n/2)^\alpha$")

def fun(dim, bb): return (dim-1)**(-bb)
popt, pcov = scipy.optimize.curve_fit(fun, dims, mean_x_max)
print(popt, np.sqrt(pcov))
axes[1].plot(dims, mean_x_max, "ko", label = "MFT")
axes[1].plot(dims, fun(dims, *popt), "r.", label = r"fit:~$(2s)^\beta$")

axes[0].set_ylabel(r"$(J\phi/U)_{\mathrm{crit}}$")
axes[1].set_ylabel(r"$\bbk{\bar\sigma_{\mathrm{x}}}_\MFT^0$")
axes[1].set_xlabel(r"spin dimension $n$")
axes[1].set_xticks(dims)

for axis in axes: axis.legend(loc = "best")
plt.subplots_adjust(hspace = 0.08)
plt.savefig(fig_dir + "crit_fields.pdf", bbox_inches = "tight")
