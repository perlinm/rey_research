#!/usr/bin/env python3

import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt

import scipy, scipy.optimize, scipy.special
import mpl_toolkits
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from dicke_methods import spin_op_x_dicke
from boson_methods import extract_avg_var_state

init_state_str = sys.argv[1]
spin_num = int(sys.argv[2])

assert( init_state_str in [ "X", "XX" ] )

data_dir = "../data/spin_bosons/"
fig_dir = "../figures/spin_bosons/"
def sys_tag(dim):
    return f"{init_state_str}_d{dim}_N{spin_num}_"

fontsize = 9
preamble = r"""
\usepackage{braket,bm}
\newcommand{\MF}{\mathrm{MF}}
\newcommand{\bbk}[1]{\langle\!\langle #1 \rangle\!\rangle}
"""
params = { "font.serif" : "Computer Modern",
           "font.size" : fontsize,
           "axes.titlesize" : fontsize,
           "axes.labelsize" : fontsize,
           "text.usetex" : True,
           "text.latex.preamble" : preamble }
plt.rcParams.update(params)
marker_color = {  2 : ( "o", "tab:blue" ),
                  4 : ( "h", "tab:orange" ),
                  6 : ( "p", "tab:green" ),
                  8 : ( "s", "tab:red" ),
                 10 : ( "^", "tab:purple" ) }

##########################################################################################

def get_info(file):
    dim = int(file.split("_")[-3][1:])
    log10_field = float(file.split("_")[-1][1:-4])
    return { "dim" : dim, "log10_field" : log10_field }

def gamma(kk):
    return scipy.special.gamma(kk-1/2) / ( np.sqrt(np.pi) * scipy.special.gamma(kk) )

if init_state_str != "XX":
    inset = mpl_toolkits.axes_grid1.inset_locator.inset_axes
    figure, axes = plt.subplots(2, figsize = (2.6,3), sharex = True, sharey = True)
    sub_axes = [ inset(axes[0], "35%", "35%", loc = "upper right"),
                 inset(axes[1], "35%", "35%", loc = "upper right") ]

files = glob.glob(data_dir + f"means_{sys_tag('*')}*")
dims = np.array(sorted(set([ get_info(file)["dim"] for file in files ])))
crits = np.zeros(dims.size)

for dim_idx, dim in enumerate(dims):
    log10_fields, dim_files \
        = zip(*sorted([ ( get_info(file)["log10_field"], file )
                        for file in files if get_info(file)["dim"] == dim ]))
    log10_fields = np.array(log10_fields)
    log10_fields_reduced = log10_fields + np.log10((dim/2)**(1/3))
    fields = 10**log10_fields
    fields_reduced = 10**log10_fields_reduced

    # normalized spin-x operator
    spin = (dim-1)/2
    sx_vec = np.array(spin_op_x_dicke(dim-1).todense()).ravel() / spin

    # collect long-time average data
    mean_sx = np.zeros(log10_fields.size)
    mean_ss = np.zeros(log10_fields.size)
    avg_spect = np.zeros((log10_fields.size,dim))
    var_spect = np.zeros((log10_fields.size,dim**2))
    for idx, file in enumerate(dim_files):
        avg_state, var_state \
            = extract_avg_var_state(np.loadtxt(file, dtype = complex), dim)
        mean_sx[idx] = ( avg_state.ravel().conj() @ sx_vec ).real
        mean_ss[idx] = np.einsum("mnnm->", var_state).real

        var_state = np.reshape(np.transpose(var_state, [0,2,1,3]), (dim**2,)*2)
        avg_spect[idx,:] = np.linalg.eigvalsh(avg_state)[::-1]
        var_spect[idx,:] = np.linalg.eigvalsh(var_state)[::-1]

    # plot spectrum of <<\bar\rho>>
    plt.figure(figsize = (3,2.4))
    plt.semilogx(fields, avg_spect, ".")
    plt.xlabel(r"$J\phi/U$")
    plt.ylabel(r"$\mathrm{spect}\,\bbk{\bar\rho}_\MF$")
    plt.tight_layout()
    plt.savefig(fig_dir + f"avg_spect_{init_state_str}_n{dim:02d}.pdf")
    plt.close()

    # plot spectrum of <<\bar\rho\otimes\bar\rho>>
    plt.figure(figsize = (3,2.4))
    plt.semilogx(fields, var_spect, ".")
    plt.xlabel(r"$J\phi/U$")
    plt.ylabel(r"$\mathrm{spect}\,\bbk{\bar\rho\otimes\bar\rho}_\MF$")
    plt.tight_layout()
    plt.savefig(fig_dir + f"var_spect_{init_state_str}_n{dim:02d}.pdf")
    plt.close()

    # skip plotting <<sx>> and <<ss>> for XX initial state
    if init_state_str == "XX": continue

    # plot main data
    marker, color = marker_color[dim]
    kwargs = dict( color = color, label = dim, zorder = -dim )
    axes[0].semilogx(fields, mean_sx, marker, **kwargs)
    axes[1].semilogx(fields, mean_ss, marker, **kwargs)

    # plot insets
    kwargs.update(dict( markersize = 3 ))
    mean_sx_inset = mean_sx / gamma(dim/2)
    mean_ss_inset = ( mean_ss - gamma(dim) ) / ( 1 - gamma(dim) )
    sub_axes[0].semilogx(fields_reduced, mean_sx_inset, marker, **kwargs)
    sub_axes[1].semilogx(fields_reduced, mean_ss_inset, marker, **kwargs)

    # locate when <<op>> first hits zero
    dx = mean_sx[1:] - mean_sx[:-1]
    dh = log10_fields[1:] - log10_fields[:-1]
    slope_peak = abs(dx/dh).argmax()
    max_zero = max(abs(mean_sx[slope_peak+2:]))
    zero_start = np.argmax(mean_sx < 2*max_zero)

    # find the critical field at which <<op>> = 0 by fitting to a polynomial
    indices = slice(zero_start-3, zero_start)
    fit = np.polyfit(log10_fields[indices], mean_sx[indices], 2)
    crits[dim_idx] = 10**min(np.roots(fit)[-1], log10_fields[zero_start])

if init_state_str == "XX": exit()

# label axes and set axis ticks
axes[0].set_ylabel(r"$\bbk{\bar\sigma_{\mathrm{x}}}_\MF$")
axes[1].set_ylabel(r"$\bbk{\bar{\bm s}\cdot\bar{\bm s}}_\MF$")
axes[1].set_xlabel(r"$J\phi/U$")
axes[1].set_xlim(0.1,10)
for axis in sub_axes:
    axis.set_xlim(0.1,10)
    axis.set_ylim(-0.1,1.1)
    axis.set_xticks([0.1,1,10])
    axis.set_yticks([0,1])
sub_axes[0].set_xticklabels(["",r"$10^0$",""])
sub_axes[1].set_xticklabels([])

# make legend and save figure
handles, labels = axes[0].get_legend_handles_labels()
legend = axes[0].legend(handles, labels, loc = "lower center",
                        bbox_to_anchor = (0.5,0.98),
                        handletextpad = 0.1,
                        ncol = dims.size, columnspacing = 0.4)
plt.subplots_adjust(hspace = 0.08)
kwargs = dict( bbox_extra_artists = (legend,), bbox_inches = "tight" )
plt.savefig(fig_dir + f"mean_sx_ss.pdf", **kwargs)
plt.close()

##################################################

def fun(dim, aa): return (dim/2)**(-aa)
popt, pcov = scipy.optimize.curve_fit(fun, dims, crits)
print("alpha:", popt, np.sqrt(pcov))
plt.figure(figsize = (2.4,1.4))
plt.plot(dims, crits, "ko", label = "mean-field")
plt.plot(dims, fun(dims, *popt), "r.", label = r"fit:~$(n/2)^{-\alpha}$")
plt.ylabel(r"$(J\phi/U)_{\mathrm{crit}}$")
plt.legend(loc = "best")
plt.savefig(fig_dir + f"crit_fields.pdf", bbox_inches = "tight")
plt.close()
