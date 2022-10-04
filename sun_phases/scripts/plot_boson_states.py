#!/usr/bin/env python3

import os, sys, glob
import numpy as np
import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt

import scipy, scipy.optimize, scipy.special

from dicke_methods import spin_op_vec_dicke
from boson_methods import extract_avg_var_state

spin_num = 100

data_dir = "../data/spin_bosons/"
fig_dir = "../figures/spin_bosons/"
def sys_tag(state_str, dim):
    return f"{state_str}_d{dim}_N{spin_num}_"

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

cmap = matplotlib.cm.get_cmap("viridis")
marker_color = {  2 : ( "o", "k" ),
                  4 : ( "s", cmap(0.10) ),
                  6 : ( "D", cmap(0.35) ),
                  8 : ( "x", cmap(0.65) ),
                 10 : ( "+", cmap(0.90) ) }

##########################################################################################

def get_info(file):
    dim = int(file.split("_")[-3][1:])
    log10_field = float(file.split("_")[-1][1:-4])
    return { "dim" : dim, "log10_field" : log10_field }

def gamma(kk):
    return scipy.special.gamma(kk-1/2) / ( np.sqrt(np.pi) * scipy.special.gamma(kk) )

figure, axes = plt.subplots(2, 3, figsize = (8,3.5), sharex = True, sharey = True)

sub_axes = np.empty(axes.shape, dtype = object)
sub_axes[0, 0] = axes[0, 0].inset_axes([0.60, 0.45, 0.35, 0.35])  # format: x, y, dx, dy
sub_axes[0, 1] = axes[0, 1].inset_axes([0.60, 0.45, 0.35, 0.35])
sub_axes[0, 2] = axes[0, 2].inset_axes([0.60, 0.45, 0.35, 0.35])
sub_axes[1, 0] = axes[1, 0].inset_axes([0.60, 0.60, 0.35, 0.35])
sub_axes[1, 1] = axes[1, 1].inset_axes([0.05, 0.05, 0.35, 0.35])
sub_axes[1, 2] = axes[1, 2].inset_axes([0.60, 0.60, 0.35, 0.35])

dims = {}
crits = {}
state = { "X": "X", "XX": "XX", "XXI": r"XX_{\mathrm{i}}" }

for col, state_str in enumerate(["X", "XX", "XXI"]):

    files = glob.glob(data_dir + f"means_{state_str}_d*_N{spin_num}_h*")
    dims[state_str] = np.array(sorted(set([ get_info(file)["dim"] for file in files ])))
    crits[state_str] = np.zeros(dims[state_str].size)

    for dim_idx, dim in enumerate(dims[state_str]):
        log10_fields, dim_files \
            = zip(*sorted([ ( get_info(file)["log10_field"], file )
                            for file in files if get_info(file)["dim"] == dim ]))
        log10_fields = np.array(log10_fields)
        fields = 10**log10_fields
        fields_reduced = fields * (dim/2)**(1/3)

        # normalized spin operator vector
        spin = (dim-1)/2
        spin_op_vec = np.array([ np.array(op.todense()).ravel() / spin
                                 for op in spin_op_vec_dicke(dim-1) ]).T

        # collect long-time average data
        mean_mag = np.zeros(fields.size)
        mean_int = np.zeros(fields.size)
        avg_spect = np.zeros((fields.size,dim))
        var_spect = np.zeros((fields.size,dim**2))
        for idx, file in enumerate(dim_files):
            avg_state, var_state \
                = extract_avg_var_state(np.loadtxt(file, dtype = complex), dim)
            mean_mag[idx] = np.linalg.norm( avg_state.ravel().conj() @ spin_op_vec )
            mean_int[idx] = np.einsum("mnnm->", var_state).real

            var_state = np.reshape(np.transpose(var_state, [0,2,1,3]), (dim**2,)*2)
            avg_spect[idx,:] = np.linalg.eigvalsh(avg_state)[::-1]
            var_spect[idx,:] = np.linalg.eigvalsh(var_state)[::-1]

        # plot spectrum of <<\bar\rho>>
        plt.figure(figsize = (3,2.4))
        plt.semilogx(fields, avg_spect, ".")
        plt.xlabel(r"$h$")
        plt.ylabel(r"$\mathrm{spect}\,\bbk{\bar\rho}_\MF$")
        plt.tight_layout()
        plt.savefig(fig_dir + f"avg_spect_{state_str}_n{dim:02d}.pdf")
        plt.close()

        # plot spectrum of <<\bar\rho\otimes\bar\rho>>
        plt.figure(figsize = (3,2.4))
        plt.semilogx(fields, var_spect, ".")
        plt.xlabel(r"$h$")
        plt.ylabel(r"$\mathrm{spect}\,\bbk{\bar\rho\otimes\bar\rho}_\MF$")
        plt.tight_layout()
        plt.savefig(fig_dir + f"var_spect_{state_str}_n{dim:02d}.pdf")
        plt.close()

        # plot magnetization and interaction data
        marker, color = marker_color[dim]
        kwargs = dict( color = color, label = dim, zorder = -dim, markersize = 4 )
        axes[0,col].semilogx(fields, mean_mag, marker, **kwargs)
        axes[1,col].semilogx(fields, mean_int, marker, **kwargs)

        # plot insets
        kwargs.update(dict( markersize = 3 ))
        mean_mag_inset = mean_mag / gamma(dim/2)
        sub_axes[0,col].semilogx(fields_reduced, mean_mag_inset, marker, **kwargs)

        if state_str == "XX" and dim == 2: continue
        mean_int_fac = gamma(dim) if state_str != "XX" else 2*gamma(dim)
        mean_int_inset = ( mean_int - mean_int_fac ) / ( 1 - mean_int_fac )
        sub_axes[1,col].semilogx(fields_reduced, mean_int_inset, marker, **kwargs)

        # don't identify critical fields for the XX initial state
        if state_str == "XX": continue

        # locate when m_MF first hits zero
        dx = mean_mag[1:] - mean_mag[:-1]
        dh = log10_fields[1:] - log10_fields[:-1]
        slope_peak = abs(dx/dh).argmax()
        max_zero = max(abs(mean_mag[slope_peak+2:]))
        zero_start = np.argmax(mean_mag < 2*max_zero)

        # find the critical field at which m_MF = 0 by fitting to a polynomial
        indices = slice(zero_start-3, zero_start)
        fit = np.polyfit(log10_fields[indices], mean_mag[indices], 2)
        crits[state_str][dim_idx] = 10**min(np.roots(fit)[-1], log10_fields[zero_start])

    # set title for this initial state
    axes[0,col].set_title(r"$\ket{\mathrm{" + state[state_str] + r"}}$")

# labels and ticks for insets
for idx, inset in enumerate(sub_axes.ravel()):
    inset.set_xlim(0.1,10)
    inset.set_ylim(-0.1,1.1)
    inset.set_xticks([0.1,1,10])
    inset.set_yticks([0,1])

    if idx == 0: continue
    inset.set_xticklabels([])
    inset.set_yticklabels([])

# labels and ticks for main axes
for col in range(3):
    axes[1,col].set_xlabel(r"$h$")
    axes[1,col].set_xlim(0.1,10)
axes[0,0].set_ylabel(r"$\sigma_\MF$")
axes[1,0].set_ylabel(r"$\bbk{\bar{\bm s}\cdot\bar{\bm s}}_\MF$")

# make legend and save figure
handles, labels = axes[0,0].get_legend_handles_labels()
legend = axes[0,1].legend(handles, labels, ncol = len(marker_color),
                          loc = "lower center",
                          bbox_to_anchor = (0.5,1.16),
                          handletextpad = 0.1,
                          columnspacing = 0.4)
plt.subplots_adjust(hspace = 0.1, wspace = 0.15)
kwargs = dict( bbox_extra_artists = (legend,), bbox_inches = "tight" )
plt.savefig(fig_dir + f"mean_mag_int.pdf", **kwargs)
plt.close()

##################################################

def fun(dim, aa): return (dim/2)**(-aa)

for state_str in [ "X", "XXI"] :
    _dims, _crits = dims[state_str], crits[state_str]
    popt, pcov = scipy.optimize.curve_fit(fun, _dims, _crits)
    print(f"state_str, alpha:  {state_str}, {popt[0]} +/- {np.sqrt(pcov[0,0])}")

    plt.figure(figsize = (2.4,1.4))
    plt.plot(_dims, _crits, "ko", label = "mean-field")
    plt.plot(_dims, fun(_dims, *popt), "r.", label = r"fit:~$(n/2)^{-\alpha}$")
    plt.ylabel(r"$h_{\mathrm{crit}}$")
    plt.xlabel(r"$n$")
    plt.xticks(_dims)
    plt.legend(loc = "best")
    plt.savefig(fig_dir + f"crit_fields_{state_str}.pdf", bbox_inches = "tight")
    plt.close()
