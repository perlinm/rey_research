#!/usr/bin/env python3

##########################################################################################
# FILE CONTENTS:
# plots optimal OAT and TAT squeezing parameters as a function of
# primary lattice depth and either confining lattice depth, system size, or SOC angle
##########################################################################################

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy import matlib
from scipy import interpolate
from matplotlib import gridspec

from sr87_olc_constants import recoil_energy_NU


show = "show" in sys.argv
save = "save" in sys.argv
assert(show or save)

methods = [ "OAT", "OAT_dec", "TAT" ]
dependent_variables = [ "L", "T", "phi" ]

lattice_dim = 2
method = "OAT_dec"
dependent_variable = "L"
assert(method in methods)
assert(dependent_variable in dependent_variables)

data_dir = "../data/"
fig_dir = "../figures/"
optimization_dir = "optimization/"

decoherence_is_on = method[4:] == "dec"
sqz_subplot = ( dependent_variable == "L" and not decoherence_is_on )

dpi = 600
if sqz_subplot:
    figsize = (7,4)
else:
    figsize = (5,4)
params = { "text.usetex" : True }
plt.rcParams.update(params)

# methods to read in 1-D and 2-D data
def pd_read_1D(fname):
    return pd.read_csv(fname, comment = "#", squeeze = True, header = None, index_col = 0)
def pd_read_2D(fname):
    return pd.read_csv(fname, comment = "#", header = 0, index_col = 0)

def to_dB(vals): return 10*np.log10(vals)

# set file names
base_name_2D = optimization_dir + f"{{}}_{method}_{dependent_variable}_{lattice_dim}D"
t_opt_base = base_name_2D.format("t_opt")
sqz_opt_base = base_name_2D.format("sqz_opt")
U_int_base = f"U_int_{lattice_dim}D"

# read in tunneling rates and interaction energies
J_0 = pd_read_1D(data_dir + "J_0.txt")
U_int = pd_read_2D(data_dir + U_int_base + ".txt")
if sqz_subplot:
    sqz_opt = -to_dB(pd_read_1D(data_dir + f"sqz_{method}.txt"))

depths = J_0.index
U_J = (U_int.T / J_0.T).T

# if we only have one confining depth, determine what it is and restrict U/J to its value
if dependent_variable is not "T":
    with open(data_dir + t_opt_base + ".txt", "r") as f:
        line = ""
        while "confining depth" not in line:
            line = f.readline()
        confining_depth = line.split()[-1]
    U_J = U_J[confining_depth]

U_J_max = U_J.values.max()
U_J_ticks = np.array(range(int(U_J_max)))


##########################################################################################
# method to make 2D plot
##########################################################################################

if dependent_variable == "L":
    ylabel = r"Linear lattice size ($\ell$)"
if dependent_variable == "T":
    ylabel = r"$V_T/E_R$"
if dependent_variable == "phi":
    ylabel = r"$\phi/\pi$"

def make_plot(base_name, label):
    data_fname = data_dir + base_name + ".txt"
    fig_fname = fig_dir + base_name + ".pdf"

    data = pd_read_2D(data_fname)
    if "t_opt" in base_name:
        data /= recoil_energy_NU
    if "sqz_opt" in base_name:
        data = -to_dB(data)

    columns = pd.to_numeric(data.columns)
    if dependent_variable == "phi":
        columns /= np.pi

    plt.figure(figsize = figsize)

    if sqz_subplot:
        grid = gridspec.GridSpec(1, 2, width_ratios = [ 1, 3 ])
        ax_sqz = plt.subplot(grid[0])
        ax_data = plt.subplot(grid[1])
        ax_data.set_yticklabels([])

        sqz_interp = interpolate.interp1d(sqz_opt.index, sqz_opt.values)
        sqz_vals = [ float(sqz_interp(L**2)) for L in columns ]
        ax_sqz.plot(sqz_vals, columns, "k")
        ax_sqz.set_ylim(columns[0], columns[-1])
        ax_sqz.set_xlim(sqz_vals[0], sqz_vals[-1])
        ax_sqz.set_xlabel(r"$-10\log_{10}(\xi_{\mathrm{opt}}^2)$")
        ax_sqz.set_ylabel(ylabel)

    else:
        grid = gridspec.GridSpec(1,1)
        ax_data = plt.subplot(grid[0])
        ax_data.set_ylabel(ylabel)

    # plot main data
    mesh = ax_data.pcolormesh(depths, columns, data.T,
                              cmap = plt.get_cmap("jet"),
                              zorder = 0, rasterized = True)
    plt.xlabel(r"Lattice depth ($V_0/E_R$)", zorder = 1)
    plt.colorbar(mesh, label = label, format = "%.1f")

    # determine values of U / J on the top axis,
    #   and plot contours of fixed U / J if appropriate
    if dependent_variable == "T":
        U_on_axis = U_J[str(columns[-1])].values
        ax_data.contour(depths, columns, U_J.T, U_J_ticks,
                        colors = "black", zorder = 0)
    else:
        U_on_axis = U_J.values

    # determine ticks and tick labels for U / J axis
    U_tick_vals = [ U for U in U_J_ticks
                    if U > U_on_axis[0] and U < U_on_axis[-1] ]
    U_interp = interpolate.interp1d(U_on_axis, depths)
    ax_U_ticks = [ float(U_interp(U)) for U in U_tick_vals ]

    # make U / J axis
    ax_U = ax_data.twiny()
    ax_U.set_xlim(depths[0],depths[-1])
    ax_U.set_xticks(ax_U_ticks)
    ax_U.set_xticklabels(U_tick_vals)
    ax_U.set_xlabel(r"$U_{\mathrm{int}}/J_0$")

    plt.tight_layout()
    if save:
        plt.gca().set_rasterization_zorder(1)
        plt.savefig(fig_fname, rasterized = True, dpi = dpi)

make_plot(t_opt_base, r"$t_{\mathrm{opt}}$ (seconds)")
if decoherence_is_on:
    make_plot(sqz_opt_base, r"$-10\log_{10}(\xi_{\mathrm{opt}}^2)$")
if dependent_variable == "T":
    make_plot(U_int_base, r"$U_{\mathrm{int}}/E_R$")

if show: plt.show()
