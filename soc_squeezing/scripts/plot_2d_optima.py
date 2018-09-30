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

from sr87_olc_constants import recoil_energy_NU

show = "show" in sys.argv
save = "save" in sys.argv
assert(show or save)

methods = [ "OAT", "OAT_dec", "TAT" ]
dependent_variables = [ "L", "T", "phi" ]

lattice_dim = 2
method = "OAT"
dependent_variable = "L"
assert(method in methods)
assert(dependent_variable in dependent_variables)

dpi = 600
figsize = (5,4)
params = { "text.usetex" : True }
plt.rcParams.update(params)

data_dir = "../data/"
fig_dir = "../figures/"
optimization_dir = "optimization/"

# set file names
base_name = optimization_dir + f"{{}}_{method}_{dependent_variable}_{lattice_dim}D"
t_opt_base = base_name.format("t_opt")
sqz_opt_base = base_name.format("sqz_opt")
U_int_base = f"U_int_{lattice_dim}D"


# read in tunneling rates and interaction energies
def pd_read_1D(fname):
    return pd.read_csv(fname, comment = "#", squeeze = True, header = None, index_col = 0)
def pd_read_2D(fname):
    return pd.read_csv(fname, comment = "#", header = 0, index_col = 0)
J_0 = pd_read_1D(data_dir + "J_0.txt")
U_int = pd_read_2D(data_dir + U_int_base + ".txt")
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
    ylabel = "Lattice size"
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

    columns = pd.to_numeric(data.columns)
    if dependent_variable == "phi":
        columns /= np.pi

    plt.figure(figsize = figsize)

    # plot main data
    plt.pcolormesh(depths, columns, data.T,
                   cmap = plt.get_cmap("jet"), zorder = 0,
                   rasterized = True)
    plt.xlabel(r"$V_0/E_R$", zorder = 1)
    plt.ylabel(ylabel, zorder = 1)
    plt.colorbar(label = label)

    # determine values of U / J on the top axis,
    #   and plot contours of fixed U / J if appropriate
    if dependent_variable == "T":
        U_on_axis = U_J[str(columns[-1])].values
        plt.contour(depths, columns, U_J.T, U_J_ticks,
                    colors = "black", zorder = 0)
    else:
        U_on_axis = U_J.values

    # determine ticks and tick labels for U / J axis
    U_tick_vals = [ U for U in U_J_ticks
                    if U > U_on_axis[0] and U < U_on_axis[-1] ]
    U_interp = interpolate.interp1d(U_on_axis, depths)
    ax_U_ticks = [ float(U_interp(U)) for U in U_tick_vals ]

    # make U / J axis
    ax_U = plt.gca().twiny()
    ax_U.set_xlim(depths[0],depths[-1])
    ax_U.set_xticks(ax_U_ticks)
    ax_U.set_xticklabels(U_tick_vals)
    ax_U.set_xlabel(r"$U_{\mathrm{int}}/J_0$")

    plt.tight_layout()
    if save:
        plt.gca().set_rasterization_zorder(1)
        plt.savefig(fig_fname, rasterized = True, dpi = dpi)

make_plot(t_opt_base, r"$t_{\mathrm{opt}}$ (seconds)")
if method[4:] == "dec":
    make_plot(sqz_opt_base, r"$-10\log_{10}(\xi^2)$")
if dependent_variable == "T":
    make_plot(U_int_base, r"$U_{\mathrm{int}}/E_R$")

if show: plt.show()
