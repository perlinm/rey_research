#!/usr/bin/env python3

##########################################################################################
# FILE CONTENTS:
# plots optimal OAT and TAT squeezing times as a function of
# primary lattice depth and system size,
# keeping confining lattice depth fixed
##########################################################################################

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.optimize import minimize_scalar

from dicke_methods import squeezing_OAT
from fermi_hubbard_methods import get_simulation_parameters
from sr87_olc_constants import g_int_LU, recoil_energy_NU

show = "show" in sys.argv
save = "save" in sys.argv
assert(show or save)

dpi = 600
figsize = (5,4)
params = { "text.usetex" : True }
plt.rcParams.update(params)

data_dir = "../data/"
fig_dir = "../figures/optimal_time_plots/"

lattice_dim = int(sys.argv[1]) # dimensions of lattice
confinement = 200 # # confinement lattice depth

# squeezing protocol requirements
h_U_target = 0.05
t_J_T_cap = 0.05
U_J_caps = [ 10, 15, 20 ]

t_J_hatch = "|"
U_J_hatches = [ "/", "\\", "-" ]

depth_min = 2
if lattice_dim == 1:
    depth_max = 8
    t_opt_SI_cap = 0.16
elif lattice_dim == 2:
    depth_max = 10
    t_opt_SI_cap = 1

max_tau = 2 # maximum value of reduced time in OAT squeezing minimization

# methods to read in 1D / 2D data
def pd_read_1D(fname):
    return pd.read_csv(fname, comment = "#", squeeze = True, header = None, index_col = 0)
def pd_read_2D(fname):
    return pd.read_csv(fname, comment = "#", header = 0, index_col = 0)

# primary and transverse tunneling rates
J_0 = pd_read_1D(data_dir + "J_0.txt")
J_T = pd_read_1D(data_dir + "J_T.txt")

# primary and transverse lattice depths
depths = J_0.index
confinements = J_T.index


##########################################################################################
# method to compute relevant parameters in an optimal squeezing protocol
##########################################################################################

def get_single_optimum_parameters(depth, size_1D):
    L = size_1D * np.ones(lattice_dim)
    _, _, _, K_0, K_T, _, _, _ = \
        get_simulation_parameters(L, depth, confinement, size_1D)

    N = np.prod(L)
    U_int = g_int_LU[1] * K_T**(3-lattice_dim) * np.prod(K_0)

    def h_std(phi): return 2**(1+lattice_dim/2)*J_0[0]*np.sin(phi/2)
    phi_opt = minimize_scalar(lambda x: abs(h_std(x)/U_int-h_U_target),
                              method = "bounded", bounds = (0, np.pi)).x

    def squeezing_OAT_val(tau):
        chi_t = tau * N**(-2/3)
        return squeezing_OAT(chi_t, N)
    tau_opt = minimize_scalar(squeezing_OAT_val,
                              method = "bounded", bounds = (0, max_tau)).x

    chi = h_std(phi_opt)**2 / ( N * (N-1) * U_int / np.prod(L) )
    t_opt = tau_opt * N **(-2/3) / chi

    return U_int, phi_opt, t_opt

def get_optimum_parameters():
    zero_frame = pd.DataFrame(data = np.zeros((len(depths),len(sizes_1D))),
                              index = depths, columns = sizes_1D)

    U_int = zero_frame.copy(deep = True)
    phi_opt = zero_frame.copy(deep = True)
    t_opt = zero_frame.copy(deep = True)

    dd, dd_cap = 0, len(depths)
    for depth in depths:
        print("{}/{}".format(dd,dd_cap))
        dd += 1
        for size_1D in sizes_1D:
            params = get_single_optimum_parameters(depth, size_1D)
            U_int.at[depth, size_1D] = params[0]
            phi_opt.at[depth, size_1D] = params[1]
            t_opt.at[depth, size_1D] = params[2]

    return U_int, phi_opt, t_opt


##########################################################################################
# method to make a plot of optimal squeezing times in the (V_0,V_T) plane
##########################################################################################

x_min, x_max, n_x = depths[0], depths[-1], len(depths)
y_min, y_max, n_y = confinements[0], confinements[-1], len(confinements)
dx, dy = ( x_max - x_min ) / n_x, ( y_max - y_min ) / n_y
x_min, x_max = x_min - dx/2, x_max + dx/2
y_min, y_max = y_min - dy/2, y_max + dy/2
axis_bounds = [ x_min, x_max, y_min, y_max ]
axis_aspect = ( x_max - x_min ) / ( y_max - y_min ) * figsize[1]/figsize[0]
def make_plot(U_int, phi_opt, t_opt, time_ratios = None):
    N_vals = pd.to_numeric(t_opt.columns)**lattice_dim
    if type(time_ratios) != type(None):
        ratios = pd.Series(index = N_vals)
        ratios.fillna(time_ratios, inplace = True)
        ratios.rename(np.log, inplace = True)
        ratios.interpolate(limit_direction = "both", inplace = True)
        ratios.rename(np.exp, inplace = True)
        N_ratios = ratios.values
    else:
        N_ratios = np.ones(len(N_vals))
    t_opt_SI = t_opt.values * N_ratios / recoil_energy_NU
    U_J_vals = (U_int.values.T / J_0.values.T).T

    depths = U_int.index.values
    sizes_1D = pd.to_numeric(U_int.columns).values

    plt.figure(figsize = figsize)

    plt.pcolormesh(depths, sizes_1D, t_opt_SI.T, vmin = 0, vmax = t_opt_SI_cap,
                   cmap = plt.get_cmap("jet"), zorder = 0)
    plt.colorbar(label = r"$t_{\mathrm{opt}}$ (seconds)")

    contour_level = [ 0.5 ]
    fill_regions = 2
    labels = []
    hatches = []

    # shade U_J exclusion zones
    for U_J_cap, hatch in zip(U_J_caps, U_J_hatches):
        U_J_shade_vals = np.ones(U_int.shape)
        U_J_shade_vals[U_J_vals > U_J_cap] = 0
        plt.contour(depths, sizes_1D, U_J_shade_vals.T, contour_level,
                    colors = "black", linewidths = (2,), zorder = 1)
        plt.contourf(depths, sizes_1D, U_J_shade_vals.T, fill_regions,
                     colors = "none", hatches = [ hatch, None, None ], zorder = 1)
        labels.append(r"$U_{{\mathrm{{int}}}}/J_0 > {}$".format(U_J_cap))
        hatches.append(4*hatch)

    # make fake handles for the legend
    handles = [ mpatches.Rectangle([0,0], 0, 0,
                                   edgecolor = "black", facecolor = "#FFFFFF",
                                   hatch = hatches[ii], label = labels[ii])
                for ii in range(len(labels)) ]
    legend = plt.legend(handles = handles, ncol = 3, framealpha = 0,
                        loc = "upper center", bbox_to_anchor = (0.5,1.15))
    legend.set_zorder(1)

    plt.xlabel(r"$V_0/E_R$", zorder = 1)
    plt.ylabel(r"$L$", zorder = 1)
    plt.xlim(depth_min,depth_max)
    plt.tight_layout(pad = 0.5)


##########################################################################################
# compute optimal squeezing protocol parameters and make optimal squeezing time plots
##########################################################################################

time_OAT_fname = data_dir + "time_OAT.txt"
time_TAT_fname = data_dir + "time_TAT.txt"
time_OAT_vals = pd_read_1D(time_OAT_fname)
time_TAT_vals = pd_read_1D(time_TAT_fname)
time_ratios = time_TAT_vals / time_OAT_vals

header_2D = "# first column = primary lattice depths\n"
header_2D += "# first row = confining lattice depths\n"
header_1D = "# first column = lattice depth\n"
header_units = "# values in units with the recoil energy"
header_units += r" E_R \approx 3.47 x 2\pi kHz equal to 1" + "\n"

# set data file names and header identifying this simulation
fname_suffix = "_{}D_T{}".format(lattice_dim,confinement)
base_fname = data_dir + "{}" + fname_suffix + ".txt"
U_int_fname = base_fname.format("U_int")
phi_opt_fname = base_fname.format("phi_opt")
t_opt_fname = base_fname.format("t_opt")

# get optimal parameters, writing them to a file if we compute them
if not os.path.isfile(U_int_fname):
    U_int, phi_opt, t_opt = get_optimum_parameters()

    if save:
        for data, fname in zip([ U_int, phi_opt, t_opt ],
                               [ U_int_fname, phi_opt_fname, t_opt_fname ]):
            with open(fname, "w") as f:
                f.write(header_2D + header_units)
            data.to_csv(fname, mode = "a")

else:
    U_int = pd_read_2D(U_int_fname)
    phi_opt = pd_read_2D(phi_opt_fname)
    t_opt = pd_read_2D(t_opt_fname)

# make and save OAT time plot
make_plot(U_int, phi_opt, t_opt)
if save:
    plt.gca().set_rasterization_zorder(1)
    plt.savefig(fig_dir + "t_opt" + fname_suffix + "_OAT.pdf",
                rasterized = True, dpi = dpi)

# make and save TAT time plot
make_plot(U_int, phi_opt, t_opt, time_ratios)
if save:
    plt.gca().set_rasterization_zorder(1)
    plt.savefig(fig_dir + "t_opt" + fname_suffix + "_TAT.pdf",
                rasterized = True, dpi = dpi)

if show: plt.show()
