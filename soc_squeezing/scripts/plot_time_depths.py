#!/usr/bin/env python3

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
figsize = (4.2,3)
params = { "text.usetex" : True }
plt.rcParams.update(params)

data_dir = "../data/"
fig_dir = "../figures/"

size_1D = int(sys.argv[1]) # lattice size (i.e. number of lattice sites along each axis)
lattice_dim = int(sys.argv[2]) # dimensions of lattice

# squeezing protocol requirements
h_U_target = 0.05
t_J_T_cap = 0.05
U_J_cap = 10
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

def get_single_optimum_parameters(depth, confinement):
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
    zero_frame = pd.DataFrame(data = np.zeros((len(depths),len(confinements))),
                              index = depths, columns = confinements)

    U_int = zero_frame.copy(deep = True)
    phi_opt = zero_frame.copy(deep = True)
    t_opt = zero_frame.copy(deep = True)

    dd, dd_cap = 0, len(depths)
    for depth in depths:
        print("{}/{}".format(dd,dd_cap))
        dd += 1
        for confinement in confinements:
            params = get_single_optimum_parameters(depth, confinement)
            U_int.at[depth, confinement] = params[0]
            phi_opt.at[depth, confinement] = params[1]
            t_opt.at[depth, confinement] = params[2]

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
def make_plot(U_int, phi_opt, t_opt, time_ratio = 1):
    t_opt = t_opt * time_ratio
    t_opt_SI = t_opt / recoil_energy_NU

    t_J_shade_vals = np.ones(U_int.shape)
    U_J_shade_vals = np.ones(U_int.shape)

    t_J_shade_vals[t_opt.values * J_T.values.T / (2*np.pi) > t_J_T_cap] = 0
    U_J_shade_vals[(U_int.values.T / J_0.values.T).T > 10] = 0

    depths = U_int.index.values
    confinements = pd.to_numeric(U_int.columns).values

    plt.figure(figsize = figsize)

    vmax = min(t_opt_SI.values.max(), t_opt_SI_cap)
    plt.pcolormesh(depths, confinements, t_opt_SI.T, vmin = 0, vmax = vmax, zorder = 0,
                   cmap = plt.get_cmap("jet"))
    plt.colorbar(label = r"$t_{\mathrm{opt}}$ (seconds)")

    contour_level = [ 0.5 ]
    fill_regions = 2
    colors = []
    labels = []
    hatches = []
    if 0 in t_J_shade_vals:
        plt.contour(depths, confinements, t_J_shade_vals.T, contour_level,
                    colors = "black", linewidths = (2,), zorder = 1)
        plt.contourf(depths, confinements, t_J_shade_vals.T, fill_regions,
                 colors = "none", hatches = [ "\\", None, None ], zorder = 1)
        colors.append("#FFFFFF")
        labels.append(r"$t_{{\mathrm{{opt}}}} J_T/2\pi > {}$".format(t_J_T_cap))
        hatches.append("\\\\\\")
    if 0 in U_J_shade_vals:
        plt.contour(depths, confinements, U_J_shade_vals.T, contour_level,
                    colors = "black", linewidths = (2,), zorder = 1)
        plt.contourf(depths, confinements, U_J_shade_vals.T, fill_regions,
                     colors = "none", hatches = [ "/", None, None ], zorder = 1)
        colors.append("#FFFFFF")
        labels.append(r"$U_{{\mathrm{{int}}}}/J_0 > {}$".format(U_J_cap))
        hatches.append("///")
    handles = [ mpatches.Rectangle([0,0], 0, 0, edgecolor = "black",
                                   facecolor = colors[ii],
                                   hatch = hatches[ii],
                                   label = labels[ii])
                for ii in range(len(colors)) ]
    if len(handles) > 0:
        legend = plt.legend(handles = handles, ncol = len(handles), framealpha = 0,
                            loc = "upper center", bbox_to_anchor = (0.5,1.2))
        legend.set_zorder(1)

    plt.xlabel(r"$V_0/E_R$", zorder = 1)
    plt.ylabel(r"$V_T/E_R$", zorder = 1)
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
fname_suffix = "_L{}_{}D".format(size_1D,lattice_dim)
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
    plt.savefig(fig_dir + "t_opt_OAT" + fname_suffix + ".pdf",
                rasterized = True, dpi = dpi)

# determine TAT : OAT optimal squeezing time ratio
N = size_1D**lattice_dim
ratio_index = time_ratios.index.get_loc(N, method = "nearest")
time_ratio = time_ratios.iat[ratio_index]

# make and save TAT time plot
make_plot(U_int, phi_opt, t_opt, time_ratio)
if save:
    plt.gca().set_rasterization_zorder(1)
    plt.savefig(fig_dir + "t_opt_TAT" + fname_suffix + ".pdf",
                rasterized = True, dpi = dpi)

if show: plt.show()
