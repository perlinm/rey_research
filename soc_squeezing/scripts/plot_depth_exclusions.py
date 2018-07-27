#!/usr/bin/env python3

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.optimize import minimize_scalar

from dicke_methods import squeezing_OAT
from fermi_hubbard_methods import get_simulation_parameters, spatial_basis, gauged_energy
from sr87_olc_constants import g_int_LU, recoil_energy_NU

show = "show" in sys.argv
save = "save" in sys.argv
assert(show or save)

figsize = (4.2,3)
params = { "text.usetex" : True }
plt.rcParams.update(params)

data_dir = "../data/"
fig_dir = "../figures/"

sites_1D = 100 # number of lattice sites along each axis of the lattice
lattice_dimensions = [ 1, 2 ] # dimensions of lattice

# set spin-orbit coupling strength
soc_int = 2
phi = 2*np.pi/sites_1D * soc_int

# min / max primary lattice depths
depth_min = 1
depth_max = 10
depth_step_size = 0.1

# min / max confinement lattice depths
confinement_min = 40
confinement_max = 100
confinement_step_size = 2

# squeezing viability bounds
t_opt_SI_cap = 2
h_std_U_cap = 0.05
t_J_T_cap = 0.05

site_number = 400 # site number for lattice calculations

depths = np.arange(depth_min,
                   depth_max + depth_step_size/2,
                   depth_step_size)
confinements = np.arange(confinement_min,
                         confinement_max + confinement_step_size/2,
                         confinement_step_size)


##########################################################################################
# method to compute parameters which determine the viability of a squeezing protocol
##########################################################################################

def get_single_lattice_parameters(L, depth, confinement):
    L, J_0, _, K_0, _, _, energies, J_T, K_T = \
        get_simulation_parameters(L, phi, depth, confinement, site_number)

    N = np.prod(L)
    U_int = g_int_LU[1] * K_T**(3-L.size) * np.prod(K_0)
    h_std = np.std([ ( gauged_energy(q, 1, phi, L, energies)
                         - gauged_energy(q, 0, phi, L, energies) )
                       for q in spatial_basis(L) ])

    def squeezing_OAT_val(tau):
        chi_t = tau * N**(-2/3)
        return squeezing_OAT(chi_t, N)
    optimum = minimize_scalar(squeezing_OAT_val, method = "bounded", bounds = (0, 2))

    chi = h_std**2 / ( N * (N-1) * U_int / np.prod(L) )
    time_opt = optimum.x * N **(-2/3) / chi

    return h_std, U_int, time_opt, J_0[0], J_T

def get_lattice_parameters(lattice_dim):
    L = sites_1D * np.ones(lattice_dim)
    zero_frame = pd.DataFrame(data = np.zeros((len(depths),len(confinements))),
                              index = depths, columns = confinements)

    h_std = zero_frame.copy(deep = True)
    U_int = zero_frame.copy(deep = True)
    time_opt = zero_frame.copy(deep = True)
    J_0 = pd.DataFrame(data = np.zeros(len(depths)), index = depths)
    J_T = pd.DataFrame(data = np.zeros(len(confinements)), index = confinements)

    print("lattice_dim:",lattice_dim)
    dd, dd_cap = 0, len(depths)
    for depth in depths:
        print(" depth: {}/{}".format(dd,dd_cap))
        dd += 1
        for confinement in confinements:
            params = get_single_lattice_parameters(L, depth, confinement)
            h_std.at[depth, confinement] = params[0]
            U_int.at[depth, confinement] = params[1]
            time_opt.at[depth, confinement] = params[2]
            J_0.at[depth] = params[3]
            J_T.at[confinement] = params[4]

    return h_std, U_int, time_opt, J_0, J_T


##########################################################################################
# method to make a color plot summarizng viability of the squeezing protocol
##########################################################################################

x_min, x_max, dx = depth_min, depth_max, depth_step_size
y_min, y_max, dy = confinement_min, confinement_max, confinement_step_size
x_min, x_max = x_min - dx/2, x_max + dx/2
y_min, y_max = y_min - dy/2, y_max + dy/2
axis_bounds = [ x_min, x_max, y_min, y_max ]
axis_aspect = ( x_max - x_min ) / ( y_max - y_min ) * figsize[1]/figsize[0]
def make_plot(h_std, U_int, t_opt, J_T, time_ratio = 1):
    h_std_U = h_std / U_int
    t_opt = t_opt * time_ratio
    t_opt_SI = t_opt / recoil_energy_NU

    t_shade_vals = np.zeros(h_std.shape)
    tJ_hatch_vals = np.zeros(h_std.shape)
    h_hatch_vals = np.zeros(h_std.shape)

    t_shade_vals[t_opt_SI.values < t_opt_SI_cap] = 1
    h_hatch_vals[h_std_U.values < h_std_U_cap] = 1
    tJ_hatch_vals[t_opt.values * J_T.values.T / (2*np.pi) < t_J_T_cap] = 1

    depths = h_std_U.index.values
    confinements = pd.to_numeric(h_std_U.columns).values

    plt.figure(figsize = figsize)

    contour_level = [ 0.5 ]
    plt.contour(depths, confinements, t_shade_vals.T, contour_level,
                colors = "black", linewidths = (2,),)
    plt.contour(depths, confinements, h_hatch_vals.T, contour_level,
                colors = "black", linewidths = (2,))
    plt.contour(depths, confinements, tJ_hatch_vals.T, contour_level,
                colors = "black", linewidths = (2,))

    fill_regions = 2
    plt.contourf(depths, confinements, -t_shade_vals.T,
                 cmap = plt.get_cmap("binary"), vmin = -1, vmax = 1)
    plt.contourf(depths, confinements, -h_hatch_vals.T, fill_regions,
                 colors = "none", hatches = [ None, None, "/" ])
    plt.contourf(depths, confinements, -tJ_hatch_vals.T, fill_regions,
                 colors = "none", hatches = [ None, None, "\\" ])

    colors = [ "#888888", "#FFFFFF", "#FFFFFF" ]
    labels = [ r"$t_{{\mathrm{{opt}}}} > {}$ seconds".format(t_opt_SI_cap),
               r"$\tilde h/U > {}$".format(h_std_U_cap),
               r"$t_{{\mathrm{{opt}}}} J_T/2\pi > {}$".format(t_J_T_cap) ]
    hatches = [ None, "///", "\\\\\\" ]

    handles = [ mpatches.Rectangle([0,0], 0, 0, edgecolor = "black",
                                   facecolor = colors[ii],
                                   hatch = hatches[ii],
                                   label = labels[ii])
                for ii in range(3) ]

    plt.legend(handles = handles, framealpha = 1, loc = "lower right")

    plt.xlabel(r"$V_0/E_R$")
    plt.ylabel(r"$V_T/E_R$")
    plt.tight_layout()


##########################################################################################
# compute lattice parameters and make squeezing viability plots
##########################################################################################

def pd_read_1D(fname):
    return pd.read_csv(fname, comment = "#", squeeze = True, header = None, index_col = 0)
def pd_read_2D(fname):
    return pd.read_csv(fname, comment = "#", header = 0, index_col = 0)

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

# loop over all lattice dimensions and make corresponding viability figures
for lattice_dim in lattice_dimensions:

    # set data file names and header identifying this simulation
    fname_suffix = "_L{}_soc{}_{}D".format(sites_1D,soc_int,lattice_dim)
    base_fname = data_dir + "{}" + fname_suffix + ".txt"
    h_std_fname = base_fname.format("h_std")
    U_int_fname = base_fname.format("U_int")
    t_opt_fname = base_fname.format("t_opt")
    J_0_fname = data_dir + "J_0.txt"
    J_T_fname = data_dir + "J_T.txt"

    # determine whether we need to compute lattice parameters,
    #   or whether we can simply read in a data file
    compute_lattice_params = False
    if not os.path.isfile(h_std_fname):
        compute_lattice_params = True
    else:
        h_std = pd.read_csv(h_std_fname, comment = "#", header = 0, index_col = 0)
        f_depths = h_std.index.values
        f_confinements = pd.to_numeric(h_std.columns).values
        depth_test = ( f_depths[0] <= depths[0] and
                       f_depths[-1] >= depths[-1] and
                       f_depths.size >= depths.size )
        confinement_test = ( f_confinements[0] <= confinements[0] and
                             f_confinements[-1] >= confinements[-1] and
                             f_confinements.size >= confinements.size )
        if not depth_test or not confinement_test:
            compute_lattice_params = True

    # get viability parameters, writing them to a file if we compute them
    if compute_lattice_params:
        h_std, U_int, t_opt, J_0, J_T = get_lattice_parameters(lattice_dim)

        if save:
            for frame_2D, name_2D in zip([ h_std, U_int, t_opt ],
                                         [ h_std_fname, U_int_fname, t_opt_fname ]):
                with open(name_2D, "w") as f:
                    f.write(header_2D + header_units)
                frame_2D.to_csv(name_2D, mode = "a")

            for frame_1D, name_1D in zip([ J_0, J_T ], [ J_0_fname, J_T_fname ]):
                with open(name_1D, "w") as f:
                    f.write(header_1D + header_units)
                frame_1D.to_csv(name_1D, header = False, mode = "a")

    else:
        h_std = pd_read_2D(h_std_fname)
        U_int = pd_read_2D(U_int_fname)
        t_opt = pd_read_2D(t_opt_fname)
        J_T = pd_read_1D(J_T_fname)

    # make and save OAT viability plot
    make_plot(h_std, U_int, t_opt, J_T)
    if save:
        plt.savefig(fig_dir + "viability_OAT" + fname_suffix + ".pdf")

    # determine TAT : OAT optimal squeezing time ratio
    N = sites_1D**lattice_dim
    ratio_index = time_ratios.index.get_loc(N, method = "nearest")
    time_ratio = time_ratios.iat[ratio_index]

    # make and save TAT viability plot
    make_plot(h_std, U_int, t_opt, J_T, time_ratio)
    if save:
        plt.savefig(fig_dir + "viability_TAT" + fname_suffix + ".pdf")

if show: plt.show()
