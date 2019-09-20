#!/usr/bin/env python3

##########################################################################################
# FILE CONTENTS:
# makes the 2-D squeezing figures for the SOC/squeezing paper
##########################################################################################

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import interpolate
from matplotlib import gridspec

from correlator_methods import squeezing_ops
from squeezing_methods import squeezing_from_correlators
from sr87_olc_constants import recoil_energy_NU

confining_depth = 60
data_dir = "../data/"
fig_dir = "../figures/"

OAT, TAT, TNT = "OAT", "TAT", "TNT"
NOAT = TAT # "not one-axis twisting"
sqz_methods = [ OAT, NOAT ]

depth_min, depth_max = 2, 7

depths_dec = np.arange(20,81,2)/10 # lattice depths in plots
sizes_dec = np.arange(10,101,5) # linear lattice sizes in plots
phi_cutoff = 10 # minimum value of (\phi/\pi)^{-1}

dpi = 600

params = { "font.family" : "sans-serif",
           "font.serif" : "Computer Modern",
           "text.usetex" : True }
plt.rcParams.update(params)

# define some common axis labels that we will use several times
depth_label = r"Lattice depth ($V_0/E_{\mathrm{R}}$)"
size_label = r"Linear lattice size ($\ell$)"
sqz_label = r"Squeezing (dB)"
time_label = r"Time"
U_J_label = r"$U/J$"
phi_label = r"$\log_{10}(\phi/\pi)$"

# read in 1-D or 2-D data files
def pd_read_1D(fname):
    return pd.read_csv(fname, comment = "#", squeeze = True, header = None, index_col = 0)
def pd_read_2D(fname):
    data = pd.read_csv(fname, comment = "#", header = 0, index_col = 0)
    data.columns = pd.to_numeric(data.columns)
    return data

J_0 = pd_read_1D(data_dir + "J_0.txt")
U_int = pd_read_2D(data_dir + "U_int_2D.txt")
all_depths = J_0.index
U_J = (U_int.T / J_0.T).T[confining_depth]
U_J_interp = interpolate.interp1d(U_J.values, all_depths)
U_J_ticks = np.array(range(int(U_J.values.max()))) + 1
depth_U_ticks = [ float(U_J_interp(val)) for val in U_J_ticks ]

# make a secondary axis marking values of U/J (against lattice depths)
def make_U_axis(axis):
    ax = axis.twiny()
    ax.set_xlim(all_depths[0],all_depths[-1])
    ax.set_xticks(depth_U_ticks)
    ax.set_xticklabels(U_J_ticks)
    ax.set_xlabel(U_J_label)
    ax.set_xlim(depth_min, depth_max)
    return ax

# add text to a sub-plot
method_box = dict(boxstyle = "round", facecolor = "white", alpha = 1)
def add_text(axis, text):
    axis.text(0.9, 0.1, text, transform = axis.transAxes, bbox = method_box,
              verticalalignment = "bottom", horizontalalignment = "right")

def to_dB(vals): return 10*np.log10(vals) # convert values to decibels


##########################################################################################
# squeezing without decoherence
##########################################################################################

figsize = (6.5,3)

# set up figure panel and linestyles
ax = {}
fig, ( ax_sqz, ax[OAT], ax[NOAT], cax ) \
    = plt.subplots(figsize = figsize, ncols = 4,
                   gridspec_kw = { "width_ratios" : [1,2,2,0.1] })
lines = { OAT : "k-",
          NOAT : "k--" }

# exctract data to plot
sqz_vals = {}
t_opt = {}
for method in sqz_methods:
    sqz_opt = -to_dB(pd_read_1D(data_dir + f"sqz_{method}.txt"))
    t_opt[method] = pd_read_2D(data_dir + f"optimization/t_opt_{method}_L_2D.txt")
    t_opt[method] /= recoil_energy_NU
    depths, sizes = t_opt[method].index, np.array(t_opt[method].columns)

    err = abs(depths[1] - depths[0]) / 2
    use_depths = (depths >= depth_min-err) & (depths <= depth_max+err)

    sqz_interp = interpolate.interp1d(sqz_opt.index, sqz_opt.values)
    sqz_vals[method] = np.array([ float(sqz_interp(L**2)) for L in sizes ])

sqz_min = min([ sqz_vals[method].min() for method in sqz_methods ])
sqz_max = max([ sqz_vals[method].max() for method in sqz_methods ])
t_min = min([ t_opt[method].values.min() for method in sqz_methods ])
t_max = max([ t_opt[method].values.max() for method in sqz_methods ])

faster_NOAT = t_opt[NOAT] > t_opt[OAT]

# plot data
for method in sqz_methods:
    ax_sqz.plot(sqz_vals[method], sizes, lines[method], label = method)

    data = t_opt[method].values[use_depths,:].T
    mesh = ax[method].pcolormesh(depths[use_depths], sizes, data,
                                 cmap = plt.get_cmap("twilight_shifted"),
                                 vmin = 0, vmax = t_max,
                                 zorder = 0, rasterized = True)

    ax[method].contour(depths[use_depths], sizes,
                       faster_NOAT.values[use_depths,:].T,
                       levels = 0, colors = [ "k" ], linestyles = "dotted")

    ax[method].set_xlabel(depth_label, zorder = 1)
    ax[method].set_yticklabels([])
    make_U_axis(ax[method])

    ax[method].set_xlim(depth_min, depth_max)

add_text(ax_sqz, r"{\bf (a)}")
add_text(ax[OAT], r"{\bf (b)} OAT")
add_text(ax[NOAT], f"{{\\bf (c)}} {NOAT}")

ax_sqz.legend(loc = "best", framealpha = 1)
cbar = fig.colorbar(mesh, cax = cax, label = time_label + " (sec)", format = "%.1f")
cbar_ticks = [ int(tick) for tick in cbar.get_ticks() if tick == int(tick) ]
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels(cbar_ticks)

# clean up figure
ax_sqz.set_xlim(10, 35)
ax_sqz.set_ylim(sizes[0], sizes[-1])
ax_sqz.set_xlabel(sqz_label, zorder = 1)
ax_sqz.set_ylabel(size_label, zorder = 1)

plt.tight_layout()
plt.gca().set_rasterization_zorder(1)
plt.savefig(fig_dir + "optimization/t_opt_L_2D.pdf",
            rasterized = True, dpi = dpi)
plt.close()


##########################################################################################
# squeezing with decoherence
##########################################################################################

def get_sqz_floor(lattice_depth, lattice_size):

    file_name = data_dir + f"trunc/{NOAT}_{lattice_depth}_{lattice_size}.txt"
    spin_num = int(lattice_size)**2

    correlators = {}
    # if we have previously computed the squeezing floor, use it
    sqz_floor = None
    with open(file_name, "r") as f:
        for line in f:
            if "sqz_floor" in line:
                sqz_floor = float(line.split()[-1])

    if sqz_floor != None: return sqz_floor

    # otherwise, calculate the squeezing floor
    derivs = np.loadtxt(file_name, dtype = complex)
    order_cap = derivs.shape[1]

    times = np.linspace(0, 2, 100) * spin_num**(-2/3)
    times_k = np.array([ times**kk for kk in range(order_cap) ])

    # find squeezing floor at all (reasonable) orders
    orders = np.arange(order_cap//2,order_cap+1)
    sqz_floors = np.zeros(orders.size)
    for order_idx, order in enumerate(orders):
        for op_idx, sqz_op in enumerate(squeezing_ops):
            correlators[sqz_op] = derivs[op_idx,:order] @ times_k[:order,:]

        sqz = squeezing_from_correlators(spin_num, correlators, in_dB = True)

        # get index of first local maximum and inflection point
        d_sqz = sqz[1:] - sqz[:-1] # first derivative
        d_sqz_idx = np.argmax(d_sqz < 0) # index of first local maximum
        dd_sqz = d_sqz[1:] - d_sqz[:-1] # second derivative
        dd_sqz_idx = np.argmax(dd_sqz > 0) # index of first inflection point

        # if either index is zero, it means we did not actually find what we needed
        if  d_sqz_idx == 0:  d_sqz_idx = sqz.size
        if dd_sqz_idx == 0: dd_sqz_idx = sqz.size

        # the peak is at whichever event occurred first
        peak_idx = min(d_sqz_idx, dd_sqz_idx+1)

        # if we have no nans before the peak, use the squeezing value at the peak
        # otherwise, use the maximum pre-nan squeezing value
        if not np.isnan(sqz[:peak_idx+1]).any():
            sqz_floors[order_idx] = sqz[peak_idx]
        else:
            nan_idx = np.argmax(np.isnan(sqz))
            sqz_floors[order_idx] = sqz[:nan_idx].max()

    # save squeezing floor to data file
    sqz_floor = sqz_floors.max()
    with open(file_name, "a") as f:
        f.write(f"# sqz_floor: {sqz_floor}\n")

    return sqz_floor

figsize = (6,3)

# get squeezing data
sqz = {}
sqz[NOAT] = np.zeros((depths_dec.size,sizes_dec.size))
for idx, _ in np.ndenumerate(sqz[NOAT]):
    sqz[NOAT][idx] = get_sqz_floor(depths_dec[idx[0]], sizes_dec[idx[1]])

depths = { NOAT : depths_dec }
sizes = { NOAT : sizes_dec }

sqz[OAT] = -to_dB(pd_read_2D(data_dir + f"optimization/sqz_opt_OAT_dec_L_2D.txt"))
depths[OAT], sizes[OAT], sqz[OAT] \
    = sqz[OAT].index, sqz[OAT].columns, sqz[OAT].values

for method in sqz_methods:
    err = abs(depths[method][1] - depths[method][0]) / 2
    use_depths = (depths[method] >= depth_min-err) & (depths[method] <= depth_max+err)
    depths[method], sqz[method] = depths[method][use_depths], sqz[method][use_depths,:]

sqz_min = min([ sqz[method].min() for method in sqz_methods ])
sqz_max = max([ sqz[method].max() for method in sqz_methods ])

# set up figure panels
ax = {}
fig, ( ax[OAT], ax[NOAT], cax ) \
    = plt.subplots(figsize = figsize, ncols = 3,
                   gridspec_kw = { "width_ratios" : [1,1,0.05] })

# plot data
for method in sqz_methods:
    mesh = ax[method].pcolormesh(depths[method], sizes[method], sqz[method].T,
                                 cmap = plt.get_cmap("jet"),
                                 vmin = sqz_min, vmax = sqz_max,
                                 zorder = 0, rasterized = True)
    ax[method].set_xlabel(depth_label, zorder = 1)
    make_U_axis(ax[method])

    ax[method].set_xlim(depth_min, depth_max)

add_text(ax[OAT], r"{\bf (a)} OAT")
add_text(ax[NOAT], f"{{\\bf (b)}} {NOAT}")

ax[OAT].set_ylabel(size_label)
ax[NOAT].set_yticklabels([])
fig.colorbar(mesh, cax = cax, label = sqz_label)

plt.tight_layout()
plt.gca().set_rasterization_zorder(1)
plt.savefig(fig_dir + "optimization/sqz_opt_dec_L_2D.pdf",
            rasterized = True, dpi = dpi)
plt.close()
