#!/usr/bin/env python3

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

sqz_methods = [ "OAT", "TAT" ]

method_TAT = "exact"
depths_TAT = np.arange(20,81,2)/10
sizes_TAT = np.arange(10,101,5)
phi_cutoff = 10 # minimum value of (\phi/\pi)^{-1}

color_map = "jet"
dpi = 600

font = { "family" : "serif",
         "sans-serif" : "Computer Modern Sans serif" }
plt.rc("font",**font)
params = { "text.usetex" : True }
plt.rcParams.update(params)

depth_label = r"Lattice depth ($V_0/E_R$)"
size_label = r"Linear lattice size ($\ell$)"
sqz_label = r"Squeezing (dB)"
time_label = r"Time"
U_J_label = r"$U/J$"
phi_label = r"$\log_{10}(\phi/\pi)$"

def pd_read_1D(fname):
    return pd.read_csv(fname, comment = "#", squeeze = True, header = None, index_col = 0)
def pd_read_2D(fname):
    data = pd.read_csv(fname, comment = "#", header = 0, index_col = 0)
    data.columns = pd.to_numeric(data.columns)
    return data

def to_dB(vals): return 10*np.log10(vals)

J_0 = pd_read_1D(data_dir + "J_0.txt")
U_int = pd_read_2D(data_dir + "U_int_2D.txt")
all_depths = J_0.index
U_J = (U_int.T / J_0.T).T[confining_depth]
U_J_interp = interpolate.interp1d(U_J.values, all_depths)
U_J_ticks = np.array(range(int(U_J.values.max()))) + 1
depth_U_ticks = [ float(U_J_interp(val)) for val in U_J_ticks ]

def make_U_axis(axis):
    ax = axis.twiny()
    ax.set_xlim(all_depths[0],all_depths[-1])
    ax.set_xticks(depth_U_ticks)
    ax.set_xticklabels(U_J_ticks)
    ax.set_xlabel(U_J_label)
    return ax

# box object for OAT / TAT labels in 2-D plots
method_box = dict(boxstyle = "round", facecolor = "white", alpha = 1)


##########################################################################################
# benchmarking figures
##########################################################################################

figsize = (6.5,3.5)

# set up figure with eight panels
fig, axes = plt.subplots(figsize = figsize, nrows = 2, ncols = 4)
lines = [ "k-", "r:", "b--" ]
labels = [ "FH", "spin", "OAT" ]
line_order = [ 0, 2, 1 ]
legend_order = [ 0, 2, 1 ]
linewidth = 2

bench_dir = data_dir + "model_benchmarking/"

# load horizontal axis ranges
U_vals = np.loadtxt(bench_dir + "U_range.dat")
phi_vals = np.loadtxt(bench_dir + "phi_range.dat")
phi_vals = np.log10(1/phi_vals[phi_vals >= phi_cutoff])

# load and plot data
for N, col_N in [ (12, 0), (9, 2) ]:
    basename = bench_dir + f"L12N{N:02d}_"
    data_U_sqz = np.loadtxt(basename + "phi_pi-50_sq.dat")
    data_U_time = np.loadtxt(basename + "phi_pi-50_topt.dat")/(2*np.pi)
    data_phi_sqz = np.loadtxt(basename + "U_2_sq.dat")
    data_phi_time = np.loadtxt(basename + "U_2_topt.dat")/(2*np.pi)

    for x_vals, y_vals, row, col in [ (U_vals, data_U_sqz.T, 0, col_N),
                                      (U_vals, data_U_time.T, 1, col_N),
                                      (phi_vals, data_phi_sqz.T, 0, col_N+1),
                                      (phi_vals, data_phi_time.T, 1, col_N+1) ]:
        for jj in line_order:
            axes[row,col].plot(x_vals, y_vals[-len(x_vals):,jj], lines[jj],
                               label = labels[jj], linewidth = linewidth)

# set axis range and ticks
for row in range(2):
    for col in range(2):
        axes[row,2*col].set_xticks([0,4,8])
for col in range(4):
    axes[0,col].set_yticks([0,2,4,6])
    ymin = axes[0,col].get_ylim()[0]
    axes[0,col].set_ylim(ymin, 6 + abs(ymin))

# set all axis labels
axes[0,0].set_ylabel(sqz_label)
axes[1,0].set_ylabel(time_label + r" $(2\pi/J)$")

axes[1,0].set_xlabel("(i) " + U_J_label)
axes[1,1].set_xlabel("(ii) " + phi_label)
axes[1,2].set_xlabel("(i) " + U_J_label)
axes[1,3].set_xlabel("(ii) " + phi_label)

# clear unused axis labels
for col in range(4):
    axes[0,col].set_xticklabels([])
    if col > 0:
        for row in range(2):
            axes[row,col].set_yticklabels([])

# draw vertical line to split left and right panels
axes[0,0].vlines(1.1, -5, 1, linewidth = 1, color = "gray",
                 clip_on = False, transform = axes[0,1].transAxes)

# add labels for left and right panels
fig.text(0.23, 0.015, r"{\large (a) $f=1$}", transform = fig.transFigure)
fig.text(0.68, 0.015, r"{\large (b) $f=3/4$}", transform = fig.transFigure)

# make legend
handles, labels = axes[0,0].get_legend_handles_labels()
handles = [ handles[jj] for jj in legend_order ]
labels = [ labels[jj] for jj in legend_order ]
axes[0,0].legend(handles, labels, loc = "best")

plt.tight_layout(rect = (0,0.03,1,1))
plt.savefig(fig_dir + "model_benchmarking.pdf")


##########################################################################################
# squeezing without decoherence
##########################################################################################

figsize = (6.5,3)

# set up figure panel and linestyles
ax = {}
fig, ( ax_sqz, ax["OAT"], ax["TAT"], cax ) \
    = plt.subplots(figsize = figsize, ncols = 4,
                   gridspec_kw = { "width_ratios" : [1,2,2,0.1] })
lines = { "OAT" : "k-",
          "TAT" : "k--" }

# exctract data to plot
sqz_vals = {}
t_opt = {}
for method in sqz_methods:
    sqz_opt = -to_dB(pd_read_1D(data_dir + f"sqz_{method}.txt"))
    t_opt[method] = pd_read_2D(data_dir + f"optimization/t_opt_{method}_L_2D.txt")
    t_opt[method] /= recoil_energy_NU
    depths, sizes = t_opt[method].index, t_opt[method].columns

    sqz_interp = interpolate.interp1d(sqz_opt.index, sqz_opt.values)
    sqz_vals[method] = np.array([ float(sqz_interp(L**2)) for L in sizes ])

sqz_min = min([ sqz_vals[method].min() for method in sqz_methods ])
sqz_max = max([ sqz_vals[method].max() for method in sqz_methods ])
t_min = min([ t_opt[method].values.min() for method in sqz_methods ])
t_max = max([ t_opt[method].values.max() for method in sqz_methods ])

# plot data
for method in sqz_methods:
    ax_sqz.plot(sqz_vals[method], sizes, lines[method], label = method)

    mesh = ax[method].pcolormesh(depths, sizes, t_opt[method].T,
                                 cmap = plt.get_cmap(color_map),
                                 vmin = t_min, vmax = t_max,
                                 zorder = 0, rasterized = True)

    ax[method].set_xlabel(depth_label, zorder = 1)
    ax[method].set_yticklabels([])
    make_U_axis(ax[method])

    ax[method].text(0.9, 0.1, method, transform = ax[method].transAxes, bbox = method_box,
                    verticalalignment = "bottom", horizontalalignment = "right")

ax_sqz.legend(loc = "best").get_frame().set_alpha(1)
fig.colorbar(mesh, cax = cax, label = time_label + " (sec)", format = "%.1f")

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

def get_sqz_floor(method, lattice_depth, lattice_size):

    file_name = data_dir + f"TAT/{method}_{lattice_depth}_{lattice_size}.txt"
    spin_num = int(lattice_size)**2

    correlators = {}
    if method == "jump":
        data = np.loadtxt(file_name, dtype = complex)
        for op_idx in range(len(squeezing_ops)):
            op = squeezing_ops[op_idx]
            correlators[op] = data[op_idx,:]

        sqz = squeezing_from_correlators(spin_num, correlators, in_dB = True)
        plt.plot(sqz, "k.")
        return sqz.max()

    if method == "exact":

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
        for order_idx in range(orders.size):
            order = orders[order_idx]
            for op_idx in range(len(squeezing_ops)):
                op = squeezing_ops[op_idx]
                correlators[op] = derivs[op_idx,:order] @ times_k[:order,:]

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
sqz["TAT"] = np.zeros((depths_TAT.size,sizes_TAT.size))
for idx, _ in np.ndenumerate(sqz["TAT"]):
    sqz["TAT"][idx] = get_sqz_floor(method_TAT, depths_TAT[idx[0]], sizes_TAT[idx[1]])

depths = { "TAT" : depths_TAT }
sizes = { "TAT" : sizes_TAT }

sqz["OAT"] = -to_dB(pd_read_2D(data_dir + f"optimization/sqz_opt_OAT_dec_L_2D.txt"))
depths["OAT"], sizes["OAT"], sqz["OAT"] \
    = sqz["OAT"].index, sqz["OAT"].columns, sqz["OAT"].values

sqz_min = min([ sqz[method].min() for method in sqz_methods ])
sqz_max = max([ sqz[method].max() for method in sqz_methods ])

# set up figure panels
ax = {}
fig, ( ax["OAT"], ax["TAT"], cax ) \
    = plt.subplots(figsize = figsize, ncols = 3,
                   gridspec_kw = { "width_ratios" : [1,1,0.05] })

# plot data
for method in sqz_methods:
    mesh = ax[method].pcolormesh(depths[method], sizes[method], sqz[method].T,
                                 cmap = plt.get_cmap(color_map),
                                 vmin = sqz_min, vmax = sqz_max,
                                 zorder = 0, rasterized = True)
    ax[method].set_xlabel(depth_label, zorder = 1)
    make_U_axis(ax[method])

    ax[method].text(0.9, 0.1, method, transform = ax[method].transAxes, bbox = method_box,
                    verticalalignment = "bottom", horizontalalignment = "right")

ax["OAT"].set_ylabel(size_label)
ax["TAT"].set_yticklabels([])
fig.colorbar(mesh, cax = cax, label = sqz_label)

plt.tight_layout()
plt.gca().set_rasterization_zorder(1)
plt.savefig(fig_dir + "optimization/sqz_opt_dec_L_2D.pdf",
            rasterized = True, dpi = dpi)
plt.close()
