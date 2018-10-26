#!/usr/bin/env python3

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import interpolate
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter

from correlator_methods import squeezing_ops
from squeezing_methods import squeezing_from_correlators
from sr87_olc_constants import recoil_energy_NU


confining_depth = 60
data_dir = "../data/"
fig_dir = "../figures/"

method_TAT = "exact"
depths_TAT = np.arange(2,8.1,0.5)
sizes_TAT = np.arange(10,101,5)

depth_label = r"Lattice depth ($V_0/E_R$)"
size_label = r"Linear lattice size ($\ell$)"
sqz_label = r"$-10\log_{10}(\xi_{\mathrm{opt}}^2)$"
U_J_label = r"$U/J$"
time_label = r"$t_{\mathrm{opt}}$"

dpi = 600
params = { "text.usetex" : True }
plt.rcParams.update(params)

def pd_read_1D(fname):
    return pd.read_csv(fname, comment = "#", squeeze = True, header = None, index_col = 0)
def pd_read_2D(fname):
    data = pd.read_csv(fname, comment = "#", header = 0, index_col = 0)
    data.columns = pd.to_numeric(data.columns)
    return data

J_0 = pd_read_1D(data_dir + "J_0.txt")
U_int = pd_read_2D(data_dir + "U_int_2D.txt")
depths = J_0.index
U_J = (U_int.T / J_0.T).T[confining_depth]
U_J_interp = interpolate.interp1d(U_J.values, depths)
U_J_ticks = np.array(range(int(U_J.values.max()))) + 1
depth_U_ticks = [ float(U_J_interp(val)) for val in U_J_ticks ]

def make_U_axis(axis):
    ax = axis.twiny()
    ax.set_xlim(depths[0],depths[-1])
    ax.set_xticks(depth_U_ticks)
    ax.set_xticklabels(U_J_ticks)
    ax.set_xlabel(U_J_label)
    return ax


##########################################################################################
# squeezing without decoherence
##########################################################################################

figsize = (5,3)

for method in [ "OAT", "TAT" ]:
    sqz_opt = pd_read_1D(data_dir + f"sqz_{method}.txt")
    t_opt = pd_read_2D(data_dir + f"optimization/t_opt_{method}_L_2D.txt")
    t_opt /= recoil_energy_NU
    depths, sizes = t_opt.index, t_opt.columns

    # set up figure with two panels
    plt.figure(figsize = figsize)
    grid = gridspec.GridSpec(1, 2, width_ratios = [ 1, 3 ])
    ax_sqz = plt.subplot(grid[0])
    ax_time = plt.subplot(grid[1])
    ax_time.set_yticklabels([])

    # plot optimal squeezing data
    sqz_interp = interpolate.interp1d(sqz_opt.index, sqz_opt.values)
    sqz_vals = [ float(sqz_interp(L**2)) for L in sizes ]
    ax_sqz.plot(sqz_vals, sizes, "k")
    ax_sqz.set_ylim(sizes[0], sizes[-1])
    ax_sqz.set_xlim(sqz_vals[0], sqz_vals[-1])
    ax_sqz.set_xlabel(sqz_label, zorder = 1)
    ax_sqz.set_ylabel(size_label, zorder = 1)

    # plot optimal time data
    mesh = ax_time.pcolormesh(depths, sizes, t_opt.T,
                              cmap = plt.get_cmap("jet"),
                              zorder = 0, rasterized = True)
    plt.xlabel(depth_label, zorder = 1)
    plt.colorbar(mesh, label = time_label, format = "%.1f")

    make_U_axis(ax_time)
    plt.tight_layout()
    plt.gca().set_rasterization_zorder(1)
    plt.savefig(fig_dir + f"optimization/t_opt_{method}_L_2D.pdf",
                rasterized = True, dpi = dpi)
    plt.close()


##########################################################################################
# OAT squeezing with decoherence
##########################################################################################

def integer_ticks(vals):
    tick_min, tick_max = int(np.ceil(vals.min())), int(np.floor(vals.max()))
    return range(tick_min, tick_max+1)

def plot_sqz(depths, sizes, sqz_vals):
    figsize = (4,3)
    plt.figure(figsize = figsize)
    mesh = plt.pcolormesh(depths, sizes, sqz_vals.T,
                          cmap = plt.get_cmap("jet"),
                          zorder = 0, rasterized = True)
    plt.xlabel(depth_label, zorder = 1)
    plt.ylabel(size_label, zorder = 1)
    plt.colorbar(mesh, label = sqz_label, ticks = integer_ticks(sqz_vals))

    make_U_axis(plt.gca())
    plt.tight_layout()
    plt.gca().set_rasterization_zorder(1)

sqz_OAT = pd_read_2D(data_dir + f"optimization/sqz_opt_OAT_dec_L_2D.txt")
plot_sqz(sqz_OAT.index, sqz_OAT.columns, sqz_OAT.values)
plt.savefig(fig_dir + f"optimization/sqz_opt_OAT_dec_L_2D.pdf",
            rasterized = True, dpi = dpi)
plt.close()


##########################################################################################
# TAT squeezing with decoherence
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

        plt.plot(squeezing_from_correlators(spin_num, correlators), "k.")

        return squeezing_from_correlators(spin_num, correlators).max()

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

            sqz = squeezing_from_correlators(spin_num, correlators)

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

sqz_TAT = np.zeros((depths_TAT.size,sizes_TAT.size))
for idx, _ in np.ndenumerate(sqz_TAT):
    sqz_TAT[idx] = get_sqz_floor(method_TAT, depths_TAT[idx[0]], sizes_TAT[idx[1]])

plot_sqz(depths_TAT, sizes_TAT, sqz_TAT)
plt.savefig(fig_dir + f"optimization/sqz_opt_TAT_dec_L_2D.pdf",
            rasterized = True, dpi = dpi)
plt.close()
