#!/usr/bin/env python3

import os, sys, glob
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from multibody_methods import dist_method
from ising_squeezing import ising_squeezing_optimum, ising_minimal_SS

np.set_printoptions(linewidth = 200)

data_dir = "../data/"
fig_dir = "../figures/XXZ/"

params = { "font.size" : 8,
           "text.usetex" : True,
           "text.latex.preamble" : [ r"\usepackage{braket}",
                                     r"\usepackage{bm}" ]}
plt.rcParams.update(params)

max_width = 8.6/2.54 # maximum single-column figure width allowed by PRL

def make_name_tag(lattice_text, alpha, data_format):
    name_tag = f"L{lattice_text}_a{alpha}"
    if data_format == "shell":
        name_tag += "_M4"
    elif data_format != "dtwa":
        print(f"data file format unrecognized : {data_format}")
    return name_tag

def pop_label(manifold):
    return r"$\braket{\mathcal{P}_{" + str(manifold) + r"}}_{\mathrm{max}}$"

def to_dB(sqz):
    return 10*np.log10(np.array(sqz))

label_SS = r"$\braket{\bm S^2}_{\mathrm{min}} / \braket{\bm S^2}_0$"
label_sqz = r"$-10\log_{10}\xi^2$"
label_sqz_opt = r"$-10\log_{10}\xi_{\mathrm{opt}}^2$"
label_time = r"$t\times J_\perp$"
label_time_opt = r"$t_{\mathrm{opt}}\times J_\perp$"

##########################################################################################
# collect shell model data

lattice = (7,7)
alpha = 3
excess_pop = 0.1
zz_lims = (-1,2)
periodic = True

spin_num = np.prod(lattice)
shell_xticks = np.arange(zz_lims[0], zz_lims[1]+1).astype(int)

dist = dist_method(lattice, periodic)
sunc_mat = np.zeros((spin_num,spin_num))
for ii in range(spin_num):
    for jj in range(ii):
        sunc_mat[ii,jj] = sunc_mat[jj,ii] = 1/dist(ii,jj)**alpha

lattice_text = "x".join([ str(ll) for ll in lattice ])
name_tag = make_name_tag(lattice_text, alpha, "shell")
sweep_file = data_dir + f"shells/sweep_{name_tag}.txt"
data = np.loadtxt(sweep_file, dtype = complex)
with open(sweep_file, "r") as file:
    for line in file:
        if "# manifolds : " in line:
            manifolds = list(map(int,line.split()[3:]))
        if line[0] != "#": break

zz_coupling = data[:,0].real
time_opt = data[:,1].real
min_sqz = data[:,2].real
min_SS = data[:,3].real
correlators_opt = { "Z" : data[:,4],
                    "+" : data[:,5],
                    "ZZ" : data[:,6],
                    "++" : data[:,7],
                    "+Z" : data[:,8],
                    "+-" : data[:,9] }
min_pops_0 = data[:,10].real

max_pops = np.zeros(( len(zz_coupling), max(manifolds) ))
for manifold, pops in zip(manifolds[1:], data[:,11:].real.T):
    max_pops[:,manifold-1] = pops

trusted_zz = zz_coupling[max_pops[:,-1] <= excess_pop]
min_zz, max_zz = ( lambda pops : [ min(pops), max(pops) ] )( trusted_zz )
min_idx, max_idx = list(zz_coupling).index(min_zz), list(zz_coupling).index(max_zz)
min_zz = np.mean([ zz_coupling[min_idx], zz_coupling[min_idx-1] ])
max_zz = np.mean([ zz_coupling[max_idx], zz_coupling[max_idx+1] ])

def shade_exclusions(axis = None):
    if axis is None:
        axis = plt.gca()
    axis.set_xlim(*zz_lims)
    axis.axvspan(zz_coupling[0], min_zz, alpha = 0.5, color = "grey")
    axis.axvspan(max_zz, zz_coupling[-1], alpha = 0.5, color = "grey")

##################################################
# make TS_4 population plot

plt.figure(figsize = (3,1.8))
for idx, pops in enumerate(max_pops.T):
    plt.plot(zz_coupling, pops, ".", label = pop_label(idx+1))
shade_exclusions()
plt.xticks(shell_xticks)
plt.xlabel(r"$J_{\mathrm{z}}/J_\perp$")
plt.ylabel("population")
plt.legend(loc = "center", handlelength = 0.5, bbox_to_anchor = (0.63,0.67))
plt.tight_layout(pad = 0.2)
plt.savefig(fig_dir + f"populations_{name_tag}.pdf")

##################################################
# make DTWA benchmarking plot

max_SS = spin_num/2 * (spin_num/2+1)

figure, axes = plt.subplots(2, figsize = (3,3))

# plot shell model results
axes[0].plot(zz_coupling, -to_dB(min_sqz), "k.", label = "TS$_4$")
axes[1].plot(zz_coupling, min_SS / max_SS, "k.")

# plot DTWA results
name_tag_dtwa = make_name_tag(lattice_text, alpha, "dtwa")
zz_coupling, min_sqz_dB, min_SS_normed \
    = np.loadtxt(data_dir + f"DTWA/dtwa_{name_tag_dtwa}.txt", unpack = True)
axes[0].plot(zz_coupling, -min_sqz_dB, "r.", label = "DTWA")
axes[1].plot(zz_coupling, min_SS_normed, "r.")

# reference: collective and Ising limits
kwargs = { "color" : "k", "linestyle" : "--", "zorder" : 0 }
_, sqz_OAT = ising_squeezing_optimum(np.ones((spin_num,spin_num)), TI = True)
axes[0].axhline(-to_dB(sqz_OAT), **kwargs, label = "OAT")
axes[1].axhline(1, **kwargs)

kwargs["linestyle"] = ":"
_, sqz_ising = ising_squeezing_optimum(sunc_mat, TI = True)
_, min_SS_ising = ising_minimal_SS(sunc_mat, TI = True)
axes[0].axhline(-to_dB(sqz_ising), **kwargs, label = "Ising")
axes[1].axhline(min_SS_ising/max_SS, **kwargs)

# tweak axis limits
axes[0].set_ylim(bottom = np.floor(-to_dB(sqz_ising)))
axes[1].set_ylim(bottom = np.floor(min_SS_ising/max_SS * 10)/10)
axes[0].set_xticks(shell_xticks)
axes[1].set_xticks(shell_xticks)
axes[0].set_xticklabels([])

axes[0].set_ylabel(label_sqz_opt)
axes[1].set_ylabel(label_SS)
axes[1].set_xlabel(r"$J_{\mathrm{z}}/J_\perp$")
shade_exclusions(axes[0])
shade_exclusions(axes[1])
axes[0].legend(loc = "center", handlelength = 1.7, bbox_to_anchor = (0.6,0.45))
plt.tight_layout(pad = 0.5)
plt.savefig(fig_dir + f"benchmarking_{name_tag}.pdf")

##########################################################################################
# plot time-series DTWA results

lattice_text = "64x64"
color_map = "viridis"

color_map = mpl.cm.get_cmap(color_map)
for alpha, zz_lims, max_time, sqz_range in [ ( 3, [-3,-1], 10, [-20,5] ),
                                             ( 4, [-2, 0], 5, [-15,5] ) ]:

    # identify info related to data file names
    name_tag = make_name_tag(lattice_text, alpha, "dtwa")
    name_format = data_dir + "DTWA/time_series/dtwa_" + name_tag + "*.txt"
    def get_zz_coupling(file):
        return float(file.split("_")[-1][1:-4])
    def in_range(zz_coupling):
        return zz_lims[0] <= zz_coupling <= zz_lims[1]
    def get_time_sqz(file):
        all_data = np.loadtxt(file)
        time = all_data[:,0]
        sqz = all_data[:,4]
        return time, sqz

    # collect all data files and the corresponding ZZ couplings
    data_files = sorted([ ( zz_coupling, file )
                          for file in glob.glob(name_format)
                          if in_range(zz_coupling := get_zz_coupling(file)) ])

    # plot squeezing over time for all ZZ couplings
    plt.figure(figsize = (2.5,1.5))
    optimal_times = np.zeros(len(data_files)) # keep track of optimal squeezing times
    for idx, ( zz_coupling, file ) in enumerate(data_files):
        color_val = 1 - idx / ( len(data_files)-1 ) # color value from 0 to 1
        time, sqz = get_time_sqz(file)
        time_scale = abs( zz_coupling - 1 ) # rescale time by | J_z - J_perp |
        plt.plot(time * time_scale, sqz, color = color_map(color_val))

        # keep track of the optimal squeezing time
        optimal_times[idx] = time[sqz.argmin()]

    # highlight squeezing over time at the collective/Ising crossover
    optimal_time_ratios = optimal_times[1:] / optimal_times[:-1]
    zz_coupling, file = data_files[optimal_time_ratios.argmax()+1]
    time, sqz = get_time_sqz(file)
    time_scale = abs( zz_coupling - 1 )
    plt.plot(time * time_scale, sqz, color = "red", linewidth = "2")

    # fix time and squeeizng axis ticks
    time_locator = mpl.ticker.MaxNLocator(5, integer = True)
    plt.gca().xaxis.set_major_locator(time_locator)

    sqz_locator = mpl.ticker.MaxNLocator(5, integer = True)
    plt.gca().yaxis.set_major_locator(sqz_locator)

    # set horizonal and vertical axis ranges
    plt.xlim(0, max_time)
    plt.ylim(sqz_range)

    # touch-up and save figure
    plt.xlabel(r"$t\times \left|J_{\mathrm{z}}-J_\perp\right|$")
    plt.ylabel(label_sqz)
    plt.tight_layout(pad = 0.3)
    plt.savefig(fig_dir + f"crossover_{name_tag}.pdf")

##########################################################################################
# plot DTWA result summary

# plot DTWA data for a given lattice on a given list of axes
def plot_dtwa_data(axes, lattice_text, alpha_text, zz_lims, alpha_lims, add_reflines):
    dim = lattice_text.count("x") + 1 # spatial dimensions

    # import all relevant data
    raw_data = {}
    name_tags = [ make_name_tag(lattice_text, alpha_text, "dtwa"),
                  make_name_tag(lattice_text, "nn", "dtwa") ]
    for name_tag in name_tags:
        data_files = data_dir + f"DTWA/dtwa_{name_tag}.txt"
        for file in glob.glob(data_files):
            alpha = file.split("_")[-1][1:-4]
            if alpha == "nn" or alpha_lims[0] <= float(alpha) <= alpha_lims[1]:
                raw_data[alpha] = np.loadtxt(file)

    # collect data for plotting
    alpha_vals = sorted(list(raw_data.keys()))
    zz_couplings = next(iter(raw_data.values()))[:,0]
    data = {}
    keep = np.where( ( np.around(zz_couplings, decimals = 1) >= zz_lims[0] ) &
                     ( np.around(zz_couplings, decimals = 1) <= zz_lims[1] ) )[0]
    data["min_sqz"] = np.array([ raw_data[alpha][keep,1] for alpha in alpha_vals ])
    data["opt_tim"] = np.array([ raw_data[alpha][keep,2] for alpha in alpha_vals ])
    data["sqr_len"] = np.array([ raw_data[alpha][keep,3] for alpha in alpha_vals ])
    del raw_data
    zz_couplings = zz_couplings[keep]

    # collect numeric values of alpha
    numeric_alpha = [ float(alpha) for alpha in alpha_vals if alpha != "nn" ]

    # insert "empty" data for zz_coupling = 1
    zz_couplings = np.insert(zz_couplings, np.where(zz_couplings > 1)[0][0], 1)
    critical_coupling_idx = np.where(zz_couplings >= 1)[0][0]
    for key in data.keys():
        data[key] = np.insert(data[key], critical_coupling_idx, None, axis = 1)

    # make copies of the data at \alpha = \infty
    alpha_range = int( max(numeric_alpha) - min(numeric_alpha) )
    nn_bins = (len(alpha_vals)-1) // alpha_range // 2 + 1 # number of "nearest neighbor bins"
    alpha_vals = alpha_vals[:-1] + [ "nn" ] * nn_bins
    for key in data.keys():
        data[key] = np.vstack([ data[key][:-1,:] ] + [ data[key][-1,:] ] * nn_bins )

    # plot data
    plot_args = { "aspect" : "auto", "origin" : "lower",
                  "interpolation" : "nearest", "cmap" : "inferno" }
    images = [ axes[0].imshow(-data["min_sqz"], **plot_args),
               axes[1].imshow(+data["sqr_len"], **plot_args),
               axes[2].imshow(+data["opt_tim"], **plot_args, norm = mpl.colors.LogNorm()) ]

    # separate finite values of \alpha from \alpha = \infty
    for axis in axes:
        axis.axhline(len(alpha_vals)-nn_bins-0.5, color = "w", linewidth = 1, zorder = 2)

    if add_reflines:
        # identify boundaries between collective and Ising-dominated squeezing behavior
        # on the left, start at values of alpha (by index) where t_opt has the biggest jump
        boundary_lft_alpha_start \
            = ( data["opt_tim"][:-1,0] / data["opt_tim"][1:,0] ).argmax()
        # for each alpha >= [value above], find the ZZ coupling that minimizes S^2
        boundary_lft_alpha = list(range(boundary_lft_alpha_start, len(alpha_vals)))
        boundary_lft_coupling \
            = [ data["sqr_len"][idx,:critical_coupling_idx-1].argmin() - 0.5
                for idx in boundary_lft_alpha ]

        # on the right, consider values of the ZZ coupling J_z/J_\perp > 1
        boundary_rht_coupling = list(range(critical_coupling_idx+1, len(zz_couplings)))
        # find the first value of \alpha that minimizes S^2(\alpha)
        def locate_minimum(sqr_len):
            # to avoid "accidental" minima at S^2 ~ 1,
            # make sure that S^2 falls to at least this value
            max_dip = data["sqr_len"][0,-1]
            return np.where( (sqr_len[:-1] <= max_dip) &
                             (sqr_len[:-1] < sqr_len[1:]) )[0][0]
        boundary_rht_alpha = [ locate_minimum(data["sqr_len"][:,idx])
                               for idx in boundary_rht_coupling ]

        for axis in axes:
            bounday_args = { "color" : "gray",
                             "linewidth" : 1,
                             "linestyle" : ":",
                             "zorder" : 1 }
            axis.plot(boundary_lft_coupling, boundary_lft_alpha, **bounday_args)
            axis.plot(boundary_rht_coupling, boundary_rht_alpha, **bounday_args)

            # mark \alpha = D
            ref_args = { "color" : "gray",
                         "linewidth" : 1,
                         "linestyle" : "--",
                         "zorder" : 1 }
            axis.axhline(numeric_alpha.index(dim), **ref_args)

        # mark the cut for time-series data
        if dim == 2:
            cut_alpha = numeric_alpha.index(3)
            cut_couplings = [ list(zz_couplings).index(-3), list(zz_couplings).index(-1) ]
            axes[2].plot(cut_couplings, [cut_alpha]*2, "c-", linewidth = 1)

        # mark parameters for neutral atoms
        if dim in [ 2, 3 ]:
            alpha = len(alpha_vals) - nn_bins / 2
            axes[0].plot([ 0, len(zz_couplings)-1 ], [alpha]*2, "y-", linewidth = 1)

        # mark parameters for polar molecules
        if dim == 2:
            alpha = numeric_alpha.index(3)
            axes[0].plot([ 0, len(zz_couplings)-1 ], [alpha]*2, "g-", markersize = 1.5)

        # mark parameters for ions
        if dim == 2:
            alpha = [ numeric_alpha.index(1), numeric_alpha.index(3) ]
            couplings = [ list(zz_couplings).index(0) ] * 2
            axes[0].plot(couplings, alpha, "b-", linewidth = 1)

        # mark parameters for Rydberg atoms
        if dim in [ 2, 3 ]:
            alpha = numeric_alpha.index(6)
            for zz in [ 0, -3 ]:
                zz_idx = list(zz_couplings).index(zz)
                axes[0].plot([zz_idx], [alpha], "ro", markersize = 1.5)

        # mark parameters for magnetic atoms
        if dim == 2:
            zz_idx = list(zz_couplings).index(-2)
            alpha = numeric_alpha.index(3)
            axes[0].plot([zz_idx], [alpha], marker = "s",
                         color = "tab:pink", markersize = 1.5)

    # set axis ticks
    alpha_ticks = [ idx for idx, alpha in enumerate(numeric_alpha)
                    if float(alpha) == int(float(alpha)) ] + [ len(alpha_vals)-1 ]
    for axis in axes:
        axis.set_yticks(alpha_ticks)
        axis.set_xticks(np.where(np.isclose(zz_couplings % 1, 0))[0])
        axis.set_xticklabels([])
        axis.set_yticklabels([])

    return images

# set axis ticks on a linear-scale color bar
def fix_ticks(color_bar, base):
    min_val, _ = color_bar.ax.xaxis.get_data_interval()
    locator = mpl.ticker.IndexLocator(base = base, offset = -min_val)
    color_bar.set_ticks(locator)

# set axis ticks on a log-scale color bar
def fix_log_ticks(color_bar):
    if color_bar.orientation == "horizontal":
        axis = color_bar.ax.xaxis
    else:
        axis = color_bar.ax.yaxis

    base, subs = 10, np.arange(0,1.1,0.1)
    major_locator = mpl.ticker.LogLocator(base = base)
    minor_locator = mpl.ticker.LogLocator(base = base, subs = subs)

    min_val, max_val = axis.get_data_interval()
    def filter(values):
        return [ val for val in values if min_val <= val <= max_val ]
    major_tick_values = filter(major_locator.tick_values(min_val, max_val))
    minor_tick_values = filter(minor_locator.tick_values(min_val, max_val))

    color_bar.set_ticks(major_tick_values)
    axis.set_ticks(minor_tick_values, minor = True)
    color_bar.update_ticks()

# make a plot of DTWA data comparing multiple lattice sizes
def make_dtwa_plots(lattice_list, alpha_text = "*",
                    zz_lims = (-3,3), alpha_lims = (1,6),
                    add_reflines = True, label_panels = True, figsize = None):
    if type(lattice_list) is str:
        lattice_list = [ lattice_list ]
    cols = len(lattice_list)

    if figsize == None:
        if cols == 1:
            figsize = (3, 4)
        elif cols == 2:
            figsize = (max_width, 3.5)
        else:
            figsize = (7, 5)

    widths = [35]*cols + [1]
    figure, all_axes = plt.subplots(3, cols+1, figsize = figsize,
                                    gridspec_kw = { "width_ratios": widths })
    all_axes.shape = (3,-1)
    axes = all_axes[:,:-1]
    bar_axes = all_axes[:,-1]

    images = np.empty(axes.shape, dtype = mpl.image.AxesImage)
    for col, ( lattice_text, lattice_axes) in enumerate(zip(lattice_list, axes.T)):
        images[:,col] \
            = plot_dtwa_data(lattice_axes, lattice_text, alpha_text,
                             zz_lims, alpha_lims, add_reflines)

        # set titles for each column
        if cols > 1:
            dim = lattice_text.count("x") + 1
            lattice_axes[0].set_title(f"$D={dim}$")

    # set horizontal tick labels
    for axis in axes[-1,:]:
        axis.set_xlabel(r"$J_{\mathrm{z}}/J_\perp$")
        xtick_min = int(np.ceil(zz_lims[0]))
        xtick_max = int(np.floor(zz_lims[1]))
        xticklabels = range(xtick_min,xtick_max+1)
        if cols > 1:
            xticklabels = [ label if label % 2 == 0 else ""
                            for label in xticklabels ]
        axis.set_xticklabels(xticklabels)

    # set vertical tick labels
    for idx, axis in enumerate(axes[:,0]):
        axis.set_ylabel(r"$\alpha$")
        ytick_num = len(axis.get_yticks())
        axis.set_yticklabels(list(range(1,ytick_num)) + [ r"$\infty$" ])
        for label in axis.get_yticklabels():
            label.set_verticalalignment("center")

    labels = [ label_sqz_opt, label_SS, label_time_opt ]
    for row_images, bar_axis, label in zip(images, bar_axes, labels):
        clim_min = min( image.get_clim()[0] for image in row_images )
        clim_max = max( image.get_clim()[1] for image in row_images )
        clim = (clim_min, clim_max)
        for image in row_images:
            image.set_clim(clim)

        bar = figure.colorbar(row_images[0], cax = bar_axis, label = label)

        if label == label_sqz_opt:
            fix_ticks(bar, 5)
        if label == label_SS:
            fix_ticks(bar, 0.2)
        if label == label_time_opt:
            fix_log_ticks(bar)

    plt.tight_layout(pad = 0.3)
    plt.subplots_adjust(wspace = 0.1, hspace = 0.1)
    return figure

lattice_text = "64x64"
make_dtwa_plots(lattice_text)
plt.savefig(fig_dir + f"dtwa_L{lattice_text}.pdf")

# lattice_list = [ "64x64", "16x16x16" ]
lattice_list = [ "64x64", "64x64" ]
make_dtwa_plots(lattice_list)
lattice_label = "_L".join(lattice_list)
plt.savefig(fig_dir + f"dtwa_L{lattice_label}.pdf")

lattice_list = [ "4096", "64x64", "16x16x16" ]
alpha_text = "?.0"
make_dtwa_plots(lattice_list, alpha_text, add_reflines = False)
lattice_label = "_L".join(lattice_list)
plt.savefig(fig_dir + f"dtwa_L{lattice_label}_int.pdf")
