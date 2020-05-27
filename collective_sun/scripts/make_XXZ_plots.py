#!/usr/bin/env python3

import os, sys, glob
import numpy as np
import scipy.optimize

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

color_map = "viridis_r"
color_map = mpl.cm.get_cmap(color_map)

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

label_SS = r"$\braket{\bm S^2} / \braket{\bm S^2}_0$"
label_SS_min = r"$\braket{\bm S^2}_{\mathrm{min}} / \braket{\bm S^2}_0$"

label_sqz = r"$-10\log_{10}\xi^2$"
label_sqz_opt = r"$-10\log_{10}\xi_{\mathrm{opt}}^2$"

label_time = r"$t\times J_\perp$"
label_time_opt = r"$t_{\mathrm{opt}}\times J_\perp$"
label_time_rescaled = r"$t\times \left|J_{\mathrm{z}}-J_\perp\right|$"

label_zz = r"$J_{\mathrm{z}}/J_\perp$"

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
plt.xlabel(label_zz)
plt.ylabel("population")
plt.legend(loc = "center", handlelength = 0.5, bbox_to_anchor = (0.63,0.67))
plt.tight_layout(pad = 0.2)
plt.savefig(fig_dir + f"populations_{name_tag}.pdf")

plt.close("all")
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
    = np.loadtxt(data_dir + f"DTWA/optima/dtwa_{name_tag_dtwa}.txt", unpack = True)
axes[0].plot(zz_coupling, -min_sqz_dB, "r.", label = "DTWA")
axes[1].plot(zz_coupling, min_SS_normed, "r.")

# reference: collective and Ising limits
kwargs = { "color" : "k", "linestyle" : "--", "zorder" : 0 }
_, sqz_OAT = ising_squeezing_optimum(np.ones((spin_num,spin_num)), TI = periodic)
axes[0].axhline(-to_dB(sqz_OAT), **kwargs, label = "OAT")
axes[1].axhline(1, **kwargs)

kwargs["linestyle"] = ":"
_, sqz_ising = ising_squeezing_optimum(sunc_mat, TI = periodic)
_, min_SS_ising = ising_minimal_SS(sunc_mat, TI = periodic)
axes[0].axhline(-to_dB(sqz_ising), **kwargs, label = "Ising")
axes[1].axhline(min_SS_ising/max_SS, **kwargs)

# tweak axis limits
axes[0].set_ylim(bottom = np.floor(-to_dB(sqz_ising)))
axes[1].set_ylim(bottom = np.floor(min_SS_ising/max_SS * 10)/10)
axes[0].set_xticks(shell_xticks)
axes[1].set_xticks(shell_xticks)
axes[0].set_xticklabels([])

axes[0].set_ylabel(label_sqz_opt)
axes[1].set_ylabel(label_SS_min)
axes[1].set_xlabel(label_zz)
shade_exclusions(axes[0])
shade_exclusions(axes[1])
axes[0].legend(loc = "center", handlelength = 1.7, bbox_to_anchor = (0.6,0.45))
plt.tight_layout(pad = 0.5)
plt.savefig(fig_dir + f"benchmarking_{name_tag}.pdf")

plt.close("all")
##########################################################################################
# plot time-series data (DTWA)

lattice_text = "64x64"
def get_zz_coupling(file):
    return float(file.split("_")[-1][1:-4])

spin_num = np.prod(list(map(int,lattice_text.split("x"))))

for alpha, zz_lims, max_time, sqz_range in [ ( 3, [-3,-1], 10, [-5,20] ),
                                             ( 4, [-2, 0], 5, [-5,15] ) ]:

    # identify info related to data file names
    name_tag = make_name_tag(lattice_text, alpha, "dtwa")
    name_format = data_dir + "DTWA/time_series/dtwa_" + name_tag + "*.txt"
    def in_range(zz_coupling):
        return zz_lims[0] <= zz_coupling <= zz_lims[1]
    def get_time_sqz_SS(file):
        file_data = np.loadtxt(file)
        time = file_data[:,0]
        sqz = -file_data[:,4]
        norm_SS = ( file_data[:,11] * spin_num )**2 / ( spin_num/2 * (spin_num/2+1) )
        return time, sqz, norm_SS

    # collect all data files and the corresponding ZZ couplings
    data_files = sorted([ ( zz_coupling, file )
                          for file in glob.glob(name_format)
                          if in_range(zz_coupling := get_zz_coupling(file)) ])
    zz_couplings = list(zip(*data_files))[0]

    # keep track of minimal squared spin length
    min_norm_SS = np.zeros(len(data_files))

    # plot squeezing over time for all ZZ couplings
    figure, axes = plt.subplots(2, figsize = (2.5,2.5), sharex = True, sharey = False)
    for idx, ( zz_coupling, file ) in enumerate(data_files):
        color_val = idx / ( len(data_files)-1 ) # color value from 0 to 1
        time, sqz, norm_SS = get_time_sqz_SS(file)
        time_scale = abs( zz_coupling - 1 ) # rescale time by | J_z - J_perp |
        axes[0].plot(time * time_scale, sqz, color = color_map(color_val))
        axes[1].plot(time * time_scale, norm_SS, color = color_map(color_val))

        min_norm_SS[idx] = min(norm_SS[:sqz.argmax()+1])

    # highlight squeezing over time right *before* the collective-to-Ising transition
    zz_coupling, file = data_files[min_norm_SS.argmin()]
    time, sqz, norm_SS = get_time_sqz_SS(file)
    time_scale = abs( zz_coupling - 1 )
    axes[0].plot(time * time_scale, sqz, color = "red", linewidth = "2")
    axes[1].plot(time * time_scale, norm_SS, color = "red", linewidth = "2")

    # fix time and squeeizng axis ticks
    time_locator = mpl.ticker.MaxNLocator(5, integer = True)
    axes[0].xaxis.set_major_locator(time_locator)

    sqz_locator = mpl.ticker.MaxNLocator(5, integer = True)
    axes[0].yaxis.set_major_locator(sqz_locator)

    # set horizonal and vertical axis ranges
    axes[0].set_xlim(0, max_time)
    axes[0].set_ylim(sqz_range)
    axes[1].set_ylim(0,1)

    # touch-up and save figure
    axes[0].set_ylabel(label_sqz)
    axes[1].set_ylabel(label_SS)
    axes[1].set_xlabel(label_time_rescaled)
    plt.tight_layout(pad = 0.7)
    plt.savefig(fig_dir + f"time_series_{name_tag}.pdf")

plt.close("all")
##########################################################################################
# plot summary of DTWA results

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

# plot DTWA data for a given lattice on a given set of axes
def plot_dtwa_data(fin_axes, inf_axes, lattice_text, alpha_text,
                   zz_lims, alpha_lims, add_markup):
    dim = lattice_text.count("x") + 1 # spatial dimensions

    # import all relevant data
    raw_fin_data = {}
    name_tags = [ make_name_tag(lattice_text, alpha_text, "dtwa"),
                  make_name_tag(lattice_text, "nn", "dtwa") ]
    for name_tag in name_tags:
        data_files = data_dir + f"DTWA/optima/dtwa_{name_tag}.txt"
        for file in glob.glob(data_files):
            alpha = file.split("_")[-1][1:-4]
            if alpha == "nn" :
                raw_inf_data = np.loadtxt(file)
            elif alpha_lims[0] <= float(alpha) <= alpha_lims[1]:
                raw_fin_data[float(alpha)] = np.loadtxt(file)

    # identify ZZ couplings and values of \alpha
    alpha_vals = np.sort(list(raw_fin_data.keys()))
    zz_couplings = next(iter(raw_fin_data.values()))[:,0]

    # pick values of the ZZ coupling to keep
    keep = np.where( ( np.around(zz_couplings, decimals = 1) >= zz_lims[0] ) &
                     ( np.around(zz_couplings, decimals = 1) <= zz_lims[1] ) )[0]
    zz_couplings = zz_couplings[keep]

    # collect data for plotting
    fin_data, inf_data = {}, {}
    fin_data["min_sqz"] = np.array([ raw_fin_data[alpha][keep,1] for alpha in alpha_vals ])
    fin_data["opt_tim"] = np.array([ raw_fin_data[alpha][keep,2] for alpha in alpha_vals ])
    fin_data["sqr_len"] = np.array([ raw_fin_data[alpha][keep,3] for alpha in alpha_vals ])
    inf_data["min_sqz"] = raw_inf_data[keep,1]
    inf_data["opt_tim"] = raw_inf_data[keep,2]
    inf_data["sqr_len"] = raw_inf_data[keep,3]
    del raw_fin_data, raw_inf_data

    # insert "empty" data for zz_coupling = 1
    zz_couplings = np.insert(zz_couplings, np.where(zz_couplings > 1)[0][0], 1)
    critical_coupling_idx = np.where(zz_couplings >= 1)[0][0]
    for key in fin_data.keys():
        fin_data[key] = np.insert(fin_data[key], critical_coupling_idx, None, axis = 1)
        inf_data[key] = np.insert(inf_data[key], critical_coupling_idx, None)

    # plot data
    dz = zz_couplings[1] - zz_couplings[0]
    da = alpha_vals[1] - alpha_vals[0]
    axis_lims = [ zz_couplings[0] - dz/2, zz_couplings[-1] + dz/2,
                  alpha_vals[0] - da/2, alpha_vals[-1] + da/2 ]
    plot_args = dict( aspect = "auto", origin = "lower",
                      interpolation = "nearest", cmap = "inferno",
                      extent = axis_lims )
    fin_axes[0].imshow(-fin_data["min_sqz"], **plot_args)
    fin_axes[1].imshow(+fin_data["sqr_len"], **plot_args)
    fin_axes[2].imshow(+fin_data["opt_tim"], **plot_args, norm = mpl.colors.LogNorm())

    plot_args["extent"] = axis_lims[:2] + [-1,1]
    inf_axes[0].imshow([-inf_data["min_sqz"]], **plot_args)
    inf_axes[1].imshow([+inf_data["sqr_len"]], **plot_args)
    inf_axes[2].imshow([+inf_data["opt_tim"]], **plot_args, norm = mpl.colors.LogNorm())

    if add_markup:
        # mark \alpha = D
        ref_args = dict( zorder = 1, color = "gray",
                         linewidth = 1, linestyle = "--" )
        for axis in fin_axes:
            axis.axhline(dim, **ref_args)

        # identify boundaries between collective and Ising-dominated squeezing behavior
        # on the left, start at values of alpha where t_opt has the biggest jump
        alpha_start_idx = ( fin_data["opt_tim"][:-1,0] / fin_data["opt_tim"][1:,0] ).argmax()
        boundary_lft_alpha = alpha_vals[alpha_start_idx:]
        # for each alpha >= [value above], find the ZZ coupling that minimizes S^2
        region_sqr_len = fin_data["sqr_len"][alpha_start_idx:,:critical_coupling_idx-1]
        boundary_lft_coupling_idx = region_sqr_len.argmin(axis = 1)
        boundary_lft_coupling = zz_couplings[boundary_lft_coupling_idx]

        # on the right, consider values of the ZZ coupling J_z/J_\perp > 1
        boundary_rht_coupling = zz_couplings[critical_coupling_idx+1:]
        # find the value of \alpha that minimizes S^2(\alpha)
        def locate_minimum(sqr_len, threshold = 0.01):
            first_dip = np.where( (sqr_len[:-1] <= 1 - threshold) &
                                  (sqr_len[:-1] < sqr_len[1:]) )[0][0]
            peak = sqr_len[first_dip:].argmax() + first_dip
            dip = sqr_len[first_dip:peak].argmin() + first_dip
            return alpha_vals[dip]
        boundary_rht_alpha = [ locate_minimum(fin_data["sqr_len"][:,idx])
                               for idx in range(critical_coupling_idx+1, len(zz_couplings)) ]

        for axis in fin_axes:
            bounday_args = dict( zorder = 1, color = "gray",
                                 linewidth = 1, linestyle = ":" )
            axis.plot(boundary_lft_coupling, boundary_lft_alpha, **bounday_args)
            axis.plot(boundary_rht_coupling, boundary_rht_alpha, **bounday_args)

        # plot left boundary at infinite \alpha
        lft_sqr_len = inf_data["sqr_len"][:critical_coupling_idx-1]
        boundary_coupling = zz_couplings[lft_sqr_len.argmin()]
        for axis in inf_axes:
            axis.plot([boundary_coupling]*2, axis.get_ylim(), **bounday_args)

        # label the Ising / collective phases
        text_args = dict( color = "black",
                          horizontalalignment = "center",
                          verticalalignment = "center" )
        fin_axes[0].text(-1, 1.25, "collective", **text_args)
        text_args["color"] = "white"
        fin_axes[0].text(2, 5, "Ising", **text_args)
        if dim == 2:
            fin_axes[0].text(-2, 5, "Ising", **text_args)
        if dim == 3:
            fin_axes[0].text(-2.35, 5.7, "Ising", **text_args)

        marker_args = dict( linewidth = 1, markersize = 1.5, clip_on = False )

        # mark parameters for neutral atoms
        if dim in [ 2, 3 ]:
            inf_axes[1].plot(zz_lims, [0,0], "y-", **marker_args)

        # mark parameters for polar molecules
        if dim == 2:
            fin_axes[1].plot(zz_lims, [3,3], "g-", **marker_args)

        # mark parameters for ions
        if dim == 2:
            fin_axes[1].plot([0,0], [ alpha_vals[0], 3 ], "b-", **marker_args)

        # mark parameters for Rydberg atoms
        if dim in [ 2, 3 ]:
            fin_axes[1].plot([0], [6], "ro", zorder = 4, **marker_args)
        if dim == 2:
            fin_axes[1].plot([0], [3], "ro", zorder = 4, **marker_args)
            fin_axes[1].plot([-0.73], [3], "ro", zorder = 4, **marker_args)

        # mark parameters for magnetic atoms
        if dim == 2:
            fin_axes[1].plot([-2], [3], marker = "s", color = "tab:pink", **marker_args)

    # set axis ticks at integer values
    zz_ticks = sorted(set(map(int,map(round, zz_couplings))))
    alpha_ticks = sorted(set(map(int,map(round, alpha_vals))))
    for axis in fin_axes:
        axis.set_xticks(zz_ticks)
        axis.set_yticks(alpha_ticks)
        axis.set_xticklabels([])
        axis.set_yticklabels([])
    for axis in inf_axes:
        axis.set_xticks([])
        axis.set_yticks([0])
        axis.set_yticklabels([])

# make a plot of DTWA data comparing multiple lattice sizes
def make_dtwa_plots(lattice_list, alpha_text = "*",
                    zz_lims = (-3,3), alpha_lims = (0.6,6),
                    add_markup = True, label_panels = True, figsize = None):
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

    # set up figure with "primary" axes for plots and color bars
    widths = [35]*cols + [1]
    figure, init_axes = plt.subplots(3, cols+1, figsize = figsize,
                                     gridspec_kw = dict( width_ratios = widths ))
    init_axes.shape = (3,-1)
    fin_axes = init_axes[:,:-1] # axes for plotting data with finite \alpha
    bar_axes = init_axes[:,-1] # axes for colorbars

    # make axes for plotting data at infinite \alpha
    inf_axes = np.empty(fin_axes.shape, dtype = fin_axes.dtype)
    for idx in np.ndindex(fin_axes.shape):
        divider = make_axes_locatable(fin_axes[idx])
        inf_axes[idx] = divider.append_axes("top", size = "8%", pad = 0.04)

    # plot all data
    for col, lattice_text in enumerate(lattice_list):
        plot_dtwa_data(fin_axes[:,col], inf_axes[:,col], lattice_text, alpha_text,
                       zz_lims, alpha_lims, add_markup)

        # set titles for each column
        if cols > 1:
            dim = lattice_text.count("x") + 1
            inf_axes[0,col].set_title(f"$D={dim}$", pad = 3)

    # set horizontal tick labels
    for axis in fin_axes[-1,:]:
        axis.set_xlabel(label_zz)
        labels = axis.get_xticks()
        if cols > 1:
            labels = [ label if label % 2 == 0 else "" for label in labels ]
        axis.set_xticklabels(labels)

    # set vertical tick labels
    for idx, axis in enumerate(fin_axes[:,0]):
        axis.set_ylabel(r"$\alpha$")
        axis.set_yticklabels(axis.get_yticks())
    for idx, axis in enumerate(inf_axes[:,0]):
        axis.set_yticklabels([r"$\infty$"], verticalalignment = "center")

    # set color bar limits and labels
    labels = [ label_sqz_opt, label_SS_min, label_time_opt ]
    for row, ( bar_axis, label ) in enumerate(zip(bar_axes, labels)):
        row_axes = list(fin_axes[row,:]) + list(inf_axes[row,:])
        row_images = [ axis.get_images()[0] for axis in row_axes ]
        clim_min = min( image.get_clim()[0] for image in row_images )
        clim_max = max( image.get_clim()[1] for image in row_images )
        clim = (clim_min, clim_max)
        for image in row_images:
            image.set_clim(clim)

        bar = figure.colorbar(row_images[0], cax = bar_axis, label = label)

        if label == label_sqz_opt:
            fix_ticks(bar, 5)
        if label == label_SS_min:
            fix_ticks(bar, 0.2)
        if label == label_time_opt:
            fix_log_ticks(bar)

    # "tighten" the plot layout, trimming empty space
    plt.tight_layout(pad = 0.3)
    plt.subplots_adjust(wspace = 0.1, hspace = 0.1)
    return figure

lattice_text = "64x64"
make_dtwa_plots(lattice_text, add_markup = False)
plt.savefig(fig_dir + f"dtwa_L{lattice_text}.pdf")

lattice_list = [ "4096", "64x64", "16x16x16" ]
alpha_text = "?.0"
make_dtwa_plots(lattice_list, alpha_text, add_markup = False)
lattice_label = "_L".join(lattice_list)
plt.savefig(fig_dir + f"dtwa_L{lattice_label}_int.pdf")

lattice_list = [ "64x64", "16x16x16" ]
make_dtwa_plots(lattice_list)
lattice_label = "_L".join(lattice_list)
plt.savefig(fig_dir + f"dtwa_L{lattice_label}.pdf")

plt.close("all")
##########################################################################################
# plot system size scaling (DTWA)

def log_form(x, a, b):
    return a * np.log(x) + b
def log_fit(x, fit_params):
    return np.vectorize(log_form)(x, *fit_params)

dim = 2
dz = 0.1
inspect_dz = 0.5

size_lims = (10,60)
dL = 5
lattice_lengths = np.arange(size_lims[0], size_lims[1] + dL/2, dL, dtype = int)


figsize = (3,1.8)

for alpha, zz_lims in [ ( 3, [-2.5,2.2] ), ( "nn", [-1.5,2.2] ) ]:
    zz_couplings = np.arange(zz_lims[0], zz_lims[1] + dz/2, dz)

    sqz_data = np.empty((len(zz_couplings), len(lattice_lengths)), None)
    ss_min_data = np.empty((len(zz_couplings), len(lattice_lengths)), None)
    for zz_idx, zz_coupling in enumerate(zz_couplings):
        # skip the critical point at J_z = 1
        if np.allclose(zz_coupling,1):
            sqz_data[zz_idx,:] = None
            continue

        zz_text = f"z{zz_coupling:.1f}"
        for ll_idx, lattice_length in enumerate(lattice_lengths):
            lattice_text = "x".join([str(lattice_length)]*dim)

            name_tag = make_name_tag(lattice_text, alpha, "dtwa")
            name_format = data_dir + "DTWA/system_size/dtwa_" + name_tag + f"_{zz_text}.txt"
            candidate_files = glob.glob(name_format)

            assert(len(candidate_files) == 1)
            file = candidate_files[0]
            file_data = np.loadtxt(file)
            sqz_min_idx = file_data[:,4].argmin()
            sqz_data[zz_idx,ll_idx] = -file_data[sqz_min_idx,4]
            ss_min_data[zz_idx,ll_idx] = min(file_data[:sqz_min_idx+1,11])
            del file_data

    # identify the dynamical phase boundary
    zz_boundaries = zz_couplings[ ss_min_data[zz_couplings < 1, :].argmin(axis = 0) ]

    # plot squeezing data
    figure = plt.figure(figsize = figsize)
    axis_lims = [ zz_lims[0] - dz/2, zz_lims[1] + dz/2,
                  size_lims[0] - dL/2, size_lims[1] + dL/2, ]
    plot_args = dict( aspect = "auto", origin = "lower",
                      interpolation = "nearest", cmap = "inferno",
                      extent = axis_lims )
    image = plt.imshow(sqz_data.T, **plot_args)

    # add color bar
    color_bar = figure.colorbar(image, label = label_sqz_opt)
    fix_ticks(color_bar, 5)

    # add reference line for dynamical phase boundary
    if alpha != "nn":
        def alternate(values, other_values):
            return np.array(list(zip(values, other_values))).flatten()
        def double(values):
            return alternate(values,values)
        LL_offsets = dL/2 * alternate(- np.ones(len(lattice_lengths)),
                                      + np.ones(len(lattice_lengths)))
        plt.plot(double(zz_boundaries), double(lattice_lengths) + LL_offsets,
                 ":", color = "gray", linewidth = 1)

    # label axes
    plt.xlabel(label_zz)
    plt.ylabel(r"$L$")

    # set axis ticks
    zz_ticks = sorted(set([ int(zz) for zz in zz_couplings ]))
    zz_labels = [ zz if zz % 2 == 0 else "" for zz in zz_ticks ]
    LL_labels = [ LL if LL % 10 == 0 else "" for LL in lattice_lengths ]
    plt.xticks(zz_ticks, zz_labels)
    plt.yticks(lattice_lengths, LL_labels)

    # label each dynamical phase
    if alpha == 3:
        text_args = dict( horizontalalignment = "center",
                          verticalalignment = "center" )
        plt.text(0, 30, "collective", color = "black", **text_args)
        plt.text(-2, 20, "Ising", color = "white", **text_args)
        plt.text(1.8, 30, "Ising", color = "white", **text_args)

    plt.tight_layout(pad = 0.1)
    plt.savefig(fig_dir + f"size_scaling_a{alpha}.pdf")

    if alpha == "nn": continue

    ### show logarithmic divergence of critical coupling

    # get system sizes
    system_sizes = lattice_lengths**2

    # find the best log fit of critical coupling as a function of system size
    fit_params, _ = scipy.optimize.curve_fit(log_form, system_sizes, zz_boundaries)

    # plot DTWA results and the log fit
    plt.figure(figsize = figsize)
    plt.semilogx(system_sizes, zz_boundaries, "ko", label = "DTWA")
    plt.semilogx(system_sizes, log_fit(system_sizes, fit_params), "k--", label = "log fit")

    plt.xlabel(r"$N$")
    plt.ylabel(r"$J_{\mathrm{z}}^{\mathrm{crit}}/J_\perp$")
    plt.legend(loc = "best", handlelength = 1.7)
    fit_handles, fit_labels = plt.gca().get_legend_handles_labels()

    plt.tight_layout()
    plt.savefig(fig_dir + f"size_divergence_a{alpha}.pdf")

    ### show power-law scaling of optimal squeezing

    plt.figure(figsize = figsize)

    # determine the values of zz couplings to plot
    inspect_zz_start = min(zz_boundaries)
    while np.sum(zz_boundaries < inspect_zz_start) < 4:
        inspect_zz_start += inspect_dz
    inspect_zz_min = np.ceil( inspect_zz_start / inspect_dz ) * inspect_dz
    inspect_zz = np.arange(inspect_zz_min, 1-inspect_dz/2, inspect_dz)

    for idx, zz_coupling in enumerate(inspect_zz):
        zz_idx = np.isclose(zz_couplings,zz_coupling)
        ll_idx = zz_boundaries < zz_coupling

        _system_sizes = system_sizes[ll_idx]
        sqz_vals = sqz_data[zz_idx, ll_idx]
        fit_params, _ = scipy.optimize.curve_fit(log_form, _system_sizes, sqz_vals)

        color_val = idx / ( len(inspect_zz) - 1 )
        color = color_map(color_val)
        plt.semilogx(_system_sizes, sqz_vals, "o", color = color)
        plt.semilogx(_system_sizes, log_fit(_system_sizes, fit_params), "--", color = color)

    plt.xlabel(r"$N$")
    plt.ylabel(label_sqz_opt)
    plt.legend(fit_handles, fit_labels, loc = "best", handlelength = 1.7)

    plt.tight_layout()
    plt.savefig(fig_dir + f"power_law_a{alpha}.pdf")

plt.close("all")
##########################################################################################
# plot dependence on filling fraction (DTWA)

figsize = (3,2)

data = np.loadtxt(data_dir + "DTWA/filling_fraction.txt")
markers = [ "o", "s", "^" ]
labels = [ f"${zz}$" for zz in [ 0.9, 0, -1 ] ]

plt.figure(figsize = figsize)
for zz_idx, marker, label in zip(reversed(range(1,4)), markers, labels):
    plt.plot(data[:,0], -data[:,zz_idx], marker, label = label)

plt.gca().set_xlim(-0.05, 1.05)
plt.gca().set_ylim(bottom = 0)

plt.xlabel("$f$")
plt.ylabel(label_sqz_opt)

plt.legend(loc = "best")
plt.tight_layout()
plt.savefig(fig_dir + "filling_fraction_a3.pdf")

plt.close("all")
