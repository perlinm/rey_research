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

params = { "font.size" : 9,
           "text.usetex" : True,
           "text.latex.preamble" : [ r"\usepackage{braket}",
                                     r"\usepackage{bm}" ]}
plt.rcParams.update(params)

max_width = 8.6/2.54 # maximum single-column figure width allowed by PRL

color_map = "viridis_r"
color_map = mpl.cm.get_cmap(color_map)

### name tags and axis labels

def make_name_tag(lattice_text, alpha, data_format):
    name_tag = f"L{lattice_text}_a{alpha}"
    if data_format == "shell":
        name_tag += "_M4"
    elif data_format != "dtwa":
        print(f"data file format unrecognized : {data_format}")
    return name_tag

def pop_label(manifold):
    return r"$\braket{\mathcal{P}_{" + str(manifold) + r"}}_{\mathrm{max}}$"

label_SS = r"$\braket{\bm S^2} / \braket{\bm S^2}_0$"
label_SS_min = r"$\braket{\bm S^2}_{\mathrm{min}} / \braket{\bm S^2}_0$"

label_sqz = r"$-10\log_{10}\xi^2$"
label_sqz_opt = r"$-10\log_{10}\xi_{\mathrm{opt}}^2$"

label_time = r"$t\times J_\perp$"
label_time_opt = r"$t_{\mathrm{opt}}\times J_\perp$"
label_time_rescaled = r"$t\times \left|J_{\mathrm{z}}-J_\perp\right|$"

label_zz = r"$J_{\mathrm{z}}/J_\perp$"

### general fitting / squeezing methods

def to_dB(sqz):
    return -10*np.log10(np.array(sqz))

def log_form(x, a, b):
    return a * np.log(x) + b
def log_fit(x, fit_params):
    return np.vectorize(log_form)(x, *fit_params)

sizes, sqz_OAT = np.loadtxt(data_dir + "sqz_OAT.txt", delimiter = ",", unpack = True)
fit_params_OAT, _ = scipy.optimize.curve_fit(log_form, sizes, to_dB(sqz_OAT))
def fit_sqz_OAT(N): return log_fit(N, fit_params_OAT)
fit_sqz_OAT = np.vectorize(fit_sqz_OAT)

### methods for setting axis ticks

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
axes[0].plot(zz_coupling, to_dB(min_sqz), "k.", label = "TS$_4$")
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
axes[0].axhline(to_dB(sqz_OAT), **kwargs, label = "OAT")
axes[1].axhline(1, **kwargs)

kwargs["linestyle"] = ":"
_, sqz_ising = ising_squeezing_optimum(sunc_mat, TI = periodic)
_, min_SS_ising = ising_minimal_SS(sunc_mat, TI = periodic)
axes[0].axhline(to_dB(sqz_ising), **kwargs, label = "Ising")
axes[1].axhline(min_SS_ising/max_SS, **kwargs)

# tweak axis limits
axes[0].set_ylim(bottom = np.floor(to_dB(sqz_ising)))
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

figsize = (2.75,2.5)
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
    figure, axes = plt.subplots(2, figsize = figsize, sharex = True, sharey = False)
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
# plot benchmarking with exact simulations

lattices = [ "3x3", "4x4" ]
figsize = (5,4)
zz_lims = (-3,3)

def get_file(sim_type, lattice, alpha):
    sim_dir = "DTWA" if sim_type == "dtwa" else sim_type
    return data_dir + f"{sim_dir}/benchmarking/{sim_type}_L{lattice}_a{alpha}.txt"

figure, all_axes = plt.subplots(2, len(lattices), figsize = figsize,
                                sharex = True, sharey = "row")
sqz_axes, SS_axes = all_axes

for lattice, sqz_axis, SS_axis in zip(lattices, sqz_axes, SS_axes):
    dim = len(lattice.split("x"))

    # add a reference line for the OAT limit
    spin_num = np.prod([ int(size) for size in lattice.split("x") ])
    _, sqz_OAT = ising_squeezing_optimum(np.ones((spin_num,spin_num)), TI = True)
    sqz_axis.axhline(to_dB(sqz_OAT), color = "k", linestyle = "--", label = "OAT")
    SS_axis.axhline(1, color = "k", linestyle = "--", label = "OAT")

    # identify values of the power-law exponent \alpha
    alpha_vals = [ set([ int(file.split("_")[-1][1:-4])
                         for file in glob.glob(get_file(sim_type, lattice, "?")) ])
                   for sim_type in [ "dtwa", "exact" ] ]
    alpha_vals = sorted(set.intersection(*alpha_vals))

    # plot all squeezing data
    for alpha in alpha_vals:
        def _get_vals(sim_type):
            data = np.loadtxt(get_file(sim_type,lattice,alpha), unpack = True)
            zz_couplings, sqz_min, SS_min = data[0,:], data[1,:], data[-1,:]
            keep = np.where( ( np.around(zz_couplings, decimals = 1) >= zz_lims[0] ) &
                             ( np.around(zz_couplings, decimals = 1) <= zz_lims[1] ) &
                             ( zz_couplings != 1 ) )[0]
            return zz_couplings[keep], -sqz_min[keep], SS_min[keep]

        exact_zz, exact_sqz, exact_SS = _get_vals("exact")
        dtwa_zz, dtwa_sqz, dtwa_SS = _get_vals("dtwa")

        exact_plot = sqz_axis.plot(exact_zz, exact_sqz, label = alpha)
        color = exact_plot[0].get_color()
        sqz_axis.plot(dtwa_zz, dtwa_sqz, ".", color = color, markersize = 2)

        SS_axis.plot(exact_zz, exact_SS, color = color, label = alpha)
        SS_axis.plot(dtwa_zz, dtwa_SS, ".", color = color, markersize = 2)

    title_text = lattice.replace("x", r"\times")
    sqz_axis.set_title(f"${title_text}$")
    SS_axis.set_xlabel(label_zz)

sqz_axes[0].set_ylabel(label_sqz_opt)
SS_axes[0].set_ylabel(label_SS_min)
sqz_axes[0].legend(loc = "center", ncol = 2, handlelength = 1.5,
                   bbox_to_anchor = (0.29,0.78), columnspacing = 1)
plt.tight_layout()

lattice_label = "_L".join(lattices)
plt.savefig(fig_dir + f"exact_L{lattice_label}.pdf")
plt.close("all")
##########################################################################################
# plot summary of DTWA results

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
            inf_axes[1].plot(zz_lims, [0,0], "c-", **marker_args)

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
                    add_markup = True, figsize = None, font_size = None):
    if font_size is not None: # if we were given a font size, set it
        orig_font_size = plt.rcParams["font.size"]
        plt.rcParams.update({ "font.size" : font_size })

    if type(lattice_list) is str:
        lattice_list = [ lattice_list ]
    cols = len(lattice_list)

    if figsize == None:
        if cols == 1:
            figsize = (2.5, 4)
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
    if font_size is not None: # set font size to its original value
        plt.rcParams.update({ "font.size" : orig_font_size })
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
make_dtwa_plots(lattice_list, font_size = 8)
lattice_label = "_L".join(lattice_list)
plt.savefig(fig_dir + f"dtwa_L{lattice_label}.pdf")

plt.close("all")
##########################################################################################
# plot system size scaling (DTWA)

dz = 0.1
inspect_dz = 0.5
label_alpha = [ 3 ]

figsize = (3,1.8)

# todo: get 16x16x16 data for 3D plots

def plot_scaling(dim, lattice_res, zz_lims, alpha, colors = None, fit_div_alpha = []):
    zz_couplings = np.arange(zz_lims[0], zz_lims[1] + dz/2, dz)

    # identify all relevant files
    def get_file(lattice_text, zz_coupling):
        name_tag = make_name_tag(lattice_text, alpha, "dtwa")
        return data_dir + "DTWA/system_size/dtwa_" + name_tag + f"_z{zz_coupling:.1f}.txt"
    file_tags = [ get_file("*", zz_coupling) for zz_coupling in zz_couplings ]
    files = [ file for file_tag in file_tags for file in glob.glob(file_tag) ]

    # determine lattice lengths
    lattice_lengths = [ int(text_parts[-1]) for file in files
                        if len( text_parts := file.split("_")[-3].split("x") ) == dim ]
    lattice_lengths = np.array(sorted(set(lattice_lengths)))

    # extract simulation data
    sqz_data = np.empty((len(zz_couplings), len(lattice_lengths)), None)
    ss_min_data = np.empty((len(zz_couplings), len(lattice_lengths)), None)
    for zz_idx, zz_coupling in enumerate(zz_couplings):
        # skip the critical point at J_z = 1
        if np.allclose(zz_coupling,1):
            sqz_data[zz_idx,:] = None
            continue

        for ll_idx, lattice_length in enumerate(lattice_lengths):
            lattice_text = "x".join([str(lattice_length)]*dim)
            data = np.loadtxt(get_file(lattice_text, zz_coupling))
            sqz_min_idx = data[:,4].argmin()
            sqz_data[zz_idx,ll_idx] = -data[sqz_min_idx,4]
            ss_min_data[zz_idx,ll_idx] = min(data[:sqz_min_idx+1,11])
            del data

    # identify the dynamical phase boundary
    zz_boundaries = zz_couplings[ ss_min_data[zz_couplings < 1, :].argmin(axis = 0) ]

    # identify lattice lengths to plot
    LL_keep = np.isclose(lattice_lengths % lattice_res, 0)
    LL_image = lattice_lengths[LL_keep]
    size_lims = LL_image[0], LL_image[-1]

    # plot squeezing data
    figure = plt.figure(figsize = figsize)
    axis_lims = [ zz_lims[0] - dz/2, zz_lims[1] + dz/2,
                  size_lims[0] - lattice_res/2, size_lims[1] + lattice_res/2 ]
    plot_args = dict( aspect = "auto", origin = "lower",
                      interpolation = "nearest", cmap = "inferno",
                      extent = axis_lims )
    image = plt.imshow(sqz_data.T[LL_keep,:], **plot_args)

    # add color bar
    color_bar = figure.colorbar(image, label = label_sqz_opt)
    fix_ticks(color_bar, 5)

    # add reference line for dynamical phase boundary
    def alternate(values, other_values):
        return np.array(list(zip(values, other_values))).flatten()
    def double(values):
        return alternate(values,values)
    LL_offsets = lattice_res/2 * alternate(- np.ones(len(LL_image)),
                                           + np.ones(len(LL_image)))
    plt.plot(double(zz_boundaries[LL_keep]), double(LL_image) + LL_offsets,
             ":", color = "gray", linewidth = 1)

    # label axes
    plt.xlabel(label_zz)
    plt.ylabel(r"$L$")

    # set axis ticks
    zz_ticks = sorted(set([ int(zz) for zz in zz_couplings ]))
    zz_labels = [ zz if zz % 2 == 0 else "" for zz in zz_ticks ]
    plt.xticks(zz_ticks, zz_labels)

    if dim == 2:
        LL_labels = [ LL if LL % 10 == 0 else "" for LL in LL_image ]
    if dim == 3:
        LL_labels = [ LL if LL % 5 == 0 else "" for LL in LL_image ]
    plt.yticks(LL_image, LL_labels)

    # label each dynamical phase
    if alpha in label_alpha and dim == 2:
        text_args = dict( horizontalalignment = "center",
                          verticalalignment = "center" )
        plt.text(0, 30, "collective", color = "black", **text_args)
        plt.text(-2, 20, "Ising", color = "white", **text_args)
        plt.text(1.6, 20, "Ising", color = "white", **text_args)

    plt.tight_layout(pad = 0.1)
    plt.savefig(fig_dir + f"size_scaling_D{dim}_a{alpha}.pdf")

    ### show logarithmic divergence of critical coupling

    # get system sizes
    system_sizes = lattice_lengths**dim

    # DTWA results
    plt.figure(figsize = figsize)
    plt.semilogx(system_sizes, zz_boundaries, "ko", label = "DTWA")

    # find the best log fit of critical coupling as a function of system size
    if alpha in fit_div_alpha:
        fit_slope, _ = scipy.optimize.curve_fit(log_form, system_sizes, zz_boundaries)
        plt.semilogx(system_sizes, log_fit(system_sizes, fit_slope), "k--", label = "fit")
    if alpha in label_alpha:
        plt.legend(loc = "best", handlelength = 1.7)

    plt.xlabel(r"$N$")
    plt.ylabel(r"$J_{\mathrm{z}}^{\mathrm{crit}}/J_\perp$")
    plt.tight_layout(pad = 0.1)
    plt.savefig(fig_dir + f"size_divergence_D{dim}_a{alpha}.pdf")

    ### show power-law scaling of optimal squeezing

    # determine the values of zz couplings to plot
    inspect_zz = np.arange(1-inspect_dz, min(zz_boundaries)-dz/2, -inspect_dz)
    inspect_zz = sorted([ zz_coupling for zz_coupling in inspect_zz
                          if np.sum(zz_boundaries < zz_coupling) >= 4 ])
    if colors is None: colors = len(inspect_zz)

    plt.figure(figsize = figsize)

    for idx, zz_coupling in enumerate(inspect_zz[::-1]):
        zz_idx = np.isclose(zz_couplings,zz_coupling)
        ll_idx = zz_boundaries < zz_coupling

        _system_sizes = system_sizes[ll_idx]
        sqz_vals = sqz_data[zz_idx, ll_idx]
        fit_params, _ = scipy.optimize.curve_fit(log_form, _system_sizes, sqz_vals)

        color_val = idx / ( colors - 1 )
        color = color_map(1-color_val)
        plt.semilogx(_system_sizes, sqz_vals, "o", color = color)
        plt.semilogx(_system_sizes, log_fit(_system_sizes, fit_params), "--", color = color)

    fit_handles = [ mpl.lines.Line2D([0], [0], color = "k", linestyle = "none", marker = "o"),
                    mpl.lines.Line2D([0], [0], color = "k", linestyle = "--") ]
    fit_labels = [ "DTWA", "fit" ]

    # add the OAT limit as a reference
    plt.semilogx(system_sizes, fit_sqz_OAT(system_sizes), "r:", label = "OAT")
    handles, labels = plt.gca().get_legend_handles_labels()
    fit_handles += handles
    fit_labels += labels

    plt.xlabel(r"$N$")
    plt.ylabel(label_sqz_opt)
    if alpha in label_alpha:
        plt.legend(fit_handles, fit_labels, loc = "best", handlelength = 1.7)

    plt.tight_layout(pad = 0.1)
    plt.savefig(fig_dir + f"power_law_D{dim}_a{alpha}.pdf")

    plt.close("all")

    return colors

dim = 2
lattice_res = 5
zz_lims = ( -2.5, 2 )
colors = None
for alpha in [ 3, 4, 5, 6, "nn" ]:
    colors = plot_scaling(dim, lattice_res, zz_lims, alpha, colors, fit_div_alpha = [ 3, 4 ])

dim = 3
lattice_res = 1
zz_lims = ( -3, 2 )
colors = None
for alpha in [ 3, 4, 5, 6 ]:
    colors = plot_scaling(dim, lattice_res, zz_lims, alpha, colors)

##########################################################################################
# plot dependence on filling fraction (DTWA)

figsize = (3,2)

data = np.loadtxt(data_dir + "DTWA/filling_fraction.txt")
markers = [ "o", "s", "^" ]
labels = [ f"${zz}$" for zz in [ 0.9, 0, -1 ] ]
spin_num = 50**2

plt.figure(figsize = figsize)
for zz_idx, marker, label in zip(reversed(range(1,4)), markers, labels):
    plt.plot(data[:,0], -data[:,zz_idx], marker, label = label)
plt.plot(data[:,0], fit_sqz_OAT(data[:,0] * spin_num), "k:", label = "OAT")

plt.gca().set_xlim(-0.05, 1.05)
plt.gca().set_ylim(bottom = 0)

plt.xlabel("$f$")
plt.ylabel(label_sqz_opt)

plt.legend(loc = "best", handlelength = 1.7)
plt.tight_layout()
plt.savefig(fig_dir + "filling_fraction_a3.pdf")

plt.close("all")
