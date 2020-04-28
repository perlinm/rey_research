#!/usr/bin/env python3

import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

np.set_printoptions(linewidth = 200)

data_dir = "../data/"
fig_dir = "../figures/XXZ/"

params = { "font.size" : 9,
           "text.usetex" : True,
           "text.latex.preamble" : [ r"\usepackage{braket}",
                                     r"\usepackage{bm}" ]}
plt.rcParams.update(params)

def make_name_tag(lattice_text, alpha, data_format = "shell"):
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
label_sqz = r"$-10\log_{10}\xi_{\mathrm{min}}^2$"
label_time = r"$t_{\mathrm{opt}}\times J_\perp$"

##########################################################################################
# collect shell model data

lattice = (7,7)
alpha = 3
excess_pop = 0.1
zz_lims = (-1,3)

spin_num = np.prod(lattice)

lattice_text = "x".join([ str(ll) for ll in lattice ])
name_tag = make_name_tag(lattice_text, alpha)
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
# make population plot

plt.figure(figsize = (3,1.8))
for idx, pops in enumerate(max_pops.T):
    plt.plot(zz_coupling, pops, ".", label = pop_label(idx+1))
shade_exclusions()
plt.xlabel(r"$J_{\mathrm{z}}/J_\perp$")
plt.ylabel("population")
plt.legend(loc = "center", handlelength = 0.5, bbox_to_anchor = (0.47,0.66))
plt.tight_layout(pad = 0.2)
plt.savefig(fig_dir + f"populations_{name_tag}.pdf")

##################################################
# make DTWA benchmarking plot

figure, axes = plt.subplots(2, figsize = (3,3))
axes[0].plot(zz_coupling, -to_dB(min_sqz), "k.", label = "TS$_4$")
axes[1].plot(zz_coupling, min_SS / (spin_num/2*(spin_num/2+1)), "k.")

# collect DTWA data
name_tag_dtwa = make_name_tag(lattice_text, alpha, "dtwa")
zz_coupling, min_sqz_dB, min_SS_normed \
    = np.loadtxt(data_dir + f"DTWA/dtwa_{name_tag_dtwa}.txt", unpack = True)

axes[0].plot(zz_coupling, -min_sqz_dB, "r.", label = "DTWA")
axes[1].plot(zz_coupling, min_SS_normed, "r.")

# reference: collective and Ising limits
spin_nums, sqz_vals = np.loadtxt(data_dir + "sqz_OAT.txt", delimiter = ",", unpack = True)
sqz_OAT_dB = np.interp(np.log(spin_num), np.log(spin_nums), -to_dB(sqz_vals))
axes[0].plot(axes[0].get_xlim(), [sqz_OAT_dB]*2, "k--")
axes[1].plot(axes[0].get_xlim(), [1]*2, "k--", label = "OAT")

axes[0].set_ylabel(label_sqz)
axes[1].set_ylabel(label_SS)
axes[1].set_xlabel(r"$J_{\mathrm{z}}/J_\perp$")
shade_exclusions(axes[0])
shade_exclusions(axes[1])
axes[0].legend(loc = "center", handlelength = 0.5, bbox_to_anchor = (0.45,0.3))
axes[1].legend(loc = "center", handlelength = 1.7, bbox_to_anchor = (0.45,0.3))
plt.tight_layout(pad = 0.8)
plt.savefig(fig_dir + f"benchmarking_{name_tag}.pdf")
# plt.show()
exit()
##########################################################################################
# plot primary DTWA results

lattice_text = "64x64"
zz_lims = (-3,3)

# import all relevant data
name_tag = make_name_tag(lattice_text, "*", "dtwa")
data_files = data_dir + f"DTWA/dtwa_{name_tag}.txt"
raw_data = {}
for file in glob.glob(data_files):
    alpha = file.split("_")[-1][1:-4]
    raw_data[alpha] = np.loadtxt(file)

# collect data for plotting
alpha_vals = sorted(list(raw_data.keys()))
zz_couplings = raw_data["nn"][:,0]
data = {}
keep = np.where(np.logical_and(np.around(zz_couplings, decimals = 1) >= zz_lims[0],
                               np.around(zz_couplings, decimals = 1) <= zz_lims[1]))[0]
data["min_sqz"] = np.array([ raw_data[alpha][keep,1] for alpha in alpha_vals ])
data["opt_tim"] = np.array([ raw_data[alpha][keep,2] for alpha in alpha_vals ])
data["sqr_len"] = np.array([ raw_data[alpha][keep,3] for alpha in alpha_vals ])
del raw_data
zz_couplings = zz_couplings[keep]

# insert "empty" data for zz_coupling = 1
zz_couplings = np.insert(zz_couplings, np.where(zz_couplings > 1)[0][0], 1)
for key in data.keys():
    data[key] = np.insert(data[key], np.where(zz_couplings >= 1)[0][0], None, axis = 1)

# plot data
figure, axes = plt.subplots(3, figsize = (3,4))
image = {}
kwargs = { "aspect" : "auto", "origin" : "lower", "cmap" : "viridis" }
image[0] = axes[0].imshow(+data["sqr_len"], **kwargs)
image[1] = axes[1].imshow(-data["min_sqz"], **kwargs)
image[2] = axes[2].imshow(+data["opt_tim"], **kwargs, norm = LogNorm())

# make colorbars
bar = {}
bar[0] = figure.colorbar(image[0], ax = axes[0], label = label_SS)
bar[1] = figure.colorbar(image[1], ax = axes[1], label = label_sqz, ticks = [ 5, 10, 15, 20 ])
bar[2] = figure.colorbar(image[2], ax = axes[2], label = label_time)

# set axis labels
for axis in axes:
    axis.set_ylabel(r"$\alpha$")
    axis.set_yticks(range(len(alpha_vals)))
    axis.set_yticklabels(alpha_vals)
    axis.set_xticks(np.where(np.isclose(zz_couplings % 1, 0))[0])
axes[0].set_xticklabels([])
axes[1].set_xticklabels([])
axes[2].set_xticklabels(sorted(set(np.around(zz_couplings).astype(int))))
axes[2].set_xlabel(r"$J_{\mathrm{z}}/J_\perp$")

# reference lines to separate nearest neighbor coupling
for axis in axes:
    axis.plot(axis.get_xlim(), [axis.get_ylim()[1]-1]*2, "w")

plt.tight_layout(pad = 0)
plt.savefig(fig_dir + f"dtwa_results_L{lattice_text}.pdf")
