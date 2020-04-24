#!/usr/bin/env python3

import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(linewidth = 200)

data_dir = "../data/"
fig_dir = "../figures/XXZ/"

params = { "font.size" : 9,
           "text.usetex" : True,
           "text.latex.preamble" : [ r"\usepackage{braket}",
                                     r"\usepackage{bm}" ]}
plt.rcParams.update(params)

def make_name_tag(lattice_text, alpha, data_format = "shell"):
    if data_format == "shell":
        return f"L{lattice_text}_a{alpha}_M4"
    if data_format == "dtwa":
        return f"{lattice_text}_alpha_{alpha}"
    else:
        print(f"data file format unrecognized : {data_format}")

def pop_label(manifold):
    return r"$\braket{\mathcal{P}_{" + str(manifold) + r"}}_{\mathrm{max}}$"

def to_dB(sqz):
    return 10*np.log10(np.array(sqz))

##########################################################################################
# collect shell model data

lattice = (6,6)
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
plt.legend(loc = "center", handlelength = 0.5, bbox_to_anchor = (0.48,0.66))
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

axes[0].set_ylabel(r"$-10\log_{10}\xi_{\mathrm{min}}^2$")
axes[1].set_ylabel(r"$\braket{\bm S^2}_{\mathrm{min}} / \braket{\bm S^2}_0$")
axes[1].set_xlabel(r"$J_{\mathrm{z}}/J_\perp$")
shade_exclusions(axes[0])
shade_exclusions(axes[1])
axes[0].legend(loc = "center", handlelength = 0.5, bbox_to_anchor = (0.43,0.3))
plt.tight_layout(pad = 0.8)
plt.savefig(fig_dir + f"benchmarking_{name_tag}.pdf")
