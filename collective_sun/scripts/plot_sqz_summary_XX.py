#!/usr/bin/env python3

import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

figsize = (3.5,2.4)
params = { "font.size" : 12,
           "text.usetex" : True }
plt.rcParams.update(params)

if len(sys.argv) != 2:
    print(f"usage: {sys.argv[0]} [max_manifold]")
    exit()

dim = 1 # only support a 1-D chain, for now
max_manifold = int(sys.argv[1])

pop_cutoff = 0.1

data_dir = "../data/shells_XX/"
partial_dir = data_dir + "partial/"
fig_dir = "../figures/shells_XX/"

name_tag = f"sim_L*_r*_M{max_manifold}"

def get_sim_data(data_file):
    manifold_shells = {}
    with open(data_file,"r") as file:
        for line in file:
            if line[:10] != "# manifold": continue
            parts = line.split()
            manifold = int(parts[2])
            shells = list(map(int,parts[4:]))
            manifold_shells[manifold] = np.array(shells)
    manifolds = list(manifold_shells.keys())
    data = np.loadtxt(data_file, dtype = complex)

    str_ops = [ "Z", "+", "ZZ", "++", "+Z", "+-" ]
    times = data[:,0].real
    sqz = data[:,len(str_ops)+1].real
    pops = data[:,len(str_ops)+2:].real

    return times, sqz, pops, manifold_shells

##################################################

for data_file in sorted(glob.glob(os.path.join(data_dir,name_tag+"*.txt"))):
    file_parts = data_file.split("/")[-1].split("_")[1:]
    lattice_shape = file_parts[0][1:]
    if lattice_shape.count("x") + 1 != dim: continue
    spin_num = np.prod([ int(size) for size in lattice_shape.split("x") ])
    radius = int(file_parts[1][1:])

    plt.figure(str(radius), figsize = figsize)
    times, sqz, pops, manifold_shells = get_sim_data(data_file)

    if "scar" in data_file:
        fill_style = "none" if "scar" in data_file else "full"
        plt.semilogy(spin_num, min(sqz), "ko", fillstyle = fill_style)

    else: # not scar
        plt.semilogy(spin_num, min(sqz), "ko")

        if max_manifold not in manifold_shells: continue
        max_manifold_pops = pops[:,manifold_shells[max_manifold]].sum(axis = 1)
        trust_sqz = sqz[ max_manifold_pops < pop_cutoff ]
        plt.semilogy(spin_num, min(trust_sqz), "ko", markerfacecolor = "gray")

for radius in plt.get_figlabels():
    plt.figure(radius)
    plt.title(r"$r_{\mathrm{b}}=" + f"{radius}$")

    kwargs = dict( color = "k", marker = "o", linestyle = "none" )
    handles = [ Line2D([0], [0], **kwargs),
                Line2D([0], [0], **kwargs, markerfacecolor = "gray"),
                Line2D([0], [0], **kwargs, fillstyle = "none") ]
    labels = [ "trunc", "trusted", "scar" ]
    kwargs = dict( loc = "best", handlelength = 0.5, labelspacing = 0.3 )
    plt.legend(handles, labels, **kwargs)

    y_btm = plt.gca().get_ylim()[0]
    y_btm_new = 10**np.floor(np.log10(y_btm))
    plt.ylim(y_btm_new, 1)
    plt.gca().set_yticklabels([], minor = True)

    x_top = plt.gca().get_xlim()[1]
    plt.xlim(0, x_top)

    plt.xlabel(r"$L$")
    plt.ylabel(r"$\xi^2_{\mathrm{min}}$")
    plt.tight_layout(pad = 0.2)
    plt.savefig(fig_dir + f"sqz_summary_r{radius}.pdf")
