#!/usr/bin/env python3

import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(linewidth = 200)

if len(sys.argv) < 4:
    print(f"usage: {sys.argv[0]} [alpha] [max_manifold] [lattice_shape] [shells/spins]")
    exit()

alpha = float(sys.argv[1]) # power-law couplings ~ 1 / r^\alpha
max_manifold = int(sys.argv[2])
lattice_shape = tuple(map(int, sys.argv[3:-1]))
sim_type = sys.argv[-1]

assert(sim_type in [ "shells", "spins" ])

project = True # use data from "projected" simulations?
plot_all_shells = False # plot the population for each shell?
squeezing_refline = True # mark optimal squeezing time in population time-series plots?

figsize = (5,4)
params = { "font.size" : 16,
           "text.usetex" : True,
           "text.latex.preamble" : [ r"\usepackage{amsmath}",
                                     r"\usepackage{braket}" ]}
plt.rcParams.update(params)

data_dir = f"../data/{sim_type}/"
fig_dir = f"../figures/{sim_type}/"

if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

if sim_type == "spins" and project:
    data_dir += "proj_"
    fig_dir += "proj_"

##########################################################################################

if np.allclose(alpha, int(alpha)): alpha = int(alpha)
def name_tag(coupling_zz = None):
    lattice_name = "_".join([ str(size) for size in lattice_shape ])
    base_tag = f"L{lattice_name}_M{max_manifold}_a{alpha}"
    if coupling_zz == None: return base_tag
    else: return base_tag + f"_z{coupling_zz}"

def pop_label(manifold, subscript = None):
    label = r"\braket{\mathcal{M}_{" + str(manifold) + r"}}"
    if subscript == None:
        return "$" + label + "$"
    else:
        return "$" + label + "_{\mathrm{" + subscript + "}}$"

def to_dB(sqz):
    return 10*np.log10(np.array(sqz))

##########################################################################################
print("plotting inspection data")

lattice_text = r"\times".join([ str(size) for size in lattice_shape ])
common_title = f"L={lattice_text},~\\alpha={alpha}"

for data_file in glob.glob(data_dir + "inspect_" + name_tag() + "_z*.txt"):
    coupling_zz = data_file.split("_")[-1][1:-4]
    title_text = f"${common_title},~J_{{\mathrm{{z}}}}/J_\perp={coupling_zz}$"

    data = np.loadtxt(data_file)
    times = data[:,0]
    sqz = data[:,1]
    pops = data[:,2:]

    manifold_shells = {}
    with open(data_file, "r") as file:
        for line in file:
            if "# manifold " in line:
                manifold = line.split()[2]
                shells = np.array([ int(shell) for shell in line.split()[4:] ])
                manifold_shells[manifold] = shells
            if line[0] != "#": break

    try:
        sqz_end = np.where(sqz[1:] > 1)[0][0] + 2
    except:
        sqz_end = len(times)

    plt.figure(figsize = figsize)
    plt.title(title_text)
    plt.plot(times[:sqz_end], to_dB(sqz[:sqz_end]), "k")
    plt.xlim(0, plt.gca().get_xlim()[1])
    plt.ylim(plt.gca().get_ylim()[0], 0)
    plt.xlabel(r"time ($J_\perp t$)")
    plt.ylabel(r"squeezing ($10\log_{10}\xi^2$)")
    plt.tight_layout()
    plt.savefig(fig_dir + f"squeezing_{name_tag(coupling_zz)}.pdf")

    plt.figure(figsize = figsize)
    plt.title(title_text)
    for manifold, shells in manifold_shells.items():
        manifold_pops = pops[:,shells].sum(axis = 1)
        if np.allclose(max(manifold_pops),0): continue
        if sim_type == "shells" and plot_all_shells:
            for shell in shells:
                plt.plot(times, pops[:,shell], color = "gray", linestyle = "--")
        plt.plot(times, manifold_pops, label = pop_label(manifold))
    if squeezing_refline:
        plt.axvline(times[np.argmin(sqz)], color = "gray", linestyle  = "--")
    plt.xlabel(r"time ($J_\perp t$)")
    plt.ylabel("population")
    plt.legend(loc = "best")
    plt.tight_layout()
    plt.savefig(fig_dir + f"populations_{name_tag(coupling_zz)}.pdf")

##########################################################################################
print("plotting sweep data")

title_text = f"${common_title}$"

sweep_file = data_dir + "sweep_" + name_tag() + ".txt"
if not os.path.isfile(sweep_file): exit()

sweep_data = np.loadtxt(sweep_file)
with open(sweep_file, "r") as file:
    for line in file:
        if "# manifolds : " in line:
            manifolds = line.split()[3:]
        if line[0] != "#": break

sweep_coupling_zz = sweep_data[:,0]
sweep_min_sqz = sweep_data[:,1]
sweep_min_pops_0 = sweep_data[:,2]
sweep_max_pops = sweep_data[:,3:]

plt.figure(figsize = figsize)
plt.title(title_text)
plt.plot(sweep_coupling_zz, to_dB(sweep_min_sqz), "ko")
plt.ylim(plt.gca().get_ylim()[0], 0)
plt.xlabel(r"$J_{\mathrm{z}}/J_\perp$")
plt.ylabel(r"$\xi_{\mathrm{min}}^2$ (dB)")
plt.tight_layout()
plt.savefig(fig_dir + f"squeezing_{name_tag()}.pdf")

plt.figure(figsize = figsize)
plt.title(title_text)
plt.plot(sweep_coupling_zz, sweep_min_pops_0, "o", label = pop_label(0,"min"))
for manifold, max_pops in zip(manifolds, sweep_max_pops.T):
    plt.plot(sweep_coupling_zz, max_pops, "o", label = pop_label(manifold,"max"))
plt.xlabel(r"$J_{\mathrm{z}}/J_\perp$")
plt.ylabel("population")
plt.legend(loc = "best", handletextpad = 0.1)
plt.tight_layout()
plt.savefig(fig_dir + f"populations_{name_tag()}.pdf")

print("completed")
