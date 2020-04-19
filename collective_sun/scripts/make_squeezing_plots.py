#!/usr/bin/env python3

import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(linewidth = 200)

if len(sys.argv) < 4:
    print(f"usage: {sys.argv[0]} [proj?] [lattice_shape] [alpha] [max_manifold] [shells/spins]")
    exit()

# determine whether to use data from "projected" simulations
#   (for the case of exact spin simulations)
if "proj" in sys.argv:
    project = True
    sys.argv.remove("proj")
else:
    project = False

lattice_shape = tuple(map(int, sys.argv[1].split("x")))
alpha = float(sys.argv[2]) # power-law couplings ~ 1 / r^\alpha
max_manifold = int(sys.argv[3])

sim_type = sys.argv[4]
assert(sim_type in [ "shells", "spins" ])

plot_all_shells = False # plot the population for each shell?
squeezing_refline = True # mark optimal squeezing time in population time-series plots?

figsize = (5,4)
params = { "font.size" : 16,
           "text.usetex" : True,
           "text.latex.preamble" : [ r"\usepackage{braket}",
                                     r"\usepackage{bm}" ]}
plt.rcParams.update(params)

data_dir = f"../data/{sim_type}/"
fig_dir = f"../figures/{sim_type}/"
inspect_dir = fig_dir + "inspect/"

if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)
if not os.path.isdir(inspect_dir):
    os.makedirs(inspect_dir)

if sim_type == "spins" and project:
    data_dir += "proj_"
    fig_dir += "proj_"

if np.allclose(alpha, int(alpha)): alpha = int(alpha)
lattice_name = "x".join([ str(size) for size in lattice_shape ])
name_tag = f"L{lattice_name}_a{alpha}_M{max_manifold}"

##########################################################################################

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
common_title = f"$L={lattice_text}$, $\\alpha={alpha}$"

inspect_files = sorted(glob.glob(data_dir + f"inspect_{name_tag}" + "_z*.txt"),
                       key = lambda name : float(name.split("_")[-1][1:-4]))
for data_file in inspect_files:
    zz_coupling = data_file.split("_")[-1][1:-4]
    coupling_title = f"{common_title}, $J_{{\mathrm{{z}}}}/J_\perp={zz_coupling}$"

    data = np.loadtxt(data_file, dtype = complex)
    times = data[:,0].real
    correlators = { "Z" : data[:,1],
                    "+" : data[:,2],
                    "ZZ" : data[:,3],
                    "++" : data[:,4],
                    "+Z" : data[:,5],
                    "+-" : data[:,6] }
    sqz = data[:,7].real
    pops = data[:,8:].real

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

    plt.figure("sqz", figsize = figsize)
    plt.plot(times[:sqz_end], to_dB(sqz[:sqz_end]), label = f"${zz_coupling}$")

    plt.figure("SS", figsize = figsize)
    correlator_SS = correlators["ZZ"].real/4 + correlators["+-"].real
    plt.plot(times, correlator_SS, label = f"${zz_coupling}$")

    plt.figure("pop", figsize = figsize)
    plt.title(coupling_title)
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
    plt.savefig(inspect_dir + f"populations_{name_tag}_z{zz_coupling}.pdf")
    plt.close("pop")

if inspect_files:
    plt.figure("sqz")
    if squeezing_refline:
        plt.axvline(times[np.argmin(sqz)], color = "gray", linestyle  = "--")
    plt.title(common_title)
    plt.gca().set_ylim(top = 0)
    plt.xlabel(r"time ($J_\perp t$)")
    plt.ylabel(r"squeezing ($10\log_{10}\xi^2$)")
    plt.legend(loc = "best")
    plt.tight_layout()
    plt.savefig(inspect_dir + f"squeezing_{name_tag}.pdf")
    plt.close("sqz")

    plt.figure("SS")
    if squeezing_refline:
        plt.axvline(times[np.argmin(sqz)], color = "gray", linestyle  = "--")
    plt.title(common_title)
    plt.xlabel(r"$J_\perp t$")
    plt.ylabel(r"$\braket{\bm S^2}$")
    plt.legend(loc = "best")
    plt.tight_layout()
    plt.savefig(inspect_dir + f"SS_{name_tag}.pdf")
    plt.close("SS")

##########################################################################################
sweep_file = data_dir + f"sweep_{name_tag}.txt"
if not os.path.isfile(sweep_file): exit()
print("plotting sweep data")

data = np.loadtxt(sweep_file, dtype = complex)
with open(sweep_file, "r") as file:
    for line in file:
        if "# manifolds : " in line:
            manifolds = line.split()[3:]
        if line[0] != "#": break

zz_coupling = data[:,0].real
time_opt = data[:,1].real
min_sqz = data[:,2].real
correlators_opt = { "Z" : data[:,3],
                    "+" : data[:,4],
                    "ZZ" : data[:,5],
                    "++" : data[:,6],
                    "+Z" : data[:,7],
                    "+-" : data[:,8] }
min_SS = data[:,9].real
min_pops_0 = data[:,10].real
max_pops = data[:,11:].real

plt.figure(figsize = figsize)
plt.title(common_title)
plt.plot(zz_coupling, to_dB(min_sqz), "ko")
plt.gca().set_ylim(top = 0)
plt.xlabel(r"$J_{\mathrm{z}}/J_\perp$")
plt.ylabel(r"$10\log_{10}\xi_{\mathrm{min}}^2$")
plt.tight_layout()
plt.savefig(fig_dir + f"squeezing_{name_tag}.pdf")

plt.figure(figsize = figsize)
plt.title(common_title)
plt.plot(zz_coupling, time_opt, "ko")
plt.xlabel(r"$J_{\mathrm{z}}/J_\perp$")
plt.ylabel(r"$t_{\mathrm{opt}} J_\perp$")
plt.tight_layout()
plt.savefig(fig_dir + f"time_opt_{name_tag}.pdf")

plt.figure(figsize = figsize)
plt.title(common_title)
plt.plot(zz_coupling, min_SS, "ko")
plt.xlabel(r"$J_{\mathrm{z}}/J_\perp$")
plt.ylabel(r"$\braket{\bm S^2}_{\mathrm{min}}$")
plt.tight_layout()
plt.savefig(fig_dir + f"SS_{name_tag}.pdf")

plt.figure(figsize = figsize)
plt.title(common_title)
plt.plot(zz_coupling, min_pops_0, "o", label = pop_label(0,"min"))
for manifold, max_pops in zip(manifolds[1:], max_pops.T):
    plt.plot(zz_coupling, max_pops, "o", label = pop_label(manifold,"max"))
plt.xlabel(r"$J_{\mathrm{z}}/J_\perp$")
plt.ylabel("population")
plt.legend(loc = "best", handletextpad = 0.1)
plt.tight_layout()
plt.savefig(fig_dir + f"populations_{name_tag}.pdf")

print("completed")
