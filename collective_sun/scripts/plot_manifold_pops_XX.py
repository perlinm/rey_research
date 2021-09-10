#!/usr/bin/env python3

import sys, scipy, glob
import numpy as np
import matplotlib.pyplot as plt

from dicke_methods import coherent_spin_state

params = { "font.size" : 12,
           "text.usetex" : True,
           "text.latex.preamble" : r"\usepackage{braket,amsmath}" }
plt.rcParams.update(params)

if len(sys.argv) != 2:
    print(f"usage: {sys.argv[0]} [max_manifold]")
    exit()

max_manifold = int(sys.argv[1])

data_dir = "../data/shells_XX/"
partial_dir = data_dir + "partial/"
fig_dir = "../figures/shells_XX/"

name_tag = f"L*_c*_M{max_manifold}"

############################################################
# population of |X> within the scar-state manifold

markers = { 1 : "o", 2 : "v", 3 : "s" }
lines = [None]*3

plt.figure(figsize = (3,2))

for file in glob.glob(data_dir + f"scar_pop_X_{name_tag}.txt"):
    cutoff = int(file.split("_")[-2][1:])
    lattice_shape = file.split("_")[-3][1:]
    spin_num = np.prod([ int(size) for size in lattice_shape.split("x") ])
    pop = np.loadtxt(file)
    lines[cutoff-1] = plt.plot(spin_num, pop, "k" + markers[cutoff])[0]

labels = [ r"$r_\mathrm{b}=" + f"{cutoff}$" for cutoff in markers.keys() ]
kwargs = dict( loc = "best", handlelength = 0.5, labelspacing = 0.3 )
plt.legend(lines, labels, **kwargs)

plt.xlabel("$L$")
plt.ylabel(r"$\braket{\mathrm{X}|\mathcal{P}_{\mathrm{scar}}|\mathrm{X}}$")
plt.ylim(0,1.1)
plt.xlim(left = 0)
plt.xticks([0,10,20,30,40])
plt.tight_layout(pad = 0.2)
plt.savefig(fig_dir + f"scar_pops_X_M{max_manifold}.pdf")

############################################################
# population of |X_scar> in different exitation manifolds

colors = { 0 : "tab:blue",
           2 : "tab:orange",
           3 : "tab:green",
           4 : "tab:red" }

plt.figure(figsize = (3.5,2))

for file in glob.glob(data_dir + f"sim_{name_tag}_scar.txt"):
    cutoff = int(file.split("_")[-3][1:])
    lattice_shape = file.split("_")[-4][1:]
    spin_num = np.prod([ int(size) for size in lattice_shape.split("x") ])

    manifold_shells = {}
    with open(file) as lines:
        for line in lines:
            if line[:10] != "# manifold": continue
            parts = line.split()
            manifold = int(parts[2])
            shells = list(map(int,parts[4:]))
            manifold_shells[manifold] = np.array(shells)
    manifolds = list(manifold_shells.keys())
    shell_pops = np.loadtxt(file, dtype = complex)[0,8:].real
    norm = shell_pops.sum()

    manifold_pops = { manifold : shell_pops[shells].sum()/norm
                      for manifold, shells in manifold_shells.items() }
    for manifold, pop in manifold_pops.items():
        kwargs = dict( marker = markers[cutoff], color = colors[manifold],
                       zorder = manifold )
        plt.plot(spin_num, pop, **kwargs)

from matplotlib.lines import Line2D
lines = [ Line2D([0], [0], color = "k", marker = markers[1], lw = 0),
          Line2D([0], [0], color = "k", marker = markers[2], lw = 0),
          Line2D([0], [0], color = "k", marker = markers[3], lw = 0),
          Line2D([0], [0], color = colors[0], lw = 3),
          Line2D([0], [0], color = colors[2], lw = 3),
          Line2D([0], [0], color = colors[3], lw = 3),
          Line2D([0], [0], color = colors[4], lw = 3) ]
labels = [ r"$r_\mathrm{b}=1$", r"$r_\mathrm{b}=2$", r"$r_\mathrm{b}=3$",
           "$M=0$", "$M=2$", "$M=3$", "$M=4$" ]
kwargs = dict( loc = "center left", bbox_to_anchor = (1,0.5),
               handlelength = 0.5, labelspacing = 0.3 )
plt.legend(lines, labels, **kwargs)

plt.xlabel("$L$")
plt.ylabel(r"$\braket{\mathrm{X}_{\mathrm{scar}}|\mathcal{P}_M|\mathrm{X}_{\mathrm{scar}}}$")
plt.ylim(0,1.1)
plt.xlim(left = 0)
plt.xticks([0,10,20,30,40])
plt.tight_layout(pad = 0.2)
plt.savefig(fig_dir + f"scar_X_manifold_pops_M{max_manifold}.pdf")
