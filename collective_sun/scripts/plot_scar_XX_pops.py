#!/usr/bin/env python3

import sys, scipy, glob
import numpy as np
import matplotlib.pyplot as plt

from dicke_methods import coherent_spin_state

figsize = (3,2)
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

##################################################

markers = { 1 : "o", 2 : "v", 3 : "s" }
lines = [None]*3

plt.figure(figsize = figsize)

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
