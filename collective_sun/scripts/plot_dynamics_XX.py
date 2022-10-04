#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 4:
    print(f"usage: {sys.argv[0]} [scar?] [lattice_shape] [radius] [alpha] [max_manifold]")
    exit()

params = { "font.size" : 12,
           "text.usetex" : True }
plt.rcParams.update(params)

# determine whether to project the initial state |X> onto the "scar manifold"
if "scar" in sys.argv:
    scar_proj = True
    sys.argv.remove("scar")
else:
    scar_proj = False

lattice_shape = tuple(map(int, sys.argv[1].split("x")))
radius_text = sys.argv[2]
radius = float(radius_text) # range of square-well interaction (in units of lattice spacing)
alpha = int(sys.argv[3])
max_manifold = int(sys.argv[4])

figsize = (3,2)
data_dir = "../data/shells_XX/"
fig_dir = "../figures/shells_XX/"

lattice_name = "x".join([ str(size) for size in lattice_shape ])
name_tag = f"L{lattice_name}_r{radius_text}_a{alpha}_M{max_manifold}"
if scar_proj: name_tag += "_scar"

data_file = data_dir + f"sim_{name_tag}.txt"
manifold_shells = {}
with open(data_file) as file:
    for line in file:
        if line[:10] != "# manifold": continue
        parts = line.split()
        manifold = int(parts[2])
        shells = list(map(int,parts[4:]))
        manifold_shells[manifold] = np.array(shells)
manifolds = list(manifold_shells.keys())
data = np.loadtxt(data_file, dtype = complex)

times = data[:,0].real
str_ops = [ "Z", "+", "ZZ", "++", "+Z", "+-" ]
correlators = {}
for str_op, op in zip(str_ops, data[:,1:].T):
    correlators[str_op] = op
sqz = data[:,len(str_ops)+1].real
pops = data[:,len(str_ops)+2:].real

plt.figure(figsize = figsize)
plt.semilogy(times, sqz)
plt.xlabel(r"$t \times J_\perp$")
plt.ylabel(r"$-10 \log_10 \xi^2_{\mathrm{min}}$")
y_btm = plt.gca().get_ylim()[0]
y_btm_new = 10**np.floor(np.log10(y_btm))
plt.ylim(y_btm_new, 1.3)
plt.tight_layout(pad = 0.1)
plt.savefig(fig_dir + f"sqz_{name_tag}.pdf")
plt.close()

plt.figure(figsize = figsize)
for manifold, shells in manifold_shells.items():
    plt.plot(times, pops[:,shells].sum(axis = 1), label = f"${manifold}$")
plt.axvline(times[sqz.argmin()], color = "gray", linestyle = "--")
plt.xlabel(r"$t \times J_\perp$")
plt.ylabel("population")
plt.legend()
plt.tight_layout(pad = 0.1)
plt.savefig(fig_dir + f"pops_{name_tag}.pdf")
plt.close()
