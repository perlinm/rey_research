#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(linewidth = 200)

figsize = (3,2.2)
params = { "text.usetex" : True,
           "font.size" : 8 }
plt.rcParams.update(params)

savename = "../figures/squeezing/sqz_dec_scaling.pdf"
basename = r"../data/squeezing/sqz_D_exact_logN2_{}_d{}.txt"

OAT, TAT, TNT = "OAT", "TAT", "TNT"
methods = [ OAT, TAT, TNT ]
colors = { OAT : "#1E4977", TAT : "#E15759", TNT : "#FFAE4B" }
markers = { OAT : "o", TAT : "s", TNT : "D" }

dec_rates = np.logspace(-3,1,13)[:-1]

sqz_min = {}
for method in methods:
    sqz_min[method] = np.zeros(len(dec_rates))
    for dec_idx in range(len(dec_rates)):
        fname = basename.format(method, f"{dec_idx:02d}")
        sqz_method_dec = np.loadtxt(fname, delimiter = ",", unpack = True)[1]
        sqz_min[method][dec_idx] = min(sqz_method_dec)

plt.figure(figsize = figsize)

for method in methods:
    plt.loglog(dec_rates[3:], sqz_min[method][3:],
               markers[method], color = colors[method], label = method)

plt.ylabel(r"$\xi^2_{\mathrm{min}}$")
plt.xlabel(r"$\gamma_0/\chi$")
plt.legend(loc = "best")
plt.tight_layout()
plt.savefig(savename)
