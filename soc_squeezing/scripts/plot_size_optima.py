#!/usr/bin/env python3

# FILE CONTENTS: plots optimal squeezing times and values as a function of system size

import sys
import pandas as pd
import matplotlib.pyplot as plt

show = "show" in sys.argv
save = "save" in sys.argv
assert(show or save)

figsize = (4,3)
params = { "text.usetex" : True }
plt.rcParams.update(params)

data_dir = "../data/"
fig_dir = "../figures/"
sqz_fname = data_dir + "sqz_{}.txt"
time_fname = data_dir + "time_{}.txt"

methods = [ "OAT", "TVF", "TAT" ]
OAT, TVF, TAT = methods

def pd_read_csv(fname):
    return pd.read_csv(fname, comment = "#", squeeze = True, header = None, index_col = 0)

sqz_vals = {}
time_vals = {}
for method in methods:
    sqz_vals[method] = pd_read_csv(sqz_fname.format(method))
    time_vals[method] = pd_read_csv(time_fname.format(method))

N_min, N_max = time_vals[OAT].index[0], time_vals[OAT].index[-1]

plt.figure(figsize = figsize)
for method in methods:
    plt.semilogx(sqz_vals[method], label = method)
plt.xlim(N_min, N_max)
plt.xlabel(r"$N$")
plt.ylabel(r"$-10\log_{10}(\xi_{\mathrm{opt}}^2)$")
plt.legend(loc = "best")
plt.tight_layout()
if save: plt.savefig(fig_dir + "optimal_squeezing.pdf")

plt.figure(figsize = figsize)
for method in methods:
    plt.loglog(time_vals[method], label = method)
plt.xlim(N_min, N_max)
plt.xlabel(r"$N$")
plt.ylabel(r"$\chi t_{\mathrm{opt}}$")
plt.legend(loc = "best")
plt.tight_layout()
if save: plt.savefig(fig_dir + "optimal_times.pdf")

if show: plt.show()
