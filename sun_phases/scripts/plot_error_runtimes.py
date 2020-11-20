#!/usr/bin/env python3

import os, glob
import numpy as np
import matplotlib.pyplot as plt

data_dir = "../data/qudit_errors/"
fig_dir = "../figures/qudit_errors/"

if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

params = { "font.size" : 10,
           "text.usetex" : True }
plt.rcParams.update(params)

figsize = (3,2)

##################################################

def max_dim(file):
    pieces = file.replace(".txt","").split("/")[-1].split("_")
    return pieces[-1].split("-")[-1]
def get_data(tag):
    files = sorted(glob.glob(data_dir + f"times_{tag}_d*.txt"), key = max_dim)
    data = np.vstack([ np.loadtxt(file) for file in files ])
    return data[:,0], data[:,1]

plt.figure(figsize = figsize)

for tag, label, marker in [ ( "CB", r"$\mathcal{S}_V$", "ro" ),
                            ( "QB", r"$\epsilon_V$", "k." ),
                            ( "RE", r"$\mathcal{E}_V(\rho)$", "b." ) ]:
    sizes, times = get_data(tag)
    plot = plt.loglog(sizes, times, marker, label = label)
    color = plot[0].get_color()

    size_lims = plt.gca().get_xlim()
    time_lims = plt.gca().get_ylim()
    fit_idx = sizes >= sizes[-1]//2
    fit = np.polyfit(np.log(sizes[fit_idx]), np.log(times[fit_idx]), deg = 1)
    plt.loglog(size_lims, np.exp(fit[0]*np.log(size_lims) + fit[1]), "--", color = color)
    plt.gca().set_xlim(size_lims)
    plt.gca().set_ylim(time_lims)

plt.xlabel(r"qudit dimension $(d)$")
plt.ylabel(r"seconds $(t)$")
plt.legend(loc = "best")
plt.tight_layout()
plt.savefig(fig_dir + "qudit_times.pdf")
