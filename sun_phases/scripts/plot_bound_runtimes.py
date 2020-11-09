#!/usr/bin/env python3

import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt

serial = "serial" in sys.argv

data_dir = "../data/error_scale/"
fig_dir = "../figures/error_scale/"

if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

params = { "font.size" : 10,
           "text.usetex" : True }
plt.rcParams.update(params)

figsize = (3,2)
fit_dim_min = 100

##################################################

files = glob.glob(data_dir + "times_*" + ("serial" if serial else "") + ".txt")

def max_dim(file):
    pieces = file.replace(".txt","").split("/")[-1].split("_")
    return pieces[-1].split("-")[-1]
files = sorted(files, key = max_dim)

data = np.vstack([ np.loadtxt(file) for file in files ])
sizes = data[:,0]
times = data[:,1]

plt.figure(figsize = figsize)
plt.loglog(sizes, times, "k.", label = "data")

size_lims = plt.gca().get_xlim()
time_lims = plt.gca().get_ylim()

fit_idx = sizes >= fit_dim_min
fit = np.polyfit(np.log(sizes[fit_idx]), np.log(times[fit_idx]), deg = 1)
plt.loglog(size_lims, np.exp(fit[0]*np.log(size_lims) + fit[1]), "k--", label = "fit")
plt.gca().set_xlim(size_lims)
plt.gca().set_ylim(time_lims)

plt.xlabel(r"qudit dimension $d$")
plt.ylabel(r"seconds to compute $\mathcal{S}_V$")
plt.legend(loc = "best")
plt.tight_layout()
plt.savefig(fig_dir + "error_times.pdf")
