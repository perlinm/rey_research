#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import random

random.seed(0)

figsize = (4,3)
font = { "family" : "serif",
         "sans-serif" : "Computer Modern Sans serif" }
plt.rc("font",**font)
params = { "text.usetex" : True,
           "font.size" : 8 }

N_vals = [ 100, 1000, 10000 ]
filling = 3/4
phi = np.pi / 50
samples = 10000

def splitting(n, sites, phi = phi):
    n_x = n / np.sqrt(sites)
    n_y = n % np.sqrt(sites)
    q_x = 2*np.pi * n_x / np.sqrt(sites)
    q_y = 2*np.pi * n_y / np.sqrt(sites)
    return np.cos(q_x + phi) + np.cos(q_y + phi) - np.cos(q_x) - np.cos(q_y)

for N in N_vals:
    sites = int(round(N/filling))
    means = np.zeros(samples)
    stds = np.zeros(samples)
    for sample in range(samples):
        splittings = [ splitting(n,sites) for n in random.sample(range(sites), N) ]
        means[sample] = np.mean(splittings)
        stds[sample] = np.std(splittings)

    rms = np.sqrt(np.mean((means/stds)**2))
    print(N, rms)
    plt.figure(figsize = figsize)
    plt.hist(abs(means/stds))
    plt.xlabel(r"$\overline{B}/\widetilde{B}$")
    plt.ylabel("Samples")
    plt.title(f"$N={N}$")

    plt.tight_layout()

plt.show()

