#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt

from mathieu_methods import mathieu_solution
from overlap_methods import pair_overlap_1D
from interaction_num_methods import renormalized_coupling

from sr87_olc_constants import g_int_LU

figsize = (4,3)
site_number = 100
bands = 13
depths = np.linspace(30, 80, 100)

params = { "text.usetex" : True,
           "text.latex.preamble" : r"\usepackage{amssymb}" }
plt.rcParams.update(params)

params = np.zeros((len(depths),4))
for dd in range(len(depths)):
    print(depths[dd])
    # determine spectral gap and ground-state overlap intergral
    momenta, fourier_vecs, energies = mathieu_solution(depths[dd], bands, site_number)
    band_energies = np.mean(energies,0)
    spectral_gap = band_energies[1] - band_energies[0]
    overlap = pair_overlap_1D(momenta, fourier_vecs)**3

    # compute effective coupling constants for given lattice depths
    couplings = np.array([ renormalized_coupling(g_int_LU[ii], np.ones(3) * depths[dd])
                           for ii in range(4) ])

    # compute perturbative parameters
    params[dd,:] = overlap * couplings / spectral_gap

plt.figure(figsize = figsize)

plt.plot(depths, params[:,0], label = "gg")
plt.plot(depths, params[:,1], label = r"eg$_-$")
plt.plot(depths, params[:,2], label = r"eg$_+$")
plt.plot(depths, params[:,3], label = "ee")

plt.xlim(depths[0], depths[-1])
plt.ylim(0, plt.gca().get_ylim()[1])

plt.xlabel(r"Lattice depth $\mathcal{U}$ ($E_R$)")
plt.ylabel(r"$K G_X/\Delta$")

plt.legend(loc = "best", ncol = 2)
plt.gca().tick_params(right = True)

plt.tight_layout()
plt.savefig("../figures/perturbative_params.pdf")

