#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

dim = 6
inv_phi_vals = [ 6, 3 ]

figsize = (4.5,1.75)
fig_dir = "../figures/"

params = { "font.size" : 9,
           "text.usetex" : True }
plt.rcParams.update(params)

##################################################

assert(dim % 2 == 0)
spin = (dim-1)/2
q_vals = np.linspace(-1,1)

def disp(qq, mu, inv_phi_val):
    phi = np.pi / inv_phi_val
    return -2*np.cos(np.pi*qq + mu*phi)
disp = np.vectorize(disp)

def sign(mu):
    return "+" if mu >= 0 else "-"

figure, axes = plt.subplots(1, len(inv_phi_vals),
                            figsize = figsize, sharex = True, sharey = True)

for axis, inv_phi_val, in zip(axes, inv_phi_vals):
    for mu in np.arange(spin,-spin-1,-1):
        axis.plot(q_vals, disp(q_vals,mu,inv_phi_val),
                  label = f"${sign(mu)}{abs(int(2*mu))}/2$")
    axis.set_title(r"$\phi=\pi/" + f"{inv_phi_val}$")
    axis.set_xlabel(r"$q/\pi$")

axes[0].set_ylabel(r"$E_{q\mu}/J$")
axes[-1].legend(loc = "center left", bbox_to_anchor = (1,0.5))

plt.tight_layout(pad = 0.4)
plt.savefig(fig_dir + "sun_dispersion.pdf")
