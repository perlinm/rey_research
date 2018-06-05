#!/usr/bin/env python3

import os, sys
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sympy.physics.quantum.cg import CG as clebsch
from itertools import combinations

fig_dir = "../figures/rabi_spectroscopy/"
if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

atom_numbers = [ 2, 3, 4, 5 ]

single_figsize = [ 3, 2 ]
table_figsize = [ 6, 4 ]
params = { "text.usetex" : True,
           "text.latex.preamble" : [ r"\usepackage{amssymb}",
                                     r"\usepackage{dsfont}" ] }
plt.rcParams.update(params)

I = sp.S(9)/2
m_vals = [ -I + ii for ii in range(2*I+1) ]

g = {}
for m in m_vals: g[m] = clebsch(I,m,1,0,I,m).doit()
for m in m_vals: g[m] /= g[m_vals[-1]]

def coeff(ms, even_state_coupling):
    g_mean = sum([ g[m] for m in ms ]) / len(ms)
    if even_state_coupling:
        return g_mean * sp.sqrt(len(ms))
    else:
        return sp.sqrt( sum( (g[m] - g_mean)**2 for m in ms ) )

def unique_vals(vals):
    return sorted(list(set(vals)))

def counts(vals):
    return [ vals.count(v) for v in unique_vals(vals) ]

def make_plot(plot, even_vals, odd_vals,
              xlabel = True, ylabel = True, legend = True, title = True):

    plt.subplot(plot)
    plt.plot(unique_vals(even_vals), counts(even_vals), "+",
             label = r"$X=+$", zorder = 1)
    plt.plot(unique_vals(odd_vals), counts(odd_vals), "x",
             label = r"$X=-$", zorder = 0)

    plt.ylim(0,plt.gca().get_ylim()[1])
    plt.xlim(-0.03,1.03)
    plt.xticks([0, 1/3, 2/3, 1], ["0", "1/3", "2/3", "1"])

    plt.legend(loc="best").get_frame().set_alpha(1)
    handles, labels = plt.gca().get_legend_handles_labels()
    if not legend: plt.gca().legend_.remove()

    if title: plt.title(r"$N={}$".format(N))
    if ylabel: plt.ylabel("Multiplicity")
    if xlabel: plt.xlabel(r"$\left|\omega_{\mathcal{N},X}/I\right|$")
    else: plt.gca().set_xticklabels([])

    return handles, labels


plt.figure("table", figsize = table_figsize)
table_grid = gridspec.GridSpec(len(atom_numbers) // 2, 2)

for N in range(2,2*I+2):
    comb = list(combinations(m_vals, N))
    even_vals = [ abs(coeff(ms,True)) / sp.sqrt(N) for ms in comb ]
    odd_vals = [ coeff(ms,False) / sp.sqrt(N) for ms in comb ]

    # individual plots
    if "table" not in sys.argv:
        plt.figure(figsize = single_figsize)
        plot = gridspec.GridSpec(1,1)
        make_plot(plot[0], even_vals, odd_vals)
        plt.tight_layout()
        plt.savefig(fig_dir+"coefficients_N{}.pdf".format(N))

    # plots in table
    if N in atom_numbers:
        aa = atom_numbers.index(N)
        xlabel = (aa == len(atom_numbers)-1 or aa == len(atom_numbers)-2)
        ylabel = not (aa % 2) # true for odd numbered plots (i.e. those on the left)
        handles, labels = make_plot(table_grid[aa], even_vals, odd_vals,
                                    xlabel, ylabel, legend = False)

plt.figure("table")
plt.gcf().legend(handles, labels, ncol = len(handles),
                 loc = "center", bbox_to_anchor = (0.5,0.96))
plt.tight_layout(rect = (0,0,1,0.95))
plt.savefig(fig_dir+"coefficients_table.pdf")
