#!/usr/bin/env python3

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator # for integer axis labels

from sr87_olc_constants import k_clock_LU, spins
from lattice_methods import shift_q, lattice_q, qn_energy, qns_energy, \
    qn_state_z, jn_state_z, laser_overlap, \
    numerical_inner_product, numerical_laser_overlap
from evolution_methods import select_populations, effective_driving_populations

fig_size = (4,3) # default figure size
params = { "text.usetex" : True,
           "text.latex.preamble" : [ r"\usepackage{physics}",
                                     r"\usepackage{amssymb}" ] }
plt.rcParams.update(params)

# label and linestyle associated with a given spin
def s_label(s, sign = False):
    assert(s in spins)
    if s == -1: return r"{}^1S_0" if not sign else "-"
    if s == 1:  return r"{}^3P_0" if not sign else "+"
def s_style(s):
    assert(s in spins)
    if s == -1: return "--"
    if s == 1:  return "-"

# label associated with a given bands and spin
def ns_label(n, s, sign = False):
    return "${},~{}$".format(n,s_label(s,sign))

def plot_labels(plot_type, bands, sign = False):
    assert(plot_type in ["all","band","spin"])
    if plot_type == "all":
        return [ ns_label(n,s,sign) for n in range(bands) for s in spins ]
    elif plot_type == "band":
        return [ "${}$".format(n) for n in range(bands) ]
    else: return [ "${}$".format(s_label(s,sign)) for s in spins ]

# colors to use in a plot
plot_colors = [ item["color"] for item in list(plt.rcParams["axes.prop_cycle"]) ]


##########################################################################################
# band structure plots
##########################################################################################

def qn_band_plot(momenta, energies, bands, detuning_frequency = 0, fig_size = fig_size):
    fig = plt.figure(figsize = fig_size)
    if momenta[-1] != -momenta[0]:
        plot_momenta = list(momenta) + [-momenta[0]]
    else:
        plot_momenta = momenta
    for n in range(len(energies[0,:])):
        energy_vals = [ qn_energy(q, n, energies, detuning_frequency)
                        for q in plot_momenta ]
        plt.plot(plot_momenta, energy_vals)
        if n == bands - 1: ylims = plt.gca().get_ylim()
    plt.xlim(-1,1)
    try: plt.ylim(ylims[0],ylims[-1])
    except: None
    plt.xlabel("$q/k_L$")
    plt.ylabel("$E_{qn}/E_R$")
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    return fig

def qns_band_plot(momenta, energies, bands, static_detuning = 0,
                  detuning_frequency = 0, fig_size = fig_size):
    bands = min(bands, len(energies[0,:]))
    fig = plt.figure(figsize = fig_size)
    if momenta[-1] != -momenta[0]:
        plot_momenta = list(momenta) + [-momenta[0]]
    else:
        plot_momenta = momenta
    for n in range(len(energies[0,:])):
        for s in spins:
            energy_vals = [ qns_energy(q, n, s, energies, detuning_frequency)
                            - s * static_detuning / 2
                            for q in plot_momenta ]
            plt.plot(plot_momenta, energy_vals,
                     color = plot_colors[n], linestyle = s_style(s))
        if n == bands - 1: ylims = plt.gca().get_ylim()
    plt.xlim(-1,1)
    try: plt.ylim(ylims[0],ylims[-1])
    except: None
    plt.xlabel("$q/k_L$")
    plt.ylabel("$E_{qns}/E_R$")

    plt.plot([0], [plt.gca().get_ylim()[0]-1], "k", linestyle = s_style(1),
             label = "${}$".format(s_label(1,True)))
    plt.plot([0], [plt.gca().get_ylim()[0]-1], "k", linestyle = s_style(-1),
             label = "${}$".format(s_label(-1,True)))
    plt.legend(loc="upper right", framealpha = 1)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    return fig

##########################################################################################
# wavefunction plots
##########################################################################################

def qn_state_plot(q, n, momenta, fourier_vecs,
                  z_lim = 1, points = 1000, fig_size = fig_size):
    fig = plt.figure(figsize = fig_size)
    z_vals = np.linspace(-z_lim, z_lim, points)
    func_vals = np.array([ qn_state_z(q,n,np.pi*z,momenta,fourier_vecs) for z in z_vals ])
    func_vals /= max(abs(func_vals))
    plt.plot(z_vals, abs(func_vals), label = r"$\abs{\phi_{qn}}$")
    plt.plot(z_vals, np.angle(func_vals)/np.pi, label = r"arg$(\phi_{qn})/\pi$")
    plt.xlim(z_vals[0], z_vals[-1])
    plt.xlabel(r"$z k_L/\pi$")
    plt.title(r"$q={},~n={}$".format(q,n))
    plt.legend(loc="best")
    plt.tight_layout()
    return fig

def jn_state_plot(j, n, momenta, fourier_vecs,
                  z_lim = 3, points = 1000, fig_size = fig_size):
    fig = plt.figure(figsize = fig_size)
    z_vals = np.linspace(j - z_lim, j + z_lim, points)
    func_vals = np.array([ jn_state_z(j,n,np.pi*z,momenta,fourier_vecs) for z in z_vals ])
    plt.plot(z_vals, np.real(func_vals), label = r"Re$(\phi_{jn})$")
    plt.plot(z_vals, np.imag(func_vals), label = r"Im$(\phi_{jn})$")
    plt.xlim(z_vals[0], z_vals[-1])
    plt.xlabel(r"$z k_L/\pi$")
    plt.title(r"$j={},~n={}$".format(j,n))
    plt.legend(loc="best")
    plt.tight_layout()
    return fig

def laser_overlap_scan(n, m, s, momenta, fourier_vecs, fig_size = fig_size):
    fig = plt.figure(figsize = fig_size)
    overlaps = [ laser_overlap(q,s,n,m,momenta,fourier_vecs) for q in momenta ]
    plt.plot(momenta, overlaps, ".")
    plt.xlim(-1,1)
    plt.xlabel(r"$q/k_L$")
    plt.title(r"$\Omega^{{q,{}}}_{{{},{}}}$".format(s_label(s,True),n,m))
    plt.tight_layout()
    return fig


##########################################################################################
# sanity checks: scans over numerical inner products
##########################################################################################

def numerical_inner_product_scan(n, m, momenta, fourier_vecs, fig_size = fig_size):
    fig = plt.figure(figsize = fig_size)
    inner_products = [ abs(numerical_inner_product(0,n,q,m,momenta, fourier_vecs))
                       for q in momenta ]
    plt.plot(momenta, inner_products, ".")
    plt.axhline(max(inner_products), color = "grey", linewidth = 0.5)
    plt.axvline(0, color = "grey", linewidth = 0.5)
    plt.xlim(-1,1)
    plt.xlabel(r"$q/k_L$")
    plt.ylabel(r"$\left<0,{}|q,{}\right>$".format(n,m))
    plt.gca().set_yscale("log")
    plt.tight_layout()
    return fig

def numerical_laser_overlap_scan(n, m, s, momenta, fourier_vecs, fig_size = fig_size):
    fig = plt.figure(figsize = fig_size)
    overlaps = [ abs(numerical_laser_overlap(0,n,s,g,m,momenta,fourier_vecs))
                 for g in momenta ]
    plt.plot(momenta, overlaps, ".")
    plt.xlim(-1,1)
    plt.xlabel(r"$q/k_L$")
    plt.title(r"$\Omega^{{0,{},{}}}_{{q,{}}}$".format(n,s_label(s,True),m))
    plt.gca().set_yscale("log")

    q_ref = shift_q(lattice_q(s*k_clock_LU, momenta))
    plt.axvline(q_ref, color = "grey", linewidth = 0.5)
    plt.tight_layout()
    return fig


##########################################################################################
# time evolution plots
##########################################################################################

# make time evolution plot for a static Hamiltonian
def static_population_plot(initial_state, hamiltonian, max_time, bands,
                           time_steps = 1000, plot_types = ["all"], fig_size = fig_size):
    bands = min(bands, len(initial_state)//2)
    times = np.linspace(0, max_time, time_steps)

    def propagator(time): return scipy.linalg.expm(-1j * time * hamiltonian)
    all_populations = np.array([ abs((propagator(time) @ initial_state)[:2*bands])**2
                                 for time in times ]).T

    figs = []
    for plot_type in plot_types:
        figs.append(plt.figure("static-"+plot_type, figsize = fig_size))
        populations = select_populations(all_populations, plot_type)
        labels = plot_labels(plot_type, bands, sign = True)
        for ii in range(len(labels)):
            plt.plot(times, populations[ii,:], label = labels[ii])
        plt.xlim(times[0], times[-1])
        plt.xlabel("$t E_R$")
        plt.title("State populations")
        plt.legend(loc="best")
        plt.tight_layout()
    return figs

# make time evolution plot for a time dependent Hamiltonian
def population_plot(q, n, s, momenta, energies, fourier_vecs,
                    rabi_coupling, mod_type, mod_freq, mod_index, detuning_mean,
                    max_time, time_scale, time_text, bands,
                    method = effective_driving_populations,
                    time_steps = 1000, plot_types = ["all"], fig_size = fig_size,
                    plot_title = None, plot_ylabel = None):
    times = np.linspace(0, max_time * time_scale, time_steps)
    all_populations = method(q, n, s, momenta, energies, fourier_vecs,
                             rabi_coupling, mod_type, mod_freq, mod_index,
                             detuning_mean, bands, times)
    figs = []
    for plot_type in plot_types:
        figs.append(plt.figure(plot_type, figsize = fig_size))
        populations = select_populations(all_populations, plot_type)
        labels = plot_labels(plot_type, bands, sign = True)
        for ii in range(len(labels)):
            plt.plot(times/time_scale, populations[ii,:], label = labels[ii])
        plt.xlim(0,max_time)
        plt.xlabel(time_text)
        if plot_title != None: plt.title(plot_title)
        if plot_ylabel != None: plt.ylabel(plot_ylabel)
        plt.legend(loc="upper right", framealpha = 1)
        plt.tight_layout()
    return figs

