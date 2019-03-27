#!/usr/bin/env python3

import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.optimize import curve_fit
from itertools import combinations, cycle

from sr87_olc_constants import colors, recoil_energy_Hz

sweep_rabi_frequency = 50 # rabi freqency of single atom with spin 9/2, in Hz
kHz = recoil_energy_Hz / 1000 # conversion from recoil energy to frequency in Hz
ms = 1000

data_dir = "../data/interaction_shifts/"
fig_dir = "../figures/rabi_spectroscopy/"
if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

two_body_shift_file = data_dir + "shift_predictions_1.txt"
shift_file = data_dir + "shift_predictions_total.txt"
atom_numbers = [ 2, 3, 4, 5 ]
expt_depth = 54

default_figsize = np.array([ 3, 2.1 ])
table_figsize = np.array([ 6, 4 ])
sweep_figsize = np.array([ 3.2, 2 ])
params = { "text.usetex" : True,
           "text.latex.preamble" : [ r"\usepackage{amssymb}",
                                     r"\usepackage{physics}",
                                     r"\usepackage{braket}",
                                     r"\usepackage{dsfont}" ] }
plt.rcParams.update(params)
fig_dpi = 600

I = 9/2
N_max = int(2*I+1)
spin_vals = [ -I + ii for ii in range(N_max) ]

def coeff(spins, symmetric):
    spin_mean = np.mean(spins)
    if symmetric:
        w = spin_mean
    else:
        w = np.sqrt(np.mean([ (spin - spin_mean)**2 for spin in spins ]))
    return abs(w) / I * np.sqrt(len(spins))

def rabi_populations(coupling, times, detunings):
    populations = np.zeros((len(detunings),len(times)))
    for dd, detuning in enumerate(detunings):
        D = np.sqrt(detuning**2 + coupling**2)
        if D == 0: continue
        for tt, time in enumerate(times):
            populations[dd,tt] += (coupling/D*np.sin(D*time))**2
    return populations

def populations(N, symmetric, times, detunings, coupling_unit = 1):
    time_vals = len(times)
    detuning_vals = len(detunings)
    spin_combs = list(combinations(spin_vals, N))
    populations = np.zeros((len(detunings),len(times)))
    for spins in spin_combs:
        coupling = coeff(spins, symmetric) * coupling_unit
        populations += rabi_populations(coupling, times, detunings)
    return populations / len(spin_combs)

def asym_spin_populations(N, times):
    time_vals = len(times)
    spin_combs = list(combinations(spin_vals, N))
    spin_populations = np.zeros((N_max,len(times)))
    for spins in spin_combs:
        spin_mean = np.mean(spins)
        spin_std = np.sqrt(np.mean([ (spin - spin_mean)**2 for spin in spins ]))
        coupling = spin_std / I * np.sqrt(len(spins))
        asymmetric_state_populations = rabi_populations(coupling, times, [0])
        for spin in spins:
            spin_fraction = (spin - spin_mean)**2 / spin_std**2 / len(spins)
            spin_populations[int(spin+9/2),:] += ( spin_fraction *
                                                   asymmetric_state_populations[0] )
    return spin_populations / len(spin_combs)


##########################################################################################
# plot rabi signals as a function of time at zero detuning
##########################################################################################

# make an individual time-resolved rabi signal plot
def make_time_plot(plot, N, cycles, samples,
                   xlabel = True, ylabel = True,
                   legend = True, title = True):

    base_times = cycles * np.linspace(0, 1, samples)
    time_factor = I*np.sqrt(N) * np.pi
    times = base_times * time_factor

    plt.subplot(plot)

    if N == 1:
        plt.plot([-1],[-1])
        plt.plot(times/time_factor, populations(N, True, times, [0]).T, zorder = 0)
        sym = "+"
        handles, labels = None, None

    elif N == N_max:
        plt.plot(times/time_factor, populations(N, False, times, [0]).T, zorder = 0)
        sym = "-"
        handles, labels = None, None

    else:
        for symmetric, label in [ [ True, r"$X=+$" ],
                                  [ False, r"$X=-$" ] ]:
            plt.plot(times/time_factor, populations(N, symmetric, times, [0]).T,
                     label = label, zorder = symmetric-1)
            sym = "X"

            plt.legend(loc="best", framealpha = 1)
            handles, labels = plt.gca().get_legend_handles_labels()
            if not legend: plt.gca().legend_.remove()

    plt.axhline(0.5, linewidth = 0.5, color = "gray", zorder = -1)

    plt.xlim(0, cycles)
    plt.ylim(0, 1)

    if title: plt.title(r"$N={}$".format(N))
    if xlabel: plt.xlabel(r"$\tau_N/\pi$", zorder = 1)
    if ylabel: plt.ylabel(r"$\Braket{{\mathcal{{P}}_{{N{}}}}}$".format(sym), zorder = 1)
    else: plt.gca().set_yticklabels([])

    return handles, labels

if "time" in sys.argv:
    print("time plots")

    samples = 500
    cycles = 5

    if "table" in sys.argv: # make table of time-series plots for chosen atom numbers
        plt.figure("table", figsize = table_figsize)
        table_grid = gridspec.GridSpec(len(atom_numbers) // 2, 2)
        for N in atom_numbers:
            aa = atom_numbers.index(N)
            plot = table_grid[aa]
            xlabel = (aa == len(atom_numbers)-1 or aa == len(atom_numbers)-2)
            ylabel = not (aa % 2) # true for odd numbered plots (i.e. those on the left)
            handles, labels = make_time_plot(plot, N, cycles, samples,
                                             xlabel, ylabel, legend = False)
        plt.gcf().legend(handles, labels, ncol = len(handles),
                         loc = "center", bbox_to_anchor = (0.5,0.97))
        plt.tight_layout(rect = (0,0,1,0.96))
        plt.savefig(fig_dir + "time_table.pdf")

    if "only" not in sys.argv:
        for N in range(1,N_max+1):
            print(N)
            plt.figure(figsize = default_figsize)
            plot = gridspec.GridSpec(1,1)
            make_time_plot(plot[0], N, cycles, samples)
            plt.tight_layout()
            plt.savefig(fig_dir + "time_N{}.pdf".format(N))

    plt.close("all")

##########################################################################################
# plot rabi signals as a function of time at zero detuning
##########################################################################################

if "detuning" in sys.argv:
    print("detuning plots")

    samples = 1000
    time = np.pi
    max_detuning = 5

    detunings = max_detuning * np.linspace(-1, 1, samples)
    for N in range(1,N_max+1):
        print(N)

        plt.figure(figsize = default_figsize)
        if N == 1:
            plt.plot([-1],[-1])
            plt.plot(detunings, populations(N, True, [time], detunings), zorder = 0)

        elif N == N_max:
            plt.plot(detunings, populations(N, False, [time], detunings), zorder = 0)

        else:
            for symmetric, label in [ [ True, "$+$" ],
                                      [ False, "$-$"] ]:
                plt.plot(detunings, populations(N, symmetric, [time], detunings),
                         label = label, zorder = symmetric-1)
                plt.legend(loc="best", framealpha = 1)

        plt.xlabel(r"$\delta/\Omega_I$", zorder = 1)
        plt.ylabel(r"Population", zorder = 1)

        plt.xlim(-max_detuning, max_detuning)
        plt.ylim(0, 1)

        plt.tight_layout()
        plt.gca().set_rasterization_zorder(1)
        plt.savefig(fig_dir + "detuning_N{}.pdf".format(N),
                    rasterized = True, dpi = fig_dpi)

    plt.close("all")

##########################################################################################
# plot 2-D (time and detuning) rabi signals
##########################################################################################

if "2D" in sys.argv:
    print("2D plots")

    samples = 200
    cycles = 4
    max_detuning = 1

    base_times = cycles * np.linspace(0, 1, samples)
    detunings = max_detuning * np.linspace(-1, 1, samples)
    for symmetric, tag in [ [ False, "-" ],
                            [ True, "+" ] ]:
        if symmetric: print("symmetric")
        else: print("asymmetric")
        for N in range(1,N_max+1):
            if (N == 1 and symmetric == False) or N == 10 and (symmetric == True): continue
            print(N)
            time_factor = I*np.sqrt(N) * np.pi
            times = base_times * time_factor

            signal = populations(N, symmetric, times, detunings)

            plt.figure(figsize = default_figsize)
            plt.pcolormesh(times/time_factor, detunings, signal, zorder = 0)

            plt.xlabel(r"$\tau_N/\pi$", zorder = 1)
            plt.ylabel(r"$\delta/\Omega_I$", zorder = 1)

            plt.yticks(range(-max_detuning, max_detuning+1))

            plt.colorbar(label="Population", ticks=[])
            plt.clim(0,1)

            plt.tight_layout()
            plt.gca().set_rasterization_zorder(1)
            plt.savefig(fig_dir + "2D_N{}{}.pdf".format(N,tag),
                        rasterized = True, dpi = fig_dpi)

    plt.close("all")

##########################################################################################
# plot generic 2-D rabi signal
##########################################################################################

if "generic" in sys.argv:
    print("generic rabi signal")

    samples = 1000
    cycles = 3
    max_detuning = 3

    times = cycles * np.linspace(0, 2*np.pi, samples)
    detunings = max_detuning * np.linspace(-1, 1, samples)
    signal = rabi_populations(1, times, detunings)

    plt.figure(figsize = default_figsize)
    plt.pcolormesh(times/(2*np.pi), detunings, signal, zorder = 0)

    plt.xlabel(r"$2\pi t/\Omega$", zorder = 1)
    plt.ylabel(r"$\delta/\Omega$", zorder = 1)

    plt.xticks(range(int(cycles+1)))
    plt.yticks(range(-max_detuning, max_detuning+1))

    plt.colorbar(label="Population", ticks=[])
    plt.clim(0,1)

    plt.tight_layout()
    plt.gca().set_rasterization_zorder(1)
    plt.savefig(fig_dir + "2D_generic.pdf", rasterized = True, dpi = fig_dpi)
    plt.close()

##########################################################################################
# plot signal from a detuning sweep across all resonances
##########################################################################################

def lorentzian(x,height,location,scale):
    return height / ( 1 + ( (x-location) / scale )**2 )

# label resonance peaks
def label_peaks(axis, peaks, expt = False):
    HR = axis.get_xlim()[1] - axis.get_xlim()[0]
    VR = axis.get_ylim()[1] - axis.get_ylim()[0]
    if expt:
        axis.text(peaks[0][0]+0.03*HR, peaks[0][1]-0.13*VR, r"$1$")
        axis.text(peaks[1][0]-0.05*HR, peaks[1][1]+0.05*VR, r"$2^-$")
        axis.text(peaks[2][0]-0.05*HR, peaks[2][1]+0.04*VR, r"$3^-$")
        axis.text(peaks[3][0]-0.01*HR, peaks[3][1]+0.06*VR, r"$4^-$")
        axis.text(peaks[4][0]-0.01*HR, peaks[4][1]+0.07*VR, r"$5^-$")
        axis.text(peaks[5][0]-0.02*HR, peaks[5][1]+0.06*VR, r"$2^+$")
        axis.text(peaks[6][0]-0.02*HR, peaks[6][1]+0.07*VR, r"$3^+$")
        axis.text(peaks[7][0]-0.02*HR, peaks[7][1]+0.08*VR, r"$4^+$")
        axis.text(peaks[8][0]-0.02*HR, peaks[8][1]+0.07*VR, r"$5^+$")
    else:
        axis.text(peaks[0][0]+0.02*HR, peaks[0][1]-0.08*VR, r"$1$")
        axis.text(peaks[1][0]-0.05*HR, peaks[1][1]+0.03*VR, r"$2^-$")
        axis.text(peaks[2][0]-0.05*HR, peaks[2][1]+0.07*VR, r"$3^-$")
        axis.text(peaks[3][0]-0.02*HR, peaks[3][1]+0.08*VR, r"$4^-$")
        axis.text(peaks[4][0]-0.02*HR, peaks[4][1]+0.05*VR, r"$5^-$")
        axis.text(peaks[5][0]-0.02*HR, peaks[5][1]+0.06*VR, r"$2^+$")
        axis.text(peaks[6][0]-0.02*HR, peaks[6][1]+0.06*VR, r"$3^+$")
        axis.text(peaks[7][0]-0.02*HR, peaks[7][1]+0.06*VR, r"$4^+$")
        axis.text(peaks[8][0]-0.02*HR, peaks[8][1]+0.06*VR, r"$5^+$")

if "sweep" in sys.argv:
    print("detuning sweep signal (experiment)")
    with open(data_dir + "spectrum_V{}.txt".format(expt_depth), 'r+') as f:
        data =  json.load(f)

    # keep track of resonance peaks for each state
    peaks = [ None for key in data.keys() ]

    detuning_min = 0
    detuning_max = 0
    plt.figure(figsize = sweep_figsize)
    for key in data.keys():
        if len(key) == 3:
            atom_number = int(key[0])
            peak_index = atom_number - 1
            if key[-1] == "s": peak_index += (len(data.keys())-1)//2
        else:
            atom_number = 1
            peak_index = 0

        detunings, signal = data[key]
        detuning_min = min(min(detunings), detuning_min)
        detuning_max = max(max(detunings), detuning_max)

        height = np.max(signal)
        location = np.mean(detunings)
        scale = 1
        p_0 = [ height, location, scale ]
        height, location, scale = curve_fit(lorentzian, detunings, signal,
                                            [ height, location, scale ])[0]
        curve = [ lorentzian(x, height, location, scale) for x in data[key][0] ]

        plt.plot(data[key][0], curve, color = colors[atom_number-1], zorder = 1)
        plt.plot(detunings, signal, "k.", markersize = 2, zorder = 1)

        peaks[peak_index] = (location, height)

    plt.xlabel(r"Detuning $\Delta$ ($2\pi$ kHz)")
    plt.ylabel(r"${}^3P_0$ population")

    xtick_min = int(np.floor(detuning_min / 2) * 2)
    xtick_max = int(np.ceil(detuning_max / 2) * 2)
    plt.xticks(range(xtick_min, xtick_max+1, 2))

    plt.ylim(0, plt.gca().get_ylim()[1])
    plt.yticks([0])

    label_peaks(plt.gca(), peaks, expt = True)
    plt.tight_layout()
    plt.savefig(fig_dir + "sweep_expt_V{}.pdf".format(expt_depth))

    for txt in plt.gca().texts: txt.set_visible(False) # hide old text
    base_expt_lines = len(plt.gca().lines)

    # plot experimental data with reference lines from predictions
    for data_file, tag in [ (two_body_shift_file, 1), (shift_file, 3) ]:
        shift_data = np.loadtxt(data_file)
        depths = ( np.prod(shift_data[:,:3], 1) )**(1/3)
        shifts = shift_data[:,3:]

        for dd, depth in enumerate(depths):
            if abs(depth - expt_depth) >= 1: continue
            for ss, ll in [ (1,"-"), (2,"+") ]:
                shifts_ss = shifts[dd,ss::3] - shifts[dd,0::3]
                for aa, atom_number in enumerate(atom_numbers):
                    plt.axvline(shifts_ss[aa] * kHz,
                                linestyle = ":", color = colors[aa+1], zorder = 0)
                    y_min, y_max = plt.gca().get_ylim()
                    text_y = y_max + (y_max - y_min) * 0.03
                    plt.gca().text(shifts_ss[aa] * kHz, text_y,
                                   r"${}^{}$".format(atom_number,ll),
                                   color = colors[aa+1])

        plt.tight_layout()
        plt.savefig(fig_dir + "sweep_expt_V{}_refs_{}.pdf".format(expt_depth, tag))

        for txt in plt.gca().texts: txt.set_visible(False) # hide old text
        while len(plt.gca().lines) > base_expt_lines:
            plt.gca().lines[-1].remove()
        plt.savefig(fig_dir + "sweep_expt_V{}_refs_{}_blank.pdf".format(expt_depth, tag))


    print("detuning sweep signal (theory)")
    unit_rabi_frequency = sweep_rabi_frequency / recoil_energy_Hz

    samples = 1000
    time = np.pi / unit_rabi_frequency
    detuning_pad = 20 * unit_rabi_frequency

    shift_data = np.loadtxt(shift_file)
    depths = ( np.prod(shift_data[:,:3], 1) )**(1/3)
    shifts = shift_data[:,3:]

    with open(shift_file,"r") as f:
        header_items = f.readline().split()
        atom_numbers = list(set([ int(item[0]) for item in header_items[4:] ]))

    for dd, depth in enumerate(depths):
        print("({}/{}) {}".format(dd,len(depths),depth))

        plt.figure(figsize = sweep_figsize)

        asym_resonances = shifts[dd,1::3] - shifts[dd,0::3]
        sym_resonances = shifts[dd,2::3] - shifts[dd,0::3]
        resonances = np.vstack([asym_resonances,sym_resonances])
        resonance_min, resonance_max = resonances.min(), resonances.max()

        detunings = np.linspace(resonance_min - detuning_pad,
                                resonance_max + detuning_pad, samples)

        # keep track of resonance peaks for each state
        peaks = []

        # total signal (i.e. population of excited electronic state)
        signal = populations(1, 1, [time], detunings, unit_rabi_frequency)
        peaks.append((0,max(signal)))

        for ss in range(2):
            for aa, atom_number in enumerate(atom_numbers):
                resonance = resonances[ss,aa]
                N_signal = populations(atom_number, ss, [time],
                                       detunings - resonance,
                                       unit_rabi_frequency) / atom_number
                signal += N_signal
                peaks.append((resonance*kHz,max(N_signal)))

        plt.plot(detunings * kHz, signal, "k")
        plt.xlabel(r"Detuning $\Delta$ ($2\pi$ kHz)")
        plt.ylabel(r"${}^3P_0$ population")

        plt.xlim(detunings[0] * kHz, detunings[-1] * kHz)
        xtick_min = int(np.ceil(detunings[0] * kHz / 2) * 2)
        xtick_max = int(np.floor(detunings[-1] * kHz / 2) * 2)
        plt.xticks(range(xtick_min, xtick_max+1, 2))

        plt.ylim(0, plt.gca().get_ylim()[1])
        plt.yticks([0])

        label_peaks(plt.gca(), peaks, expt = False)
        plt.tight_layout()

        int_depth = int(round(depths[dd]))
        fig_name = "sweep_V{}_O{}.pdf".format(int_depth, sweep_rabi_frequency)
        plt.savefig(fig_dir + fig_name)

    plt.close("all")
