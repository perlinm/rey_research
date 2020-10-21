#!/usr/bin/env python3

import os, sys, glob
import numpy as np
import scipy.signal, scipy.linalg
import matplotlib.pyplot as plt
import itertools as it

from multilevel_methods import drive_op, drive_scale

spin_dim = int(sys.argv[1])
spin_num = int(sys.argv[2])
init_state_str = sys.argv[3]

assert( init_state_str in [ "X-L", "XX-L", "P-Z" ] )

figsize = (4,3)
figsize_ext = (5,3)
figsize_double = (5,4)

# plotting parameters
log10_field_vals_summary = np.arange(-2,1.01,1)
plot_peaks = 50
freq_num = 1000
freq_scale = 2

data_dir = "../data/oscillations/"
fig_dir = "../figures/oscillations/"
sys_tag = f"n{spin_dim}_N{spin_num}_{init_state_str}"

if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

params = { "font.size" : 10,
           "text.usetex" : True }
plt.rcParams.update(params)

def get_interp_vals(xs, ys, kind = "cubic"):
    interp = scipy.interpolate.interp1d(xs, ys, kind = kind)
    new_xs = np.linspace(xs[0], xs[-1], xs.size)
    new_ys = interp(new_xs)
    return new_xs, new_ys

# convert upper triangle of density matrix to full density matrix
def mean_vals_to_states(mean_vals):
    mean_states = np.empty(( spin_dim, spin_dim, mean_vals.shape[0] ), dtype = complex)
    for idx, ( mu, nu ) in enumerate(zip(*np.triu_indices(spin_dim))):
        mean_states[mu,nu,:] = mean_vals[:,idx]
        if mu != nu:
            mean_states[nu,mu,:] = mean_vals[:,idx].conj()
    return mean_states

def get_mean_states(mean_vals):
    if init_state_str in [ "X-L", "XX-L" ]:
        return mean_vals_to_states(mean_vals)
    if init_state_str == "P-Z":
        vals = mean_vals.shape[1]
        mean_vals_fst = mean_vals[:,:vals//2]
        mean_vals_snd = mean_vals[:,vals//2:]
        mean_vals_tot = ( mean_vals_fst + mean_vals_snd ) / 2
        return mean_vals_to_states(mean_vals_tot)

##########################################################################################
# plot summary figures
##########################################################################################

# get all time files
time_files = [ file for file in glob.glob(data_dir + f"times_{sys_tag}*")
               if any([ f"h{field:.2f}" in file for field in log10_field_vals_summary ]) ]

def get_field_tag(file):
    return file.split("_")[-2][1:] # return field tag, without the prefix
def get_angle_tag(file):
    angle_tag = file.split("_")[-1][1:] # get angle tag, without the prefix
    return ".".join(angle_tag.split(".")[:2]) # remove the file extension

def angle_filter(angle_tag):
    return lambda file : f"a{angle_tag}" in file

angle_tags = set([ get_angle_tag(file) for file in time_files ])
angle_time_files = { angle_tag : [ file for file in time_files if f"a{angle_tag}" in file ]
                     for angle_tag in angle_tags }

for angle_idx, ( angle_tag, time_files ) in enumerate(angle_time_files.items()):
    angle = float(angle_tag)
    time_files = list(sorted(time_files, key = lambda file : float(get_field_tag(file))))

    summary_plots = len(log10_field_vals_summary)
    figure, axes = plt.subplots(summary_plots, 3, figsize = (6.45, summary_plots+1))

    for field_idx, time_file in enumerate(time_files):
        field_tag = get_field_tag(time_file)
        mean_state_file = time_file.replace("times", "mean_states")

        print(f"{angle_idx}/{len(angle_tags)} " + \
              f"{field_idx}/{len(time_files)} " + \
              f"(angle_frac = {angle_tag}, log10_field = {field_tag})")
        sys.stdout.flush()

        field = 10**float(field_tag)
        times = np.loadtxt(time_file) / (2*np.pi) # "reduced" times
        mean_state_vals = np.loadtxt(mean_state_file, dtype = complex)
        mean_state_vals.shape = ( times.size, -1 )
        mean_states = get_mean_states(mean_state_vals)

        # interpolate at evenly spaces times
        times, mean_states = get_interp_vals(times, mean_states)
        freqs = np.fft.rfftfreq(times.size, times[1])

        # compute time-series and and frequency spectrum of S^2
        ss_vals = np.einsum("mnt,nmt->t", mean_states, mean_states).real
        ss_amps = np.fft.rfft(ss_vals) / times.size

        # identify peaks in time-series of S^2
        peaks, _ = scipy.signal.find_peaks(ss_vals)
        if plot_peaks < peaks.size:
            end_idx = peaks[plot_peaks]
        else:
            if peaks.size > 0:
                end_idx = peaks[-1]
            else:
                end_idx = -1

        # set maximum time / frequency
        max_plot_time = times[end_idx]
        max_plot_freq = max(1, peaks.size / times[-1] * freq_scale)
        mean_states = mean_states[:, :, times <= max_plot_time]
        ss_vals = ss_vals[times <= max_plot_time]
        ss_amps = ss_amps[freqs <= max_plot_freq]
        times = times[times <= max_plot_time]
        freqs = freqs[freqs <= max_plot_freq]

        ### plot time series and power spectrum of S^2
        axes[field_idx,0].plot(times, ss_vals) # time series data
        axes[field_idx,0].set_xlim(0, max_plot_time)

        axes[field_idx,1].semilogy(freqs[1:], abs(ss_amps[1:])**2) # power spectrum
        axes[field_idx,1].set_xlim(0, max_plot_freq)
        axes[field_idx,1].set_yticks([])

        ### spectrum of mean reduced single-particle density matrix
        spectrum = [ np.linalg.eigvalsh(mean_states[:,:,tt])[::-1]
                     for tt in range(times.size) ]
        axes[field_idx,2].plot(times, spectrum)
        axes[field_idx,2].set_xlim(0, max_plot_time)

        # show value of tunneling
        axes[field_idx,2].set_ylabel(f"${field_tag}$")
        axes[field_idx,2].yaxis.set_label_position("right")

    axes[0,0].set_title(r"$s^2$")
    axes[0,1].set_title(r"$\log\widetilde{s^2}$")
    axes[0,2].set_title(r"$\mathrm{spect}(\bar\rho)$")
    axes[-1,0].set_xlabel(r"$t \times U/2\pi$")
    axes[-1,1].set_xlabel(r"$\omega/U$")
    axes[-1,2].set_xlabel(r"$t \times U/2\pi$")
    figure.text(0.99, 0.5, r"$\log_{10}(h/U)$", rotation = 90, va = "center", ha = "right")
    figure.tight_layout(pad = 0.2, rect = (0,0,0.96,1))

    figure.savefig(fig_dir + f"ss_spect_{sys_tag}_a{angle_tag}.pdf")
    plt.close(figure)

##########################################################################################
# plot bulk data
##########################################################################################

# [to be completed...]
# for each angle, plot:
# - spectrum of SS as a function of field strength
# - time-series spectrum of density matrix for each fild strength
