#!/usr/bin/env python3

import sys
import numpy as np
import scipy.signal, scipy.linalg
import matplotlib.pyplot as plt
import itertools as it

from multilevel_methods import drive_op, drive_scale

spin_dim = int(sys.argv[1])
spin_num = int(sys.argv[2])
init_state_str = sys.argv[3]

figsize = (4,3)
figsize_ext = (5,3)
figsize_double = (5,4)

# plotting parameters
log10_tun_vals_smry = np.linspace(-2,1,7)
log10_tun_vals_bulk = np.arange(-2,1.01,0.05)
soc_frac_vals = [ 0.1, 0.5, 0.9, 1.0 ]
plot_peaks = 50
freq_num = 1000
freq_scale = 2

data_dir = "../data/oscillations/"
fig_dir = "../figures/oscillations/time_series/"
sys_tag = f"n{spin_dim}_N{spin_num}_{init_state_str}"

params = { "font.size" : 10,
           "text.usetex" : True }
plt.rcParams.update(params)

def get_interp_vals(xs, ys, kind = "cubic"):
    interp = scipy.interpolate.interp1d(xs, ys, kind = kind)
    new_xs = np.linspace(xs[0], xs[-1], xs.size)
    new_ys = interp(new_xs)
    return new_xs, new_ys

Sx = drive_op(spin_dim, 1, +1) * drive_scale(spin_dim, 1)
Sy = drive_op(spin_dim, 1, -1) * drive_scale(spin_dim, 1)
R_ZX = scipy.linalg.expm(-1j*Sy*np.pi/2) # maps Z --> X

def D_z(L, M): return drive_op(spin_dim, L, M)
def D_x(L, M): return R_ZX @ D_z(L,M) @ R_ZX.conj().T
all_LM = [ (L,M) for L in range(1,spin_dim) for M in range(L,-L-1,-1) ]

##########################################################################################
# plot summary figures
##########################################################################################

for idx_soc, soc_frac in enumerate(soc_frac_vals):
    soc_tag = f"{soc_frac:.2f}"

    figure, axes = plt.subplots(7, 3, figsize = (6.45, 8))

    for idx_tun, log10_tun in enumerate(log10_tun_vals_smry):
        tun_tag = f"{log10_tun:.2f}"
        file_tag = f"{sys_tag}_J{tun_tag}_p{soc_tag}"

        print(f"{idx_soc}/{len(soc_frac_vals)} " + \
              f"{idx_tun}/{len(log10_tun_vals_smry)} " + \
              f"(soc_frac = {soc_tag}, log10_tun = {tun_tag})")
        sys.stdout.flush()

        tunneling = 10**log10_tun
        times = np.loadtxt(data_dir + f"times_{file_tag}.txt") / (2*np.pi)
        spin_mat_vals = np.loadtxt(data_dir + f"spin_mats_{file_tag}.txt", dtype = complex)

        spin_mat_vals.shape = ( times.size, -1 )
        spin_mats = np.empty(( spin_dim, spin_dim, times.size ), dtype = complex)
        for idx, ( mu, nu ) in enumerate(zip(*np.triu_indices(spin_dim))):
            spin_mats[mu,nu,:] = spin_mat_vals[:,idx]
            if mu != nu:
                spin_mats[nu,mu,:] = spin_mat_vals[:,idx].conj()

        times, spin_mats = get_interp_vals(times, spin_mats)
        freqs = np.fft.rfftfreq(times.size, times[1])

        ss_vals = np.einsum("mnt,nmt->t", spin_mats, spin_mats).real
        ss_amps = np.fft.rfft(ss_vals) / times.size

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
        spin_mats = spin_mats[:, :, times <= max_plot_time]
        ss_vals = ss_vals[times <= max_plot_time]
        ss_amps = ss_amps[freqs <= max_plot_freq]
        times = times[times <= max_plot_time]
        freqs = freqs[freqs <= max_plot_freq]

        ### plot time series and power spectrum of S^2
        axes[idx_tun,0].plot(times, ss_vals) # time series data
        axes[idx_tun,0].set_xlim(0, max_plot_time)

        axes[idx_tun,1].semilogy(freqs[1:], abs(ss_amps[1:])**2) # power spectrum
        axes[idx_tun,1].set_xlim(0, max_plot_freq)
        axes[idx_tun,1].set_yticks([])

        ### spectrum of mean reduced single-particle density matrix
        spectrum = [ np.linalg.eigvalsh(spin_mats[:,:,tt])[::-1]
                     for tt in range(times.size) ]
        axes[idx_tun,2].plot(times, spectrum)
        axes[idx_tun,2].set_xlim(0, max_plot_time)

        # show value of tunneling
        axes[idx_tun,2].set_ylabel(f"${tun_tag}$")
        axes[idx_tun,2].yaxis.set_label_position("right")

    axes[0,0].set_title(r"$s^2$")
    axes[0,1].set_title(r"$\log\widetilde{s^2}$")
    axes[0,2].set_title(r"$\mathrm{spect}(\bar\rho)$")
    axes[-1,0].set_xlabel(r"$t \times U/2\pi$")
    axes[-1,1].set_xlabel(r"$\omega/U$")
    axes[-1,2].set_xlabel(r"$t \times U/2\pi$")
    figure.text(0.99, 0.5, r"$\log_{10}(J/U)$", rotation = 90, va = "center", ha = "right")
    figure.tight_layout(pad = 0.1, rect = (0,0,0.96,1))

    fig_tag = f"{sys_tag}_p{soc_tag}"
    figure.savefig(fig_dir + f"ss_spect_{fig_tag}.pdf")

    plt.close(figure)

##########################################################################################
# plot bulk data
##########################################################################################

param_generator = it.product( enumerate(soc_frac_vals), enumerate(log10_tun_vals_bulk) )
for ( idx_soc, soc_frac ), ( idx_tun, log10_tun ) in param_generator:
    soc_tag = f"{soc_frac:.2f}"
    tun_tag = f"{log10_tun:.2f}"
    file_tag = f"{sys_tag}_J{tun_tag}_p{soc_tag}"
    title = r"$\log_{10}(J/U)=" + tun_tag + r",~\varphi/\pi=" + soc_tag + "$"

    print(f"{idx_soc}/{len(soc_frac_vals)} " + \
          f"{idx_tun}/{len(log10_tun_vals_bulk)} " + \
          f"(soc_frac = {soc_tag}, log10_tun = {tun_tag})")
    sys.stdout.flush()

    tunneling = 10**log10_tun
    times = np.loadtxt(data_dir + f"times_{file_tag}.txt") / (2*np.pi)
    spin_mat_vals = np.loadtxt(data_dir + f"spin_mats_{file_tag}.txt", dtype = complex)

    spin_mat_vals.shape = ( times.size, -1 )
    spin_mats = np.empty(( spin_dim, spin_dim, times.size ), dtype = complex)
    for idx, ( mu, nu ) in enumerate(zip(*np.triu_indices(spin_dim))):
        spin_mats[mu,nu,:] = spin_mat_vals[:,idx]
        if mu != nu:
            spin_mats[nu,mu,:] = spin_mat_vals[:,idx].conj()

    times, spin_mats = get_interp_vals(times, spin_mats)
    freqs = np.fft.rfftfreq(times.size, times[1])

    ss_vals = np.einsum("mnt,nmt->t", spin_mats, spin_mats).real
    ss_amps = np.fft.rfft(ss_vals) / times.size

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
    spin_mats = spin_mats[:, :, times <= max_plot_time]
    ss_vals = ss_vals[times <= max_plot_time]
    ss_amps = ss_amps[freqs <= max_plot_freq]
    times = times[times <= max_plot_time]
    freqs = freqs[freqs <= max_plot_freq]

    ### plot time series and power spectrum of S^2
    figure, axes = plt.subplots(2, figsize = figsize_double)
    axes[0].set_title(title)
    axes[0].plot(times, ss_vals) # time series data
    axes[0].set_xlim(0, max_plot_time)
    axes[0].set_xlabel(r"$t \times U/2\pi$")
    axes[0].set_ylabel(r"$s^2$")
    axes[1].semilogy(freqs[1:], abs(ss_amps[1:])**2) # power spectrum
    axes[1].set_xlim(0, max_plot_freq)
    axes[1].set_xlabel(r"$\omega/U$")
    axes[1].set_ylabel(r"$\log P(\omega)$")
    axes[1].set_yticks([])
    figure.tight_layout(pad = 0.3)
    figure.savefig(fig_dir + f"ss_{file_tag}.pdf")

    ### spectrum of mean reduced single-particle density matrix
    spectrum = [ np.linalg.eigvalsh(spin_mats[:,:,tt])[::-1]
                 for tt in range(times.size) ]
    plt.figure(figsize = figsize)
    plt.title(title)
    plt.plot(times, spectrum)
    plt.xlim(0, max_plot_time)
    plt.xlabel(r"$t \times U/2\pi$")
    plt.tight_layout()
    plt.savefig(fig_dir + f"spect_{file_tag}.pdf")

    plt.close("all")
