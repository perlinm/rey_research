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

assert( spin_num % 2 == 0 )
assert( init_state_str in [ "X", "XX", "DS" ] )

figsize = (4,3)
figsize_ext = (5,3)
figsize_double = (5,4)

# plotting parameters
log10_tun_vals = np.linspace(1,-2,13)
soc_frac_vals = np.linspace(0,1,11)
plot_peaks = 100
freq_num = 1000
freq_scale = 2

data_dir = "../data/oscillations/"
fig_dir = "../figures/oscillations/time_series/"
sys_tag = f"n{spin_dim}_N{spin_num}_{init_state_str}"

params = { "font.size" : 12,
           "text.usetex" : True,
           "text.latex.preamble" : r"\usepackage{physics,braket,bm}" }
plt.rcParams.update(params)

def get_interp_vals(xs, ys, kind = "cubic"):
    interp = scipy.interpolate.interp1d(xs, ys, kind = kind)
    new_xs = np.linspace(xs[0], xs[-1], xs.size)
    new_ys = interp(new_xs)
    return new_xs, new_ys

Sx = drive_op(spin_dim, 1, +1) * drive_scale(spin_dim, 1)
Sy = drive_op(spin_dim, 1, -1) * drive_scale(spin_dim, 1)
R_ZX = scipy.linalg.expm(-1j*Sy*np.pi/2) # maps Z --> X

def D_Z(L, M): return drive_op(spin_dim, L, M)
def D_X(L, M): return R_ZX @ D_Z(L,M) @ R_ZX.conj().T
all_LM = [ (L,M) for L in range(1,spin_dim) for M in range(L,-L-1,-1) ]

##########################################################################################
# plot data
##########################################################################################
soc_frac_vals = soc_frac_vals[soc_frac_vals != 0]

param_generator = it.product( enumerate(soc_frac_vals), enumerate(log10_tun_vals) )
for ( idx_soc, soc_frac ), ( idx_tun, log10_tun ) in param_generator:
    soc_tag = f"{soc_frac:.2f}"
    tun_tag = f"{log10_tun:.2f}"
    file_tag = f"{sys_tag}_J{tun_tag}_p{soc_tag}"
    title = r"$\log_{10}(J/U)=" + tun_tag + r",~\varphi/\pi=" + soc_tag + "$"

    print(f"{idx_soc}/{soc_frac_vals.size} " + \
          f"{idx_tun}/{log10_tun_vals.size} " + \
          f"(soc_frac = {soc_tag}, log10_tun = {tun_tag})")
    sys.stdout.flush()

    tunneling = 10**log10_tun
    times = np.loadtxt(data_dir + f"times_{file_tag}.txt")
    spin_mat_vals = np.loadtxt(data_dir + f"spin_mats_{file_tag}.txt", dtype = complex)

    spin_mat_vals.shape = ( times.size, -1 )
    spin_mats = np.empty(( spin_dim, spin_dim, times.size ), dtype = complex)
    for idx, ( mu, nu ) in enumerate(zip(*np.triu_indices(spin_dim))):
        spin_mats[mu,nu,:] = spin_mat_vals[:,idx]
        if mu != nu:
            spin_mats[nu,mu,:] = spin_mat_vals[:,idx].conj()

    times, spin_mats = get_interp_vals(times, spin_mats)
    freqs = 2*np.pi * np.fft.rfftfreq(times.size, times[1])

    SS_vals = np.einsum("mnt,nmt->t", spin_mats, spin_mats).real
    SS_amps = np.fft.rfft(SS_vals-np.mean(SS_vals)) / times.size

    peaks, _ = scipy.signal.find_peaks(SS_vals)
    if plot_peaks < peaks.size:
        end_idx = peaks[plot_peaks]
    else:
        if peaks.size > 0:
            end_idx = peaks[-1]
        else:
            end_idx = -1

    max_plot_time = times[end_idx]
    max_plot_freq = (2*np.pi) * peaks.size / times[-1] * freq_scale
    spin_mats = spin_mats[:, :, times <= max_plot_time]
    SS_vals = SS_vals[times <= max_plot_time]
    SS_amps = SS_amps[freqs <= max_plot_freq]
    times = times[times <= max_plot_time]
    freqs = freqs[freqs <= max_plot_freq]

    ### plot time series and power spectrum of S^2
    figure, axes = plt.subplots(2, figsize = figsize_double)
    axes[0].set_title(title)
    axes[0].plot(times/(2*np.pi), SS_vals) # time series data
    axes[0].set_xlabel(r"$t \times U/2\pi$")
    axes[0].set_ylabel(r"$s^2$")
    axes[1].plot(freqs, abs(SS_amps)**2) # plot power spectrum
    axes[1].set_xlabel(r"$\omega/U$")
    axes[1].set_ylabel(r"$P(\omega)$")
    axes[1].set_yticks([])
    figure.tight_layout(pad = 0.3)
    figure.savefig(fig_dir + f"SS_{file_tag}.pdf")

    ### spectrum of mean reduced single-particle density matrix
    spectrum = [ np.linalg.eigvalsh(spin_mats[:,:,tt])[::-1]
                 for tt in range(times.size) ]
    plt.figure(figsize = figsize)
    plt.title(title)
    plt.plot(times, spectrum)
    plt.xlabel(r"$t \times U/2\pi$")
    plt.tight_layout()
    plt.savefig(fig_dir + f"spect_{file_tag}.pdf")

    ### z-basis drive operators
    LM_var = [ (L,M) for L, M in all_LM
               if ( M > 0 and (L + M) % 2 == 0 ) or
                  ( M < 0 and (L + M) % 2 == 1 ) ] # remove all (L,M) with abs(M) == 5?
    def D(L,M): return D_Z(L,M)
    plt.figure(figsize = figsize)
    plt.title(title)
    for L, M in LM_var:
        D_vals = np.einsum("mnt,nm->t", spin_mats, D(L,M)).real
        plt.plot(times/(2*np.pi), D_vals, label = f"${L},{M}$")
    plt.xlabel(r"$t \times U/2\pi$")
    plt.ylabel(r"$\braket{D_{LM,\mathrm{z}}}$")
    if spin_dim <= 4:
        plt.legend(bbox_to_anchor = (1, 1), loc = "upper left")
        plt.gcf().set_size_inches(*figsize_ext)
    plt.tight_layout()
    plt.savefig(fig_dir + f"drive_z_{file_tag}.pdf")

    ### x-basis drive operators
    LM_var = [ (L,M) for L, M in all_LM if M % 2 == 0 ]
    def D(L,M): return D_X(L,M)
    plt.figure(figsize = figsize)
    plt.title(title)
    for L, M in LM_var:
        D_vals = np.einsum("mnt,nm->t", spin_mats, D(L,M)).real
        plt.plot(times/(2*np.pi), D_vals, label = f"${L},{M}$")
    plt.xlabel(r"$t \times U/2\pi$")
    plt.ylabel(r"$\braket{D_{LM,\mathrm{x}}}$")
    if spin_dim <= 4:
        plt.legend(bbox_to_anchor = (1, 1), loc = "upper left")
        plt.gcf().set_size_inches(*figsize_ext)
    plt.tight_layout()
    plt.savefig(fig_dir + f"drive_x_{file_tag}.pdf")

    ### x-basis populations
    plt.figure(figsize = figsize_ext)
    plt.title(title)
    vals, vecs = np.linalg.eigh(Sx)
    for val, vec in list(zip(vals, vecs.T))[::-1]:
        pops = np.einsum("mnt,m,n->t", spin_mats, vec, vec.conj()).real
        plt.plot(times/(2*np.pi), pops, label = f"${val:.1f}$")
    plt.xlabel(r"$t \times U/2\pi$")
    plt.ylabel(r"$\braket{\mathcal{P}_{\mu,\mathrm{x}}}$")
    plt.legend(bbox_to_anchor = (1, 1), loc = "upper left")
    plt.tight_layout()
    plt.savefig(fig_dir + f"pops_x_{file_tag}.pdf")

    plt.close("all")
