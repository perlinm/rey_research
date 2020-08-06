#!/usr/bin/env python3

import sys
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

np.set_printoptions(linewidth = 200)

spin_dim = int(sys.argv[1])
spin_num = int(sys.argv[2])
if len(sys.argv) > 3:
    init_state_str = sys.argv[3]
else:
    init_state_str = "X"

assert( spin_num % 2 == 0 )
assert( init_state_str in [ "X", "DS" ] )
if spin_dim == 2: init_state_str = "X"

figsize = (4,3)
color_map = "inferno"

# plotting parameters
log10_tun_vals = np.linspace(-2,1,13)
soc_frac_vals = np.linspace(0,1,11)
plot_peaks = 100
freq_num = 1000
freq_scale = 2

data_dir = "../data/oscillations/"
fig_dir = "../figures/oscillations/"
sys_tag = f"n{spin_dim}_N{spin_num}_{init_state_str}"

params = { "font.size" : 12,
           "text.usetex" : True,
           "text.latex.preamble" : [ r"\usepackage{physics}",
                                     r"\usepackage{braket}" ]}
plt.rcParams.update(params)

def get_interp_vals(xs, ys, kind = "cubic"):
    interp = scipy.interpolate.interp1d(xs, ys, kind = kind)
    new_xs = np.linspace(xs[0], xs[-1], xs.size)
    new_ys = interp(new_xs)
    return new_xs, new_ys

##########################################################################################
# plot data
##########################################################################################

param_generator = zip( enumerate(soc_frac_vals), enumerate(log10_tun_vals) )
for ( idx_soc, soc_frac ), ( idx_tun, log10_tun ) in param_generator:
    soc_tag = f"{soc_frac:.2f}"
    tun_tag = f"{log10_tun:.2f}"
    file_tag = f"{sys_tag}_J{tun_tag}_p{soc_tag}"
    title = r"$\log_{10}(J/U)=" + tun_tag + r",~\phi/\pi=" + soc_tag + "$"

    print(f"{idx_soc}/{soc_frac_vals.size} " + \
          f"{idx_tun}/{log10_tun_vals.size} " + \
          f"(soc_frac, log10_tun = {soc_tag}, {tun_tag})")
    sys.stdout.flush()

    tunneling = 10**log10_tun
    times = np.loadtxt(data_dir + f"times_{file_tag}.txt")
    spin_mat_vals = np.loadtxt(data_dir + f"spin_mats_{file_tag}.txt", dtype = complex)

    spin_mat_vals.shape = ( times.size, -1 )
    spin_mats = np.empty(( times.size, spin_dim, spin_dim ), dtype = complex)
    for idx, ( mu, nu ) in enumerate(zip(*np.triu_indices(spin_dim))):
        spin_mats[:,mu,nu] = spin_mat_vals[:,idx]
        if mu != nu:
            spin_mats[:,nu,mu] = spin_mat_vals[:,idx].conj()

    SS_vals = np.einsum("tmn,tnm->t", spin_mats, spin_mats).real

    peaks, _ = scipy.signal.find_peaks(SS_vals)

    end_idx = peaks[plot_peaks] if plot_peaks < peaks.size else peaks[-1]
    max_plot_time = times[end_idx]
    max_plot_freq = (2*np.pi) * peaks.size / times[-1] * freq_scale

    figure, axes = plt.subplots(2)
    axes[0].set_title(title)

    # plot time-series data
    axes[0].plot(times/(2*np.pi), SS_vals)
    axes[0].set_xlabel(r"$t \times U/2\pi$")
    axes[0].set_ylabel(r"$\braket{S^2}$")

    # plot power spectrum
    times, SS_vals = get_interp_vals(times, SS_vals)
    freqs = 2*np.pi * np.fft.rfftfreq(times.size, times[1])
    SS_amps = np.fft.rfft(SS_vals-np.mean(SS_vals)) / times.size
    axes[1].plot(freqs, abs(SS_amps)**2)
    axes[1].set_xlabel(r"$\omega/U$")
    axes[1].set_ylabel(r"$P(\omega)$")
    axes[1].set_yticks([])

    axes[0].set_xlim(0, max_plot_time/(2*np.pi))
    axes[1].set_xlim(0, max_plot_freq)
    figure.tight_layout()
    figure.savefig(fig_dir + f"time_series/SS_{file_tag}.pdf")
    plt.close(figure)
