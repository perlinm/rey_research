#!/usr/bin/env python3

import os, sys
import numpy as np

from mathieu_methods import mathieu_solution
from lattice_methods import qns_energy_gap, laser_overlap
from evolution_methods import ns_vec, select_populations, \
    time_integrated_populations, effective_driving_populations
from figure_methods import s_label
from sr87_olc_constants import recoil_energy_NU

# method used to compute populations
integration = (len(sys.argv) > 1)
if integration:
    propagation_method = time_integrated_populations
    method_text = "direct integration"
else:
    propagation_method = effective_driving_populations
    method_text = "effective hamiltonian"
data_dir = "../data/"


bands = 3
site_number = 100
time_steps = 500

rabi_coupling_NU = 1000 # Hz
q_ref, n_ref, s_ref = 0, 0, -1

lattice_depths = [ 80, 40, 10 ] # in recoil energies
mod_index = 1


assert(n_ref + 1 < bands)
initial_state = ns_vec(n_ref, s_ref, bands)
rabi_coupling = rabi_coupling_NU / recoil_energy_NU
time_text_tags = [q_ref, s_label(s_ref, sign = True), n_ref+1, n_ref]
time_text = r"$t\Omega^{{{},{}}}_{{{},{}}}/2\pi$".format(*time_text_tags)

# options:
#   maximum simulation time (in units of inter-band coupling at reference state)
#   mean detuning in units of band bap
#   ratio of (band gap) to (modulation frequency)
#   lattice depth
freq_options = [ [ 3, 0, 1, depth ] for depth in lattice_depths ]
amp_options = [ [ 5, -1/2, 2, depth ] for depth in lattice_depths ]

mod_options_dict = { "freq": freq_options,
                     "amp": amp_options }

mod_options = [ [ key, *options ]
                for key in mod_options_dict.keys()
                for options in mod_options_dict[key] ]

if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

def file_header(lattice_depth, rabi_coupling,
                mod_freq, mod_index, detuning_mean,
                coupling_within_band, coupling_across_bands,
                max_time, time_units, time_text):
    header = "# simulation method: {}\n".format(method_text)
    header += "# reference q, n, s: {}, {}, {}\n".format(q_ref, n_ref, s_ref)
    for description, value in [ [ "lattice depth", lattice_depth ],
                                [ "modulation frequency", mod_freq ],
                                [ "modulation index", mod_index ],
                                [ "detuning mean", detuning_mean ],
                                [ "rabi frequency", rabi_coupling ],
                                [ "intra-band coupling", coupling_within_band ],
                                [ "inter-band coupling", coupling_across_bands ] ]:
        header += "# {}: {}\n".format(description, value)
    header += "\n"
    header += "# maximum time: {}\n".format(max_time)
    header += "# time units (seconds): {}\n".format(time_units / recoil_energy_NU)
    header += "# frequency units (Hz): {}\n".format(recoil_energy_NU / time_units)
    header += "# time label: {}\n".format(time_text)
    header += "\n"
    header += "# q/k_L population...\n"
    return header

for mod_option in mod_options:
    mod_type, max_time, detuning_mean, gap_freq_ratio, lattice_depth = mod_option
    print(*mod_option)

    momenta, fourier_vecs, energies = mathieu_solution(lattice_depth, bands, site_number)
    band_gap = np.mean(energies[:,1]) - np.mean(energies[:,0])
    detuning_mean *= band_gap
    mod_freq = band_gap / gap_freq_ratio

    # laser-induced overlap within a band and across a band
    within_overlap = laser_overlap(q_ref, s_ref, n_ref, n_ref, momenta, fourier_vecs)
    across_overlap = laser_overlap(q_ref, s_ref, n_ref, n_ref + 1, momenta, fourier_vecs)

    coupling_within_band = abs(rabi_coupling / 2 * within_overlap)
    coupling_across_bands = abs(rabi_coupling / 2 * across_overlap)
    time_scale = 2*np.pi / coupling_across_bands

    times = np.linspace(0, max_time * time_scale, time_steps)

    header = file_header(lattice_depth, rabi_coupling,
                         mod_freq, mod_index, detuning_mean,
                         coupling_within_band, coupling_across_bands,
                         max_time, time_scale, time_text)

    base_name = data_dir + "{}_mod_{{}}_V{}.txt".format(mod_type, lattice_depth)
    band_ouput = open(base_name.format("band"), "w")
    spin_ouput = open(base_name.format("spin"), "w")
    band_ouput.write(header)
    spin_ouput.write(header)

    for q in momenta:
        print(" ",q)

        populations = propagation_method(q, n_ref, s_ref, momenta, energies, fourier_vecs,
                                         rabi_coupling, mod_type, mod_freq, mod_index,
                                         detuning_mean, bands, times)
        band_populations = select_populations(populations,"band")
        excited_band_populations = sum(np.delete(band_populations,n_ref,0), 0)
        excited_spin_populations = select_populations(populations,"spin")[1]

        band_ouput.write(str(q) + " ")
        spin_ouput.write(str(q) + " ")
        band_ouput.write(" ".join([ str(p) for p in excited_band_populations]) + "\n")
        spin_ouput.write(" ".join([ str(p) for p in excited_spin_populations]) + "\n")

    band_ouput.close()
    spin_ouput.close()
