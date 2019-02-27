#!/usr/bin/env python3

import sys
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from numpy.matlib import repmat
from scipy.integrate import quad
from scipy.optimize import minimize
from mathieu_methods import mathieu_solution
from interaction_num_methods import energy_correction_coefficients
from interaction_sym_methods import H_2_1, H_3_2, H_3_3, H_4_3, nCk, \
    convert_eigenvalues, sorted_eigenvalues
from overlap_methods import kinetic_overlap_1D, pair_overlap_1D
from sr87_olc_constants import g_int_LU, C6_AU

save = "save" in sys.argv

max_bands = 15
site_number = 100
figsize = (4,3)
params = { "text.usetex" : True,
           "text.latex.preamble" : [ r"\usepackage{dsfont}" ],
           "errorbar.capsize" : 3 }
plt.rcParams.update(params)

data_dir = "../data/"
fig_dir = "../figures/broadening_lifetimes/"

data_file = "broadening.txt"

state_tags = [ "ground", "asymmetric", "symmetric" ]
states = [ r"$g\cdots$", r"$eg\cdots^-$", r"$eg\cdots^+$" ]

linestyles = ["-","--","-.",":"]
atom_numbers = [ 2, 3, 4, 5 ]
lattice_depths = [ 30, 40, 50, 60, 70 ]

loss_depth = 40
loss_depth_index = lattice_depths.index(loss_depth)

# 1-D broadened single-particle energy from overlap with lattice
def lattice_energy_1D(lattice_depth, scale_factor, momenta, fourier_vecs,
                      subinterval_limit = 500):
    site_number = len(fourier_vecs[:,0,0])
    fourier_terms = len(fourier_vecs[0,0,:])
    k_offset = int(fourier_terms)//2
    k_values = 2 * (np.arange(fourier_terms) - k_offset)

    def integrand(z):
        q_phases = repmat(np.exp(complex(0,1) * momenta * z), fourier_terms, 1).T
        k_phases = repmat(np.exp(complex(0,1) * k_values * z), site_number, 1)
        return ( np.sin(scale_factor * z)**2 *
                 np.real(np.sum(fourier_vecs[:,0,:] * q_phases * k_phases))**2 )

    mw = np.sqrt(lattice_depth)
    z_max = np.sqrt( 1 / mw ) + 2*np.pi
    normalization = np.pi * site_number**2
    integral = 2 * quad(integrand, 0, z_max, limit = subinterval_limit)[0]
    return lattice_depth * integral / normalization

# compute eigenvalues of Hamiltonians sans spacial integral factors
e_2_1 = np.array(sorted_eigenvalues(H_2_1(g_int_LU))).astype(float)
e_3_2 = np.array(sorted_eigenvalues(H_3_2(g_int_LU))).astype(float)
e_3_3_S, e_3_3_O = [ np.array(sorted_eigenvalues(H_3_3_X)).astype(float)
                     for H_3_3_X in H_3_3(g_int_LU) ]
e_4_3_B, e_4_3_C = [ np.array(sorted_eigenvalues(H_4_3_X)).astype(float)
                     for H_4_3_X in H_4_3(g_int_LU) ]
eigenvalues_2_1 = np.zeros((len(atom_numbers),3))
eigenvalues_3_2 = np.zeros((len(atom_numbers),3))
eigenvalues_3_3_S = np.zeros((len(atom_numbers),3))
eigenvalues_3_3_O = np.zeros((len(atom_numbers),3))
eigenvalues_4_3_B = np.zeros((len(atom_numbers),3))
eigenvalues_4_3_C = np.zeros((len(atom_numbers),3))
for [ vals, num, e_X ] in [ [ eigenvalues_2_1, 2, e_2_1 ],
                            [ eigenvalues_3_2, 3, e_3_2 ],
                            [ eigenvalues_3_3_S, 3, e_3_3_S ],
                            [ eigenvalues_3_3_O, 3, e_3_3_O ],
                            [ eigenvalues_4_3_B, 4, e_4_3_B ],
                            [ eigenvalues_4_3_C, 4, e_4_3_C ] ]:
    for aa in range(len(atom_numbers)):
        conversion = np.array(convert_eigenvalues(num,atom_numbers[aa])).astype(float)
        vals[aa,:] = (conversion @ e_X).T

# compute wavefunction broadening estimates
scale_factors = np.zeros((len(atom_numbers),len(lattice_depths),3))
for dd in range(len(lattice_depths)):
    lattice_depth = lattice_depths[dd]
    print()
    print("lattice_depth:", lattice_depth)
    print("-"*80)
    print()

    momenta, fourier_vecs, energies = mathieu_solution(lattice_depth,
                                                       max_bands, site_number)

    kinetic_energy = kinetic_overlap_1D(momenta, fourier_vecs)
    pair_overlap = pair_overlap_1D(momenta, fourier_vecs)

    overlaps = energy_correction_coefficients(lattice_depth * np.ones(3), site_number)
    a_2_1, a_prime_2_1, a_3_2, a_3_3_1, a_3_3_2, a_4_3_1, a_4_3_2, a_4_3_3, a_5_3 \
        = overlaps[:8]

    for aa in range(len(atom_numbers)):
        atom_number = atom_numbers[aa]
        print("atom number:", atom_number)

        # TODO: incorporate effect of renormalization and momentum-dependent coupling
        multi_body_energies = ( a_2_1 * eigenvalues_2_1[aa,:]
                                - a_3_2 * eigenvalues_3_2[aa,:]
                                + (a_3_3_1 - a_5_3) * eigenvalues_3_3_S[aa,:]
                                + (2*a_3_3_1 - a_4_3_3 - a_5_3) * eigenvalues_3_3_O[aa,:]
                                + (2*a_4_3_1 - a_5_3) * eigenvalues_4_3_B[aa,:]
                                + (a_4_3_2 - a_5_3) * eigenvalues_4_3_C[aa,:] )

        def E_nonint(scale_factor):
            lattice_energy = lattice_energy_1D(lattice_depth, scale_factor,
                                               momenta, fourier_vecs)
            return 3 * ( kinetic_energy / scale_factor**2 + lattice_energy) * atom_number

        def E_int(scale_factor, ss):
            conversion = np.array(convert_eigenvalues(2,atom_number)).astype(float)
            factors = conversion @ g_int_LU[:3]
            return factors[ss] * (pair_overlap/scale_factor)**3

        for ss in range(3):
            print("-"*40)
            print("state:", state_tags[ss])

            def total_energy(scale_factor):
                return E_nonint(scale_factor) + E_int(scale_factor,ss)

            minimization_results = minimize(total_energy, 1)
            scale_factors[aa,dd,ss] = minimization_results["x"][0]

            scale_factor = scale_factors[aa,dd,ss]
            E_int_broad = E_int(scale_factor,ss)
            E_nonint_broad = E_nonint(scale_factor)
            E_nonint_normal = E_nonint(1)
            DE_broadening = E_int_broad + E_nonint_broad - E_nonint_normal

            print("scale factor:", scale_factor)
            print("-"*20)
            print("%.5s broadened interaction energy" % str(E_int_broad))
            print("%.5s lattice+kinetic energy correction from broadening"
                  % str(E_nonint_broad - E_nonint_normal))
            print("%.5s total broadening + interaction energy shift" % str(DE_broadening))
            print("-"*20)
            print("%.5s 'actual' multi-body interaction energy"
                  % str(multi_body_energies[ss]))
            print("-"*20)
            print("%.5s broadening : 'actual' correction ratio"
                  % str(DE_broadening/multi_body_energies[ss]))

        print()

# make plots of scale factor dependence on lattice depth
for ss in range(3):
    plt.figure(figsize=figsize)
    for aa in range(len(atom_numbers))[::-1]:
        plt.plot(lattice_depths, (scale_factors[aa,:,ss] - 1) * 100,
                 "k" + linestyles[::-1][aa],
                 label = r"$N={}$".format(atom_numbers[aa]))

    plt.title("State: " + states[ss])
    plt.xlim(lattice_depths[0],lattice_depths[-1])
    plt.ylim(0,plt.gca().get_ylim()[1])

    plt.xlabel(r"Lattice depth ($E_R$)")
    plt.ylabel(r"Broadening (\%)")

    plt.legend(loc="best").get_frame().set_alpha(1)
    plt.gca().tick_params(bottom=True,left=True,right=True)
    plt.tight_layout()

    plt.savefig(fig_dir + "broadening_{}.pdf".format(state_tags[ss]))

plt.close("all") # close all figures

# scale factors for loss data parameters; indexed by aa, ss
loss_scalings = scale_factors[:,loss_depth_index,:]

##########################################################################################
# make loss figures
##########################################################################################

M = 3 # losses dominated by 3-body processes
loss_file = "../data/lifetimes.txt"

momenta, fourier_vecs, energies = mathieu_solution(loss_depth, max_bands, site_number)

# import loss data
# lifetimes and error in seconds
data = np.loadtxt(loss_file)
data_atom_numbers = data[0,:].astype(int)
egg_p_lifetimes = data[1,:]
egg_p_errs = data[2,:]
egg_m_lifetimes = data[3,:]
egg_m_errs = data[4,:]
ggg_lifetimes = data[5,:]
ggg_errs = data[6,:]

# all loss rates indexed by [ eigenstate, spectator_atoms ]
all_lifetimes = sp.Matrix([ ggg_lifetimes, egg_p_lifetimes, egg_m_lifetimes ])
all_errs = sp.Matrix([ ggg_errs, egg_p_errs, egg_m_errs ])

# collect vector of 3-body loss rates
lifetimes_num = all_lifetimes[:,0]
errs_num = all_errs[:,0]

for ss in range(3):
    plt.figure(ss, figsize = figsize)
    plt.errorbar(data_atom_numbers, list(all_lifetimes[ss,:]), list(all_errs[ss,:]),
                 fmt = ".", label = "Measured")

def invert(vec):
    new_vec = sp.zeros(vec.shape[0],vec.shape[1])
    for ii in range(len(new_vec)):
        new_vec[ii] = 1 / vec[ii]
    return new_vec

def predicted_rates(N, lifetimes, use_scaling):
    rates = sp.zeros(3,1)
    for ss in range(3):
        if use_scaling:
            scale_3 = loss_scalings[atom_numbers.index(3),ss]
            scale_N = loss_scalings[atom_numbers.index(N),ss]
            overlap_factor = (scale_3/scale_N)**3
        else:
            overlap_factor = 1
        rates[ss] = 1/lifetimes[ss] * overlap_factor
    return sp.factor(sp.simplify(convert_eigenvalues(M,N) @ rates))

for use_scaling, label in [ [ False, "Expected (no broadening)" ],
                            [ True, "Expected (with broadening)" ] ]:

    all_rates_predicted = sp.zeros(3,len(data_atom_numbers))
    all_rates_max = sp.zeros(3,len(data_atom_numbers))
    all_rates_min = sp.zeros(3,len(data_atom_numbers))

    for aa in range(len(data_atom_numbers)):
        N = data_atom_numbers[aa]
        assert(N >= M)
        all_rates_predicted[:,aa] = predicted_rates(N, lifetimes_num, use_scaling)
        all_rates_max[:,aa] = predicted_rates(N, lifetimes_num - errs_num, use_scaling)
        all_rates_min[:,aa] = predicted_rates(N, lifetimes_num + errs_num, use_scaling)

    all_lifetimes_predicted = invert(all_rates_predicted)
    all_lifetimes_max = invert(all_rates_min)
    all_lifetimes_min = invert(all_rates_max)
    all_errs_upper = all_lifetimes_max - all_lifetimes_predicted
    all_errs_lower = all_lifetimes_predicted - all_lifetimes_min

    for ss in range(3):
        plt.figure(ss)
        plt.errorbar(data_atom_numbers[1:], list(all_lifetimes_predicted[ss,1:]),
                     [ list(all_errs_upper[ss,1:]), list(all_errs_lower[ss,1:]) ],
                     fmt = ".", label = label)

for ss in range(3):
    plt.figure(ss)
    plt.title("State: " + states[ss])
    plt.xlabel("Atom number")
    plt.ylabel("Lifetime (s)")
    plt.ylim(0,plt.gca().get_ylim()[1])
    plt.xticks(data_atom_numbers)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(fig_dir + "lifetimes_{}.pdf".format(state_tags[ss]))
