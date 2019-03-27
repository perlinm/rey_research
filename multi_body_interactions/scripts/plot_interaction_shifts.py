#!/usr/bin/env python3

import os, sys
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats.mstats import gmean

from qubit_methods import act_on_subsets
from interaction_sym_methods import H_2_1, H_3_2, H_3_3, H_4_3, \
    sorted_eigenvalues, convert_eigenvalues
from interaction_num_methods import energy_correction_coefficients, \
    renormalized_coupling, momentum_coupling, excited_state_coupling
from sr87_olc_constants import colors, recoil_energy_Hz, g_int_LU, C6_AU

# simulation and plotting options
site_number = 100
bands = 15
pt_order = 3
single_figsize = np.array([3.3,2.7])
pair_figsize = np.array([6.5,2.7])
table_figsize = np.array([6.5,7])
summary_figsize = np.array([6,2.9])
params = { "text.usetex" : True,
           "text.latex.preamble" : [ r"\usepackage{amssymb}",
                                     r"\usepackage{dsfont}" ],
           "errorbar.capsize" : 3,
           "legend.fontsize" : "6" }
plt.rcParams.update(params)
np.set_printoptions(precision = 2, linewidth  = 200)
large_fontsize = 8

if "help" in sys.argv:
    print("usage: {} [save] [read] [split] [show]".format(sys.argv[0]))
    exit()

# process arguments
save = "save" in sys.argv # save figures
read = "read" in sys.argv # read in existing files containing theoretical shifts
split = "split" in sys.argv # read in existing files containing theoretical shifts
show = "show" in sys.argv # show figures

# set figure directory and (a)symmetric state flags
fig_dir = "../figures/interaction_shifts/"
symmetry_flags = [ "-", "+" ]

# import data
data_dir = "../data/interaction_shifts/"
data_file = data_dir + "shift_data.txt"
output_file_base = data_dir + "shift_predictions_{}.txt"
assert(os.path.isfile(data_file))

two_body_row = 8
def n_body_row(N):
    return two_body_row + 4 * ( N - 2 )

# collect experimental data
atom_numbers = [ 2, 3, 4, 5 ]
data = np.loadtxt(data_file)
data[two_body_row:,:] /= 1000
depths = data[[2,4,6],:].transpose()
depth_err = data[1,:]

kHz = recoil_energy_Hz / 1000 # conversion from lattice unit of energy to kHz
two_body_gg_to_eg_m_shifts = data[two_body_row,:] / kHz
two_body_gg_to_eg_p_shifts = data[two_body_row+2,:] / kHz

# predicted energy shifts at a given order
data_shape = (3,len(atom_numbers),len(depths))
shifts_1 = np.zeros(data_shape) # first order shifts
shifts_2 = np.zeros(data_shape) # second order shifts
shifts_3_3 = np.zeros(data_shape) # third order shifts (3-body)
shifts_3_4 = np.zeros(data_shape) # third order shifts (4-body)

# # shifts without renormalization
err_shifts_1 = np.zeros(data_shape)
err_shifts_2 = np.zeros(data_shape)
err_shifts_3_3 = np.zeros(data_shape)
err_shifts_3_4 = np.zeros(data_shape)

# approximate error due to nearest-neighbor effects
errs_tun = np.zeros(data_shape)

##########################################################################################
# calculations of interaction energies
##########################################################################################

if read:
    # if we are reading in existing predictions, we don't need to compute anything,
    #   so empty the list of lattice depths at which to compute energy shifts
    depths = []
else:
    # otherwise compute all eigenvalues of reduced Hamiltonians
    G_g, G_m, G_p, G_e = sp.symbols("g m p e")
    couplings = [ G_g, G_m, G_p, G_e ]

    # compute eigenvalues of Hamiltonians sans spatial / energy prefactors
    e_2_1 = sorted_eigenvalues(H_2_1(couplings))
    e_3_2 = sorted_eigenvalues(H_3_2(couplings))
    e_3_3_S, e_3_3_O = [ sorted_eigenvalues(H_3_3_X) for H_3_3_X in H_3_3(couplings) ]
    e_4_3_B, e_4_3_C = [ sorted_eigenvalues(H_4_3_X) for H_4_3_X in H_4_3(couplings) ]
    eigenvalues_2_1 = sp.zeros(3,len(atom_numbers))
    eigenvalues_3_2 = sp.zeros(3,len(atom_numbers))
    eigenvalues_3_3_S = sp.zeros(3,len(atom_numbers))
    eigenvalues_3_3_O = sp.zeros(3,len(atom_numbers))
    eigenvalues_4_3_B = sp.zeros(3,len(atom_numbers))
    eigenvalues_4_3_C = sp.zeros(3,len(atom_numbers))
    for [ vals, num, e_X ] in [ [ eigenvalues_2_1, 2, e_2_1 ],
                                [ eigenvalues_3_2, 3, e_3_2 ],
                                [ eigenvalues_3_3_S, 3, e_3_3_S ],
                                [ eigenvalues_3_3_O, 3, e_3_3_O ],
                                [ eigenvalues_4_3_B, 4, e_4_3_B ],
                                [ eigenvalues_4_3_C, 4, e_4_3_C ] ]:
        for aa, atom_number in enumerate(atom_numbers):
            conversion = convert_eigenvalues(num,atom_number)
            vals[:,aa] = conversion @ e_X

for dd in range(len(depths)):
    print("depths: {}/{}".format(dd,len(depths)))
    sys.stdout.flush()

    # compute spatial overlap factors
    overlaps = energy_correction_coefficients(depths[dd], site_number, pt_order, bands)
    a_2_1, a_prime_2_1 = overlaps[:2]
    if pt_order > 1:
        a_3_2 = overlaps[2]
    if pt_order > 2:
        a_3_3_1, a_3_3_2, a_4_3_1, a_4_3_2, a_4_3_3, a_5_3, g_2_2, g_3_2_1, g_3_2_2 \
            = overlaps[3:]

    # determine renormalized coupling constants and corresponding eigenvalues
    couplings = np.zeros(4)
    couplings[0] = renormalized_coupling(g_int_LU[0], depths[dd])
    couplings[1] = excited_state_coupling(two_body_gg_to_eg_m_shifts[dd],
                                          a_2_1, a_prime_2_1, couplings[0])
    couplings[2] = excited_state_coupling(two_body_gg_to_eg_p_shifts[dd],
                                          a_2_1, a_prime_2_1, couplings[0])
    couplings[3] = renormalized_coupling(g_int_LU[3], depths[dd])

    subs = { G_g: couplings[0], G_m: couplings[1], G_p: couplings[2] }
    e_2_1 = np.array(eigenvalues_2_1.subs(subs)).astype(float)
    e_3_2 = np.array(eigenvalues_3_2.subs(subs)).astype(float)
    e_3_3_S = np.array(eigenvalues_3_3_S.subs(subs)).astype(float)
    e_3_3_O = np.array(eigenvalues_3_3_O.subs(subs)).astype(float)
    e_4_3_B = np.array(eigenvalues_4_3_B.subs(subs)).astype(float)
    e_4_3_C = np.array(eigenvalues_4_3_C.subs(subs)).astype(float)

    # determine coupling constants and eigenvalues without renormalization
    err_couplings = np.zeros(4)
    err_couplings[0] = g_int_LU[0]
    err_couplings[1] = excited_state_coupling(two_body_gg_to_eg_m_shifts[dd],
                                              a_2_1, a_prime_2_1, err_couplings[0])
    err_couplings[2] = excited_state_coupling(two_body_gg_to_eg_p_shifts[dd],
                                              a_2_1, a_prime_2_1, err_couplings[0])
    err_couplings[3] = g_int_LU[3]

    err_subs = { G_g: err_couplings[0], G_m: err_couplings[1], G_p: err_couplings[2] }
    err_e_2_1 = np.array(eigenvalues_2_1.subs(err_subs)).astype(float)
    err_e_3_2 = np.array(eigenvalues_3_2.subs(err_subs)).astype(float)
    err_e_3_3_S = np.array(eigenvalues_3_3_S.subs(err_subs)).astype(float)
    err_e_3_3_O = np.array(eigenvalues_3_3_O.subs(err_subs)).astype(float)
    err_e_4_3_B = np.array(eigenvalues_4_3_B.subs(err_subs)).astype(float)
    err_e_4_3_C = np.array(eigenvalues_4_3_C.subs(err_subs)).astype(float)

    # determine momentum-dependent couplings and eigenvalues
    couplings_prime = np.array([ momentum_coupling(couplings[ii], C6_AU[ii])
                                 for ii in range(4) ])
    subs_prime = { G_g: couplings_prime[0],
                   G_m: couplings_prime[1],
                   G_p: couplings_prime[2] }
    e_prime_2_1 = np.array(eigenvalues_2_1.subs(subs_prime)).astype(float)

    err_couplings_prime = np.array([ momentum_coupling(err_couplings[ii], C6_AU[ii])
                                     for ii in range(4) ])
    err_subs_prime = { G_g: err_couplings_prime[0],
                       G_m: err_couplings_prime[1],
                       G_p: err_couplings_prime[2] }
    err_e_prime_2_1 = np.array(eigenvalues_2_1.subs(err_subs_prime)).astype(float)

    # first order energy shifts
    shifts_1[:,:,dd] = a_2_1 * e_2_1 + a_prime_2_1 * e_prime_2_1
    err_shifts_1[:,:,dd] = a_2_1 * err_e_2_1 + a_prime_2_1 * err_e_prime_2_1

    if pt_order == 1: continue

    # second order energy shifts
    shifts_2[:,:,dd] = -a_3_2 * e_3_2
    err_shifts_2[:,:,dd] = -a_3_2 * err_e_3_2

    if pt_order == 2: continue

    # third order, three-body energy shifts
    shifts_3_3[:,:,dd] = ( (a_3_3_1 - a_5_3) * e_3_3_S +
                           (2*a_3_3_2 - a_4_3_3 - a_5_3) * e_3_3_O )
    err_shifts_3_3[:,:,dd] = ( (a_3_3_1 - a_5_3) * err_e_3_3_S +
                               (2*a_3_3_2 - a_4_3_3 - a_5_3) * err_e_3_3_O )

    # third order, four-body energy shifts
    shifts_3_4[:,:,dd] = ( (2*a_4_3_1 - a_5_3) * e_4_3_B +
                           (a_4_3_2 - a_5_3) * e_4_3_C )
    err_shifts_3_4[:,:,dd] = ( (2*a_4_3_1 - a_5_3) * err_e_4_3_B +
                               (a_4_3_2 - a_5_3) * err_e_4_3_C )

    for aa in range(len(atom_numbers)):
        # estimate of error due to nearest-neighbor effects
        N = atom_numbers[aa]
        # determine prefactors for all states (ground and excited)
        # there are 6 neighboring sites
        # two-body processes have no participating atoms in neighboring sites
        # three-body processes have N-1 participating atoms in neighboring sites
        errs_tun[:,aa,dd] = 6 * max( abs(g_2_2), (N-1) * abs(g_3_2_1 + g_3_2_2) )
        # ground-state error: { N \choose 2 } = N(N-1)/2 on-site interaction terms
        errs_tun[0,aa,dd] *= N*(N-1)/2 * couplings[0]**2
        # excited-state error: the excited atom must be the one to hop,
        #   and it has N-1 on-site atoms to interact with
        errs_tun[1,aa,dd] *= (N-1) * max(couplings[:3])**2 # maximum possible couplings
        errs_tun[2,aa,dd] *= (N-1) * max(couplings[:3])**2 # maximum possible couplings

# total energy shifts
shifts_total = shifts_1 + shifts_2 + shifts_3_3 + shifts_3_4

# total potential error from renormalization
errs_ren \
    = ( shifts_1 + shifts_2 + shifts_3_3 + shifts_3_4 ) \
    - ( err_shifts_1 + err_shifts_2 + err_shifts_3_3 + err_shifts_3_4 ) )

##########################################################################################
# reading / writing interaction energies
##########################################################################################

# read in existing files containing theoretical shifts
if read:
    for shifts, tag in [ [ shifts_1, 1 ],
                         [ shifts_2, 2 ],
                         [ shifts_3_3, "3_3" ],
                         [ shifts_3_4, "3_4" ],
                         [ errs_ren, "errs_ren" ],
                         [ errs_tun, "errs_tun" ] ]:
        input_data = np.loadtxt(output_file_base.format(str(tag)))
        depths = input_data[:,:3]
        for aa in range(len(atom_numbers)):
            for ss in range(3):
                for dd in range(input_data.shape[0]):
                    base_column = 3 + 3*aa + ss
                    shifts[ss,aa,dd] = input_data[dd,base_column]

    shifts_total = shifts_1 + shifts_2 + shifts_3_3 + shifts_3_4

# save calculated predictions into files
elif save: # and not read
    def write_file(shifts, output_file, experiment = False):
        output_header = "# depth_x depth_y depth_z"
        for atom_number in atom_numbers:
            if not experiment:
                output_header += " {0}_g".format(atom_number)
            output_header += " {0}_- {0}_+".format(atom_number)
        output_header += "\n"

        with open(output_file, "w") as f:
            f.write(output_header)
            for dd in range(len(depths)):
                f.write("{} {} {}".format(*depths[dd]))
                for aa in range(len(atom_numbers)):
                    f.write(" {}".format(shifts[0,aa,dd]))
                    f.write(" {}".format(shifts[1,aa,dd]))
                    if not experiment:
                        f.write(" {}".format(shifts[2,aa,dd]))
                f.write("\n")

    shifts_expt = shifts_total[1:,:,:] - shifts_total[0,:,:]
    errs_expt = abs(errs_ren[1:,:,:] - errs_ren[0,:,:])
    errs_expt += abs(errs_tun[1:,:,:]) + abs(errs_tun[0,:,:])

    write_file(shifts_expt * kHz, output_file_base.format("expt"), experiment = True)
    write_file(errs_expt * kHz, output_file_base.format("expt_errs"), experiment = True)

    write_file(shifts_1, output_file_base.format(1))
    if pt_order >= 2:
        write_file(shifts_2, output_file_base.format(2))
    if pt_order >= 3:
        write_file(shifts_3_3, output_file_base.format("3_3"))
        write_file(shifts_3_4, output_file_base.format("3_4"))
        write_file(shifts_total, output_file_base.format("total"))
        write_file(errs_ren, output_file_base.format("errs_ren"))
        write_file(errs_tun, output_file_base.format("errs_tun"))

depths = gmean(depths.transpose()) # we only care about the mean depths from here on


##########################################################################################
# method for plotting data from individual data series
##########################################################################################

def make_plot(grid, top_index, bottom_index, atom_index, sym_index,
              xlabel = True, ylabel = True, legend = True,
              x_range = [30,80]):
    aa = atom_index
    ss = sym_index
    atom_number = atom_numbers[aa]

    # identify experimental data for this number of atoms
    data_row = n_body_row(atom_number)
    shift_data = data[[data_row,data_row+2],:]
    shift_data_err = data[[data_row+1,data_row+3],:]

    # determine excitation energies
    diffs_1 = shifts_1[1:,aa,:] - shifts_1[0,aa,:]
    diffs_2 = diffs_1 + shifts_2[1:,aa,:] - shifts_2[0,aa,:]
    diffs_3_3 = diffs_2 + shifts_3_3[1:,aa,:] - shifts_3_3[0,aa,:] # only 3-body
    diffs_3 = diffs_3_3 + shifts_3_4[1:,aa,:] - shifts_3_4[0,aa,:] # full third order

    # determine erros in excitation energies
    diff_errs_ren = abs(errs_ren[1:,aa,:] - errs_ren[0,aa,:])
    diff_errs_tun = abs(errs_tun[1:,aa,:]) + abs(errs_tun[0,aa,:])
    errs = diff_errs_ren + diff_errs_tun

    # determine points where the experimental data is nonzero
    points = shift_data[ss,:] != 0

    # bare data plot
    title = r"$N,X={},{}$".format(atom_number,symmetry_flags[ss])
    plt.subplot(grid[top_index]).set_title(title)

    plt.plot(depths[points], diffs_1[ss,points] * kHz, ".",
             label = r"$\mathcal{O}(G)$ theory")

    if pt_order >= 2 and atom_number >= 3:

        plt.plot(depths[points], diffs_2[ss,points] * kHz, ".",
                 label = r"$\mathcal{O}(G^2)$ theory")

    if pt_order >= 3 and atom_number >= 3:

        if atom_number >= 4 and split:
            plt.errorbar(depths[points], diffs_3_3[ss,points] * kHz,
                         errs[ss,points] * kHz,
                         fmt = ".", label = r"3-body $\mathcal{O}(G^3)$ theory")
            tag = "Full "
        else: tag = ""

        plt.errorbar(depths[points], diffs_3[ss,points] * kHz,
                     errs[ss,points] * kHz,
                     fmt = ".", label = tag + r"$\mathcal{O}(G^3)$ theory")

    # plot experimental data
    plt.errorbar(depths[points], shift_data[ss,points],
                 xerr = depth_err[points], yerr = shift_data_err[ss,points],
                 fmt = "k.", markerfacecolor = "none", zorder = 0,
                 label = "Experiment")

    # set nice horizontal axis limits (mutiples of 10 recoil energies)
    if x_range == None:
        x_min, x_max = depths[points][[0,-1]]
        plt.xlim(np.floor(x_min/10)*10, np.ceil(x_max/10)*10)
    else:
        plt.xlim(x_range[0], x_range[1])
    plt.gca().set_xticklabels([])

    if ylabel: plt.ylabel(r"$\Delta_{{NX}}$ ($2\pi$ kHz)")
    plt.legend(loc="best", framealpha = 1)
    handles, labels = plt.gca().get_legend_handles_labels()
    if not legend: plt.gca().legend_.remove()

    # residual plot
    plt.subplot(grid[bottom_index])
    plt.axhline(0,color="k",linewidth=0.5)

    # compute fractional residual frequency and error
    def frac_res(diffs):
        residuals = diffs * kHz - shift_data[ss,points]
        return residuals / shift_data[ss,points]
    combined_errs = np.sqrt((errs[ss,points] * kHz)**2 + shift_data_err[ss,points]**2)
    frac_errs = combined_errs / shift_data[ss,points]

    plt.plot(depths[points], frac_res(diffs_1[ss,points]), ".")

    if pt_order >= 2 and atom_number >= 3:

        plt.plot(depths[points], frac_res(diffs_2[ss,points]), ".")

    if pt_order >= 3 and atom_number >= 3:

        if atom_number >= 4 and split:
            plt.errorbar(depths[points], frac_res(diffs_3_3[ss,points]),
                         frac_errs, fmt = ".")

        plt.errorbar(depths[points], frac_res(diffs_3[ss,points]),
                     frac_errs, fmt = ".")

    if x_range == None:
        plt.xlim(np.floor(x_min/10)*10, np.ceil(x_max/10)*10)
    else:
        plt.xlim(x_range[0], x_range[1])

    if xlabel: plt.xlabel(r"Lattice depth $\mathcal{U}$ ($E_R$)")
    if ylabel: plt.ylabel(r"$\eta_{{NX}}$")

    return handles, labels


##########################################################################################
# make figures
##########################################################################################

# make figure directory if we need it and it does not exist
if save and not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

grid = gridspec.GridSpec
subgrid = gridspec.GridSpecFromSubplotSpec

# make all (resonance frequency) vs. (depth) plots
plt.figure("table", figsize = table_figsize)
table_rows = len(atom_numbers) - 1
table_grid = grid(table_rows, 1)
table_subgrids = [ subgrid(2, 2, hspace = 0.15, height_ratios = [2,1],
                           subplot_spec = table_grid[rr])
                   for rr in range(table_rows) ]

for aa in range(len(atom_numbers)):
    plt.figure("pair"+str(aa), figsize = pair_figsize)
    pair_grid = grid(1,1)
    pair_subgrid = subgrid(2, 2, hspace = 0.15, height_ratios = [2,1],
                           subplot_spec = pair_grid[0])

    for ss in [ 0, 1 ]:
        plt.figure(figsize = single_figsize)
        single_grid = grid(1,1)
        single_subgrid = subgrid(2, 1, hspace = 0.15, height_ratios = [2,1],
                                 subplot_spec = single_grid[0])
        make_plot(single_subgrid, 0, 1, aa, ss)
        plt.tight_layout()
        if save:
            name = "shifts_a{}_s{}.pdf".format(atom_numbers[aa],symmetry_flags[ss])
            plt.savefig(fig_dir + name)
            plt.close()

        plt.figure("pair"+str(aa))
        make_plot(pair_subgrid, ss, ss+2, aa, ss, ylabel = (not ss), legend = (not ss))

        if aa == 0: continue
        plt.figure("table")
        handles, labels = make_plot(table_subgrids[aa-1], ss, ss+2, aa, ss,
                                    xlabel = (aa == len(atom_numbers)-1),
                                    ylabel = (not ss), legend = False)

    plt.figure("pair"+str(aa))
    plt.tight_layout()
    if save:
        name = "shifts_a{}.pdf".format(atom_numbers[aa])
        plt.savefig(fig_dir + name)
        plt.close()

plt.figure("table")
plt.gcf().legend(handles, labels, ncol = len(handles), fontsize = large_fontsize,
                 loc = "center", bbox_to_anchor = (0.5,0.98))
plt.tight_layout(rect = (0,0,1,0.97))
if save:
    plt.savefig(fig_dir + "shifts_table.pdf")
    plt.close()

# make summary plot of experiment vs. theory comparison
plt.figure(figsize = summary_figsize)
plt.plot(np.zeros(len(depths)), depths, ".", color = colors[0])
for aa in range(len(atom_numbers)):
    N = atom_numbers[aa]
    color = colors[aa+1]
    data_row = n_body_row(N)
    shift_data = data[[data_row,data_row+2],:]
    points = shift_data[0,:] != 0
    for ss in range(2):
        plt.plot(shift_data[ss,points], depths[points], ".", color = color)

x_min, x_max = plt.gca().get_xlim()[0], 13
y_min, y_max = 30, plt.gca().get_ylim()[1]
plt.plot([0,0], depths[[0,-1]], "-", color = colors[0], linewidth = 0.5)
for aa in range(len(atom_numbers)):
    for ss in [ 1, 2 ]:
        shifts = ( shifts_total[ss,aa,:] - shifts_total[0,aa,:] )
        old_shifts = ( shifts_1[ss,aa,:] - shifts_1[0,aa,:] )
        plt.plot(shifts * kHz, depths, "-", color = colors[aa+1], linewidth = 0.5)
        plt.plot(old_shifts * kHz, depths, "--", color = colors[aa+1],
                 linewidth = 0.5, dashes = (5,5))

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xlabel("Detuning $\Delta$ ($2\pi$ kHz)")
plt.ylabel(r"Lattice depth $\mathcal{U}$ ($E_R$)")

# legend
plt.plot([], [], "k.", label = r"Experiment")
plt.plot([], [], "k--", label = r"$\mathcal{O}(G)$ theory", dashes = (5,5))
plt.plot([], [], "k-", label = r"$\mathcal{O}(G^3)$ theory")
legend = plt.gca().legend(loc = "best", fontsize = large_fontsize, framealpha = 0.3)
for line in legend.get_lines():
    line.set_linewidth(0.5)

# markers identifying data series
peaks = [ (0) ] + [ ( shifts_total[ss+1,aa,-1] - shifts_total[0,aa,-1] ) * kHz
                    for ss in range(2)
                    for aa in range(len(atom_numbers)) ]

HR = plt.gca().get_xlim()[1] - plt.gca().get_xlim()[0]
VR = plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]
plt.gca().text(peaks[0]+0.00*HR, depths[-1]+0.08*VR, r"$1$", color = colors[0])
plt.gca().text(peaks[1]-0.03*HR, depths[-1]+0.08*VR, r"$2^-$", color = colors[1])
plt.gca().text(peaks[2]-0.04*HR, depths[-1]+0.08*VR, r"$3^-$", color = colors[2])
plt.gca().text(peaks[3]+0.01*HR, depths[-1]+0.08*VR, r"$4^-$", color = colors[3])
plt.gca().text(peaks[4]+0.01*HR, depths[-1]+0.08*VR, r"$5^-$", color = colors[4])
plt.gca().text(peaks[5]-0.01*HR, depths[-1]+0.08*VR, r"$2^+$", color = colors[1])
plt.gca().text(peaks[6]-0.00*HR, depths[-1]+0.08*VR, r"$3^+$", color = colors[2])
plt.gca().text(peaks[7]-0.00*HR, depths[-1]+0.08*VR, r"$4^+$", color = colors[3])
plt.gca().text(peaks[8]-0.02*HR, depths[-1]+0.08*VR, r"$5^+$", color = colors[4])

plt.tight_layout(rect = (0,0,1,0.95))
if save:
    plt.savefig(fig_dir + "shifts_summary.pdf")
    plt.close()

if show: plt.show()
