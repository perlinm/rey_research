#!/usr/bin/env python3

import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt

from dicke_methods import spin_op_vec_dicke

dim = int(sys.argv[1])

figsize = (4,3)
data_dir = "../data/spin_pair/"
fig_dir = "../figures/spin_pair/"

# set fonts and use latex packages
params = { "font.family" : "serif",
           "font.serif" : "Computer Modern",
           "text.usetex" : True,
           "text.latex.preamble" : r"\usepackage{bm}",
           "font.size" : 10 }
plt.rcParams.update(params)

##################################################

def get_field(file, string = False):
    field = file.split("_")[-1][1:-4]
    if not string: field = float(field)
    return field

def wrap_label(label, double_braket = False):
    if not double_braket:
        return r"$\langle " + label + r"\rangle$"
    else:
        return r"$\langle\!\langle " + label + r"\rangle\!\rangle$"

def time_series_name(file, tag):
    basename = os.path.basename(file).replace("states",tag).replace(".txt",".png")
    return fig_dir + "time_series/" + basename
def LTA_name(file, tag):
    tmp = os.path.basename(file).replace("states",tag)
    basename = "_".join(tmp.split("_")[:-1]) + ".png"
    return fig_dir + "LTAs/" + basename

##################################################

spin = (dim-1)/2

# spin matrices
spin_ops = [ op.todense()/spin for op in spin_op_vec_dicke(dim-1) ]
Sz, Sx, Sy = spin_ops

def vals(op, states):
    return np.einsum("ij,ti,tj->t", op, states.conj(), states)

files = glob.glob(data_dir + f"states_d{dim:02d}_h*.txt")
files = sorted(files, key = get_field)
fields = list(map(get_field,files))
LTAs = {} # long-time averages
LTA_info = {}

for field_idx, ( field, file ) in enumerate(sorted(zip(fields,files))):
    data = np.loadtxt(file, dtype = complex)
    times, states_0, states_1 = data[:,0], data[:,1:dim+1], data[:,dim+1:]
    times = times.real

    # construct human-readable title
    field_str = get_field(file, string = True)
    if field < 0:
        sign = "-"
        abs_field = field_str[1:]
    else:
        sign = "+"
        abs_field = field_str
    title = r"$\log_{10}(h/U)=" + f"{sign}{abs_field}$"

    # compute observables of interest
    exchange = abs(np.einsum("ti,ti->t", states_0.conj(), states_1))**2
    Sx_vals = vals(Sx, states_0).real
    Sx_Sx_vals = vals(Sx @ Sx, states_0).real
    Sz_Sz_vals = vals(Sz @ Sz, states_0).real

    # make time-series figures
    for tag, label, values in [ ("ss", r"\bm{s}_0\cdot\bm{s}_1", exchange),
                                ("sx", r"s_{\mathrm{x}}", Sx_vals),
                                ("sx_sx", r"s_{\mathrm{x}}^2", Sx_Sx_vals),
                                ("sz_sz", r"s_{\mathrm{z}}^2", Sz_Sz_vals) ]:
        plt.figure(figsize = figsize)
        plt.title(title)
        plt.plot(times, values, "k-")
        plt.xlabel(r"$tU$")
        plt.ylabel(wrap_label(label))
        plt.tight_layout()
        plt.savefig(time_series_name(file, tag))
        plt.close()

        if tag not in LTAs:
            LTAs[tag] = np.zeros(len(fields))
            LTA_info[tag] = ( LTA_name(file, tag), label )
        LTAs[tag][field_idx] = np.mean(values)

for tag, values in LTAs.items():
    fname, label = LTA_info[tag]
    plt.figure(figsize = figsize)
    plt.plot(fields, values, "ko")
    plt.xlabel(r"$\log_{10}(h/U)$")
    plt.ylabel(wrap_label(label, double_braket = True))
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
