#!/usr/bin/env python3

import os, glob, qutip
import numpy as np
import matplotlib.pyplot as plt

from dicke_methods import spin_op_vec_dicke, plot_dicke_state

data_dir = "../data/spin_pair/"
fig_dir = "../figures/spin_pair/spin_vecs/"

figsize = (3,3)

# set fonts and use latex packages
params = { "font.family" : "serif",
           "font.serif" : "Computer Modern",
           "text.usetex" : True,
           "font.size" : 10 }
plt.rcParams.update(params)

##################################################

def get_info(file, string = False):
    _, dim, log10_field = os.path.basename(file).split("_")
    dim = dim[1:]
    log10_field = log10_field[1:-4]
    if not string:
        dim = int(dim)
        log10_field = float(log10_field)
    return { "dim" : dim, "log10_field" : log10_field }

files = sorted(glob.glob(data_dir + "*.txt"),
               key = lambda file : list(get_info(file).values()))
dims = sorted(set( get_info(file)["dim"] for file in files ))

def spin_op_vec(dim):
    spin = (dim-1)/2
    sz, sx, sy = spin_op_vec_dicke(dim-1)
    return np.array([ sx.todense(), sy.todense(), sz.todense() ]) / spin
spin_op_vecs = { dim : spin_op_vec(dim) for dim in dims }

##################################################

def fig_name(file, aux_dir, tag):
    _fig_dir = fig_dir + aux_dir + "/"
    field_text = file.split("_")[-1][1:-4]
    if len(field_text) in [ 3, 4 ]:
        _fig_dir += "sweep/"
    else:
        _fig_dir += "zoom/"
    name = file.replace(data_dir + "states", _fig_dir + f"{tag}")
    return name.replace(".txt", ".png")

def spin_vec(state):
    return np.einsum("aij,i,j", spin_op_vecs[state.size], state.conj(), state).real

def make_bloch(spin_vecs, view = None):
    figure = plt.figure(figsize = figsize)
    bloch = qutip.Bloch(figure)
    bloch.add_points(spin_vecs.T)
    bloch.xlabel = [ "", "" ]
    bloch.ylabel = [ "", "" ]
    bloch.zlabel = [ "", "" ]
    if view: bloch.view = view
    return bloch

for file in files:
    print(file)
    dim = get_info(file)["dim"]
    spin = (dim-1)/2

    data = np.loadtxt(file, dtype = complex)
    states = data[:,1:dim+1]

    # plot mean state
    mean_state = np.einsum("ti,tj->tij", states, states.conj()).mean(axis = 0)
    plot_dicke_state(mean_state, single_sphere = False)
    plt.savefig(fig_name(file, "mean_states", "state"))
    plt.close()

    # plot mean spin vectors
    spin_vecs = np.einsum("aij,ti,tj->ta", spin_op_vecs[dim], states.conj(), states)
    bloch = make_bloch(spin_vecs)
    bloch.zlabel = [ r"$\log_{10}h=" + get_info(file, True)["log10_field"] + "$" , "" ]
    bloch.zlpos[0] = 1.4
    bloch.save(fig_name(file, "spin_vecs", "bloch"))
    plt.close()

    bloch = make_bloch(spin_vecs, [0,90])
    bloch.save(fig_name(file, "spin_vecs", "top"))
    plt.close()

    bloch = make_bloch(spin_vecs, [0,0])
    bloch.save(fig_name(file, "spin_vecs", "side"))
    plt.close()
