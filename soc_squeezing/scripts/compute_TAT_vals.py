#!/usr/bin/env python3

import os, sys, scipy
import numpy as np

from dicke_methods import coherent_spin_state
from correlator_methods import compute_correlators, dec_mat_drive
from jump_methods import correlators_from_trajectories

if len(sys.argv[1:]) != 3:
    print(f"usage: {sys.argv[0]} method lattice_depth lattice_size")
    exit()

method = sys.argv[1]
assert(method in [ "exact", "jump" ])
assert(len(sys.argv[2]) == 3)
lattice_depth = float(sys.argv[2])
lattice_size = int(sys.argv[3])

data_dir = "../data/"
output_dir = data_dir + "TAT/"
file_name = "_".join(sys.argv[1:]) + ".txt"

lattice_dim = 2
confining_depth = 60 # recoil energies
excited_lifetime_SI = 10 # seconds
order_cap = 70
trajectories = 1000
time_steps = 100

recoil_energy_NU = 21801.397815091557
drive_mod_index_zy = 0.9057195866712102

spin_num = lattice_size**lattice_dim

def get_val_1D(depth, file_name):
    with open(data_dir + "J_0.txt", "r") as f:
        for line in f:
            if line[0] == "#": continue
            if float(line.split(",")[0]) == lattice_depth:
                return float(line.split(",")[-1])

def get_val_2D(depth, confinement, file_name):
    conf_idx = None
    with open(data_dir + f"U_int_{lattice_dim}D.txt", "r") as f:
        for line in f:
            if line[0] == "#": continue
            if conf_idx == None:
                conf_idx = [ int(x) for x in line[1:].split(",") ].index(confinement) + 1
                continue
            vals = [ float(x) for x in line.split(",") ]
            if vals[0] == depth:
                return vals[conf_idx]

J = get_val_1D(lattice_depth, "J_0.txt")
U = get_val_2D(lattice_depth, confining_depth, f"U_int_{lattice_dim}D.txt")
phi = get_val_2D(lattice_depth, confining_depth, f"phi_opt_{lattice_dim}D.txt")

if None in [ J, U, phi ]:
    print("could not find value for J, U, or phi... you should inquire")
    exit()

h_std = 2**(1+lattice_dim/2)*J*np.sin(phi/2)
chi = h_std**2 / U / (spin_num-1)

decay_rate_LU = 1/excited_lifetime_SI / recoil_energy_NU
decay_rate = decay_rate_LU / chi
dec_rates = [ (0, 0, decay_rate), (0, 0, 0) ]

times = np.linspace(0, 2, time_steps) * spin_num**(-2/3)

h_TAT = { (2,0,0) : +1/3,
          (0,0,2) : -1/3 }
init_state = "+X"
init_state_vec = coherent_spin_state([0,1,0], spin_num)
dec_mat_TAT = dec_mat_drive(scipy.special.jv(0,drive_mod_index_zy))

header = f"# lattice_dim: {lattice_dim}\n"
header += f"# confining depth (E_R): {confining_depth}\n"
if method == "exact":
    header += f"# order_cap: {order_cap}\n"
    op_vals = compute_correlators(spin_num, order_cap, times, init_state, h_TAT,
                                  dec_rates, dec_mat_TAT, return_derivs = True)
if method == "jump":
    header += f"# trajectories: {trajectories}\n"
    header += f"# time_steps: {time_steps}\n"
    op_vals = correlators_from_trajectories(spin_num, trajectories, times, init_state_vec,
                                            h_TAT, dec_rates, dec_mat_TAT)

if not os.path.isdir(output_dir): os.mkdir(output_dir)

with open(output_dir + file_name, "w") as f:
    f.write(header)
    f.write("# operators: " + " ".join([ str(op) for op, _ in op_vals.items() ]) + "\n")
    for _, vals in op_vals.items():
        f.write(" ".join([ str(val) for val in vals ]) + "\n")
