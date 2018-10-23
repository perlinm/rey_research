#!/usr/bin/env python3

import os, sys, scipy
import numpy as np
import pandas as pd

from dicke_methods import coherent_spin_state
from correlator_methods import compute_correlators, dec_mat_drive
from jump_methods import correlators_from_trajectories

if len(sys.argv[1:]) != 3:
    print(f"usage: {sys.argv[0]} lattice_depth lattice_size method")
    exit()

assert(len(sys.argv[1]) == 3)
lattice_depth = float(sys.argv[1])
lattice_size = int(sys.argv[2])
method = sys.argv[3]
assert(method in [ "exact", "jump" ])
use_exact = method == "exact"

data_dir = "../data/"
output_dir = data_dir + "TAT/"

lattice_dim = 2
confining_depth = 60 # recoil energies
excited_lifetime_SI = 10 # seconds
order_cap = 60
trajectories = 1000
time_steps = 100

recoil_energy_NU = 21801.397815091557
drive_mod_index_zy = 0.9057195866712102

spin_num = lattice_size**lattice_dim

def pd_read_1D(fname):
    return pd.read_csv(fname, comment = "#", squeeze = True, header = None, index_col = 0)
def pd_read_2D(fname):
    return pd.read_csv(fname, comment = "#", header = 0, index_col = 0)
J = pd_read_1D(data_dir + "J_0.txt")[lattice_depth]
U, phi = [ pd_read_2D(data_dir + fname).at[lattice_depth,str(confining_depth)]
           for fname in [ f"U_int_{lattice_dim}D.txt", f"phi_opt_{lattice_dim}D.txt" ] ]

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

if use_exact:
    op_vals = compute_correlators(spin_num, order_cap, times, init_state, h_TAT,
                                  dec_rates, dec_mat_TAT, return_derivs = True)
else:
    op_vals = correlators_from_trajectories(spin_num, trajectories, times, init_state_vec,
                                            h_TAT, dec_rates, dec_mat_TAT)

if not os.path.isdir(output_dir): os.mkdir(output_dir)

with open(output_dir + "_".join(sys.argv[1:][::-1]) + ".txt", "w") as f:
    ops = [ str(op) for op, _ in op_vals.items() ]
    f.write("# operators: " + " ".join(ops) + "\n")
    for _, vals in op_vals.items():
        f.write(" ".join([ str(val) for val in vals ]) + "\n")
