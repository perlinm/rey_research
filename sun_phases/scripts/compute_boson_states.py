#!/usr/bin/env python3

import os, sys, time
import numpy as np

from dicke_methods import spin_op_z_dicke
from boson_methods import polarized_state, field_tensor, evolve_mft, compute_mean_states

np.set_printoptions(linewidth = 200)

spin_dim = int(sys.argv[1])
spin_num = int(sys.argv[2])
log10_field = sys.argv[3]
init_state_str = sys.argv[4]

assert( spin_dim % 2 == 0 )
assert( spin_num % 2 == 0 )
assert( init_state_str in [ "X" ] )

sim_time = 10**4
time_step = 0.1
ivp_tolerance = 2.220446049250313e-14 # smallest value allowed

data_dir = f"../data/spin_bosons/"
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

data_file = data_dir \
          + f"states_{init_state_str}_d{spin_dim}_N{spin_num}_h{log10_field}.txt"

genesis = time.time()
####################

dim = int(spin_dim)
spin = (spin_dim-1)/2
log10_field = float(log10_field)
field_strength = 10**log10_field

# construct inhomogeneous magnetic field
sz = spin_op_z_dicke(dim-1).diagonal()
def spin_angle(qq):
    return 2*np.pi*(qq+1/2)/spin_num
field = [ field_strength * sz, np.sin(spin_angle(np.arange(spin_num))) ]

# construct initial state
init_state = polarized_state("+X", spin_dim, spin_num)

# simulate using boson MFT
times = np.linspace(0, sim_time, int(sim_time/time_step+1))
states = evolve_mft(init_state, times, field)

# compute mean spin state in the ensemble at each time point
mean_states = compute_mean_states(states)

# save simulation results
header = "time"
for mu, nu in zip(*np.triu_indices(spin_dim)):
    header += f", ({mu},{nu})"
data = np.hstack([ times[:,None], mean_states ])
np.savetxt(data_file, data, header = header)

print("runtime:", time.time()-genesis, "seconds")
