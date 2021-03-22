#!/usr/bin/env python3

import os, sys, time
import numpy as np

from dicke_methods import spin_op_z_dicke
from boson_methods import polarized_state, field_tensor, evolve_mft, compute_mean_states

np.set_printoptions(linewidth = 200)

save_all_states = "all" in sys.argv
if save_all_states:
    sys.argv.remove("all")

init_state_str = sys.argv[1]
spin_dim = int(sys.argv[2])
spin_num = int(sys.argv[3])
log10_field = sys.argv[4]

assert( init_state_str in [ "X" ] )
assert( spin_dim % 2 == 0 )
assert( spin_num % 2 == 0 )

sim_time = 10**4
time_step = 0.1
ivp_tolerance = 2.220446049250313e-14 # smallest value allowed

data_dir = f"../data/spin_bosons/"
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

sys_tag = f"{init_state_str}_d{spin_dim:02d}_N{spin_num}_h{log10_field}"
def data_file(tag):
    return data_dir + f"{tag}_{sys_tag}.txt"

genesis = time.time()
####################

# construct inhomogeneous magnetic field
field = 10**float(log10_field)
sz = spin_op_z_dicke(spin_dim-1).diagonal()
def spin_coeff(qq):
    spin_angle = 2*np.pi*(qq+1/2)/spin_num
    return np.sin(spin_angle)
field_data = [ sz, field * spin_coeff(np.arange(spin_num)) ]

# construct initial state
init_state = polarized_state("+X", spin_dim, spin_num)

# simulate using boson MFT
times = np.linspace(0, sim_time, int(sim_time/time_step+1))
states = evolve_mft(init_state, times, field_data, ivp_tolerance = ivp_tolerance)

# compute mean spin state in the ensemble at each time point
mean_states = compute_mean_states(states)

# save simulation results
header = "time"
for mu, nu in zip(*np.triu_indices(spin_dim)):
    header += f", ({mu},{nu})"

if save_all_states:
    data = np.hstack([ times[:,None], mean_states ])
    np.savetxt(data_file("states"), data, header = header)

# save long-time average state
header = "sim_time, time_step: {sim_time}, {time_step}\n" + header
mean_state = np.mean(mean_states, axis = 0)
np.savetxt(data_file("mean_state"), mean_state, header = header)

print("runtime:", time.time()-genesis, "seconds")
