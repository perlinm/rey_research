#!/usr/bin/env python3

import os, sys, time, scipy
import numpy as np

from dicke_methods import spin_op_z_dicke, spin_op_y_dicke, coherent_spin_state
from boson_methods import polarized_state, boson_mft_state, evolve_mft, \
    compute_avg_var_vals

init_state_str = sys.argv[1]
spin_dim = int(sys.argv[2])
spin_num = int(sys.argv[3])
log10_field = sys.argv[4]

assert( init_state_str in [ "X", "XX" ] )
assert( spin_dim % 2 == 0 )
assert( spin_num % 2 == 0 )

sim_time = 10**5
time_step = 0.1
save_points = 100
ivp_tolerance = 1e-10

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
if init_state_str == "X":
    init_state = polarized_state("+X", spin_dim, spin_num)
elif init_state_str == "XX":
    sy = spin_op_y_dicke(spin_dim-1)
    state_z = coherent_spin_state("+Z", spin_dim-1)
    state_xx = scipy.sparse.linalg.expm_multiply(-1j * np.pi/2 * sy @ sy, state_z)
    init_state = boson_mft_state(state_xx, spin_dim, spin_num)

prev_state = init_state # state at last simulated time
prev_mean = 0 # running time-averaged state
prev_steps = 0 # running number of time points

# determine simulation time points
sim_times = np.linspace(0, sim_time, int(sim_time/time_step+1))
time_indices = np.arange(0, sim_times.size, sim_times.size//save_points)
if time_indices[-1] != sim_times.size-1:
    time_indices.append(-1)

# given list [ a, b, c, ... ], iterate over [ (a,b), (b,c), ... ]
def adjacent_pairs(items):
    yield from zip(items[:-1],items[1:])

# simulate using boson mean-field theory
for save_point, ( idx_fst, idx_snd ) in enumerate(adjacent_pairs(time_indices),1):
    times = sim_times[idx_fst:idx_snd+1]
    states = evolve_mft(prev_state, times, field_data, ivp_tolerance = ivp_tolerance)

    # compute spacetime average of states \rho and their second moments \rho\otimes\rho
    this_mean = compute_avg_var_vals(states[1:,:,:])
    this_steps = times.size-1 # number of averaged time points

    # compute and save running average
    prev_ratio = prev_steps / ( prev_steps + this_steps )
    this_ratio = this_steps / ( prev_steps + this_steps )
    save_mean = prev_mean * prev_ratio + this_mean * this_ratio
    header = f"sim_time, time_step: {times[-1]}, {time_step}"
    np.savetxt(data_file("means"), save_mean, header = header)

    # update running state and number of time points
    prev_mean = save_mean
    prev_steps += this_steps
    prev_state = states[-1,:,:]
    print(f"save {save_point}/{time_indices.size-1}:", time.time()-genesis, "seconds")
