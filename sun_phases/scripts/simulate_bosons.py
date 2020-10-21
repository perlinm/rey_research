#!/usr/bin/env python3

import os, sys, time
import numpy as np

from dicke_methods import spin_op_z_dicke
from boson_methods import polarized_state, spin_state, boson_mft_state, \
    field_tensor, bare_lattice_field, evolve_mft, compute_mean_states

np.set_printoptions(linewidth = 200)

spin_dim = int(sys.argv[1])
spin_num = int(sys.argv[2])
init_state_str = sys.argv[3]

assert( spin_dim % 2 == 0 )
assert( spin_num % 2 == 0 )
assert( init_state_str in [ "X-L", "XX-L", "P-Z" ] )

# simulation parameters
log10_field_vals = np.arange(-2,1.01,0.5)
angle_frac_vals = np.array([1,0])
periods = 1000

data_dir = "../data/oscillations/"
sys_tag = f"n{spin_dim}_N{spin_num}_{init_state_str}"

if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

##########################################################################################
# set up objects for simulation
##########################################################################################

# initialize two big spins pointing along the equator with a given opening angle
def split_state(angle):
    xx = np.cos(angle/2)
    yy = np.sin(angle/2)
    spin_states = [ spin_state([0,xx,+yy], spin_dim) ] * (spin_num//2) \
                + [ spin_state([0,xx,-yy], spin_dim) ] * (spin_num//2)
    return boson_mft_state(spin_states, spin_dim, spin_num)

# field, initial state, and mean state for a "lattice" field
if init_state_str[-1] == "L":

    def build_field(angle):
        return field_tensor([ 2 * bare_lattice_field(qq, angle, spin_dim, spin_num)
                              for qq in range(spin_num) ])

    def build_init_state(_):
        if init_state_str == "X-L":
            return split_state(0)
        if init_state_str == "XX-L":
            return split_state(np.pi)

    def get_mean_states(states):
        return compute_mean_states(states)

# field, initial state, and mean state for two big spins
if init_state_str == "P-Z":

    Sz = spin_op_z_dicke(spin_dim-1).todense()
    def build_field(_):
        return field_tensor([ +Sz ] * (spin_num//2) +
                            [ -Sz ] * (spin_num//2))

    build_init_state = split_state

    def get_mean_states(states):
        states_fst = states[:,:,:spin_num//2]
        states_snd = states[:,:,spin_num//2:]
        mean_states_fst = compute_mean_states(states[:,:,:spin_num//2])
        mean_states_snd = compute_mean_states(states[:,:,spin_num//2:])
        return np.hstack([ mean_states_fst, mean_states_snd ])

# determine spectral range of a single-particle field
def _range(vals):
    return max(vals) - min(vals)
def spectral_range(field):
    return max([ _range(np.linalg.eigvalsh(field[:,:,qq]))
                for qq in range(spin_num) ])

##########################################################################################
# simulate!
##########################################################################################
sim_start = time.time()

for idx_soc, angle_frac in enumerate(angle_frac_vals):
    angle_tag = f"{angle_frac:.2f}"
    angle = angle_frac * np.pi

    bare_field = build_field(angle)
    init_state = build_init_state(angle)

    bare_field_strength = spectral_range(bare_field)

    for idx_field, log10_field in enumerate(log10_field_vals):
        field_tag = f"{log10_field:.2f}"
        file_tag = f"{sys_tag}_h{field_tag}_a{angle_tag}"

        print(f"{idx_soc}/{len(angle_frac_vals)} " + \
              f"{idx_field}/{len(log10_field_vals)} " + \
              f"(angle_frac = {angle_tag}, log10_field = {field_tag})", end = "")
        sys.stdout.flush()

        this_start = time.time()

        # set single-particle field
        field_strength = 10**log10_field
        field = field_strength * bare_field

        # determine simulation time based on estimated "field frequency"
        field_freq = 2 * field_strength * bare_field_strength
        sim_time = 2*np.pi / np.sqrt(1 + field_freq)**2 * periods

        # simulate
        times, states = evolve_mft(init_state, sim_time, field)

        # save simulation results
        mean_states = get_mean_states(states)
        np.savetxt(data_dir + f"times_{file_tag}.txt", times)
        np.savetxt(data_dir + f"mean_states_{file_tag}.txt", mean_states.ravel())

        this_runtime = int( time.time() - this_start )
        print(f" {this_runtime} sec")

sim_runtime = int( time.time() - sim_start )
seconds = sim_runtime % 60
minutes = sim_runtime // 60 % 60
hours = sim_runtime // 60**2
print(f"total runtime: {hours:02d}:{minutes:02d}:{seconds:02d}")
