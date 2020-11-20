#!/usr/bin/env python3

import os, sys, time
import numpy as np

genesis = time.time()

# flags for what to compute:
CB = "CB" # classical error bound
QB = "QB" # quantum error bound
RE = "RE" # reconstruction error

compute = RE # default
for idx, arg in enumerate(sys.argv):
    if arg in [ CB, QB, RE ]:
        compute = arg
        sys.argv.remove(arg)

save_data = False

# get minimum / maximum qudit dimensions from command-line inputs
min_dim = int(sys.argv[1])
try: max_dim = int(sys.argv[2])
except: max_dim = min_dim

sample_cap = 1000 # maximum number of times we choose a random set of measurement axes
time_cap = 300 # maximum time to run per qubit dimension, in seconds

seed = 0 # set random number seed. change at your own peril
np.random.seed(seed)

# determine the directory and file for saving data
data_dir = "../data/qudit_tomo/"
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)
data_file = data_dir + f"times_{compute}_d{min_dim}-{max_dim}.txt"

##########################################################################################
# import qudit tomography methods

import qudit_tomo_methods as tomo

random_axes = tomo.random_axes
meas_mat, meas_scales = tomo.get_meas_methods(max_dim)

if compute == CB:
    error_scale = tomo.classical_error_scale
if compute == QB:
    error_scale = tomo.quantum_error_scale

inv_struct_bands = tomo.get_struct_method(max_dim)
random_state = tomo.random_state
recon_error = tomo.recon_error

##########################################################################################
# simulate!

if save_data:
    # initialize data file
    with open(data_file, "w") as file:
        file.write(f"# sample_cap: {sample_cap}\n")
        file.write(f"# time_cap: {time_cap} sec\n")
        file.write(f"# seed: {seed}\n")
        file.write(f"# dim, mean_time")
        if compute in [ CB, QB ]:
            file.write(f", error_scale")
        file.write(f"\n")

for dim in range(min_dim, max_dim+1):

    if compute in [ CB, QB ]:
        # initialize the smallest error scale we've seen
        min_error_scale = np.inf

    if compute == RE:
        # pre-compute diagonal bands of the "inverted" matrix of structure constants
        _inv_struct_bands = inv_struct_bands(dim)

    # record serial runtime for the simulations, excluding the pre-computations above
    start = time.time()

    # run simulations with random axes, states, etc.
    axis_num = 2*dim-1 # minimum number of axes
    for sample in range(sample_cap):
        _axes = random_axes(axis_num)

        if compute in [ CB, QB ]:
            _meas_scales = meas_scales(dim, _axes)
            rnd_error_scale = error_scale(_meas_scales)
            min_error_scale = min(min_error_scale, rnd_error_scale)

        if compute == RE:
            _state = random_state(dim)
            _meas_mat = lambda LL : meas_mat(LL, _axes)
            recon_error(_state, _meas_mat, _inv_struct_bands)

        if time.time() - start > time_cap: break

    # determine the average simulation time, and save/print updates
    mean_time = ( time.time() - start ) / (sample+1)
    update = f"{dim} {mean_time}"
    if compute in [ CB, QB ]:
        update += f" {min_error_scale}"

    print(update)
    if save_data:
        with open(data_file, "a") as file:
            file.write(update + "\n")
    sys.stdout.flush()

# print the total time it took to run everything
runtime = time.time() - genesis
runtime_text = f"total runtime: {runtime}"
print(runtime_text)
if save_data:
    with open(data_file, "a") as file:
        file.write(f"# {runtime_text}\n")
