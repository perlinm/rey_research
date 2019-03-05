#!/usr/bin/env python3

import sys
import numpy as np

from dicke_methods import coherent_spin_state
from correlator_methods import compute_correlators, convert_zxy_mat, convert_zxy
from jump_methods import correlators_from_trajectories

assert(len(sys.argv[1:]) in [ 2, 3 ])
method = sys.argv[1]
seed = int(sys.argv[2])

TAT, TNT = "TAT", "TNT"
if method == TAT:
    memory = "5G"
elif method == TNT:
    memory = "25G"
else:
    print("method must be one of TAT or TNT")
    exit()

log10_N = 4
N = 10**log10_N
trajectories = 100

log_dir = "./logs/"
data_dir = "../data/squeezing/jumps/"
job_name = f"sqz_D_exact_logN{log10_N}_{method}_s{seed:03d}"

log_text = f"""#!/bin/sh

#SBATCH --partition=nistq,jila
#SBATCH --mem={memory}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}{job_name}.o
#SBATCH --error={log_dir}{job_name}.e
#SBATCH --time=05-00

module load python3

python3 {sys.argv[0]} {method} {seed}
"""

if len(sys.argv[1:]) == 3:
    with open(log_dir + job_name + ".sh", "w") as f:
        f.write(log_text)
    exit()

max_tau = 1
time_steps = 100
dec_rates = [ (1,1,1), (0,0,0) ]

init_state = "-Z"
h_TAT = { (0,2,0) : +1/3,
          (0,0,2) : -1/3 }
h_TNT = { (0,2,0) : 1,
          (1,0,0) : -N/2 }
h_vec = { TAT : h_TAT, TNT : h_TNT }

dec_mat_zxy = np.array([ [ 0, -1, 0 ],
                         [ 1,  0, 0 ],
                         [ 0,  0, 1 ]])
dec_mat = convert_zxy_mat(dec_mat_zxy)

max_time = max_tau * N**(-2/3)
times = np.linspace(0, max_time, time_steps)

init_state_vec = coherent_spin_state(init_state, N)
def jump_args(hamiltonian):
    return [ N, trajectories, times, init_state_vec, hamiltonian, dec_rates, dec_mat ]

correlators = correlators_from_trajectories(*jump_args(h_vec[method]), seed = seed)
with open(data_dir + job_name + ".txt", "w") as f:
    f.write(f"# trajectories: {trajectories}\n")
    f.write(f"# max_tau: {max_tau}\n")
    ops = list(correlators.keys())
    f.write("# ops: " + " ".join([ str(op) for op in ops]) + "\n")
    for op in ops:
        f.write(",".join([ str(val) for val in correlators[op] ]) + "\n")

