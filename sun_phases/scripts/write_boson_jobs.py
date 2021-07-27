#!/usr/bin/env python3

import os, sys

spin_num = 100

log_dir = "./logs/"

time = "07-00"
memory_cap = "10G"
cpus = 1

def write_job(dim, log10_field, state):
    basename = f"spin_bosons_{state}_d{dim}_N{spin_num}_h{log10_field}"
    script = f"""#!/bin/sh

#SBATCH --mem={memory_cap}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --job-name={basename}
#SBATCH --output={log_dir}{basename}.o
#SBATCH --error={log_dir}{basename}.e
#SBATCH --time={time}

module load intelpython3

python3 compute_boson_states.py {state} {dim} {spin_num} {log10_field}
"""

    with open(f"{log_dir}{basename}.sh", "w") as f:
        f.write(script)

for dim in range(2,11,2):

    for field_idx in range(-10,11):
        log10_field = field_idx / 10
        write_job(f"{dim:02d}", f"{log10_field:.1f}", "X")
        if dim > 2:
            write_job(f"{dim:02d}", f"{log10_field:.1f}", "XX")
            write_job(f"{dim:02d}", f"{log10_field:.1f}", "XXI")

        if field_idx % 10 == 0: continue

        trans = {   2 :  0.0,
                    4 : -0.1,
                    6 : -0.1,
                    8 : -0.2,
                   10 : -0.2 }
        log10_field = trans[dim] + field_idx / 100
        write_job(f"{dim:02d}", f"{log10_field:.2f}", "X")
        if dim > 2:
            write_job(f"{dim:02d}", f"{log10_field:.1f}", "XX")
            write_job(f"{dim:02d}", f"{log10_field:.1f}", "XXI")
