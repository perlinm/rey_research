#!/usr/bin/env python3

import os, sys

if len(sys.argv[1:]) != 3:
    print(f"usage: {sys.argv[0]} method lattice_depth lattice_size")
    exit()

method = sys.argv[1]

spin_num = int(sys.argv[3])**2

if method == "jump":
    memory_cap = "1G"
    cores = 4
elif method == "exact":
    memory_cap = "2G"
    cores = 1
else:
    print("method must be one of 'jump' or 'exact'")
    exit()

basename = "_".join(sys.argv[1:])

header = f"""#!/bin/sh

#SBATCH --partition=nistq,jila
#SBATCH --mem={memory_cap}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cores}
#SBATCH --job-name={basename}
#SBATCH --output=./logs/{basename}.o
#SBATCH --error=./logs/{basename}.e
#SBATCH --time=01:00:00

module load python3

python3 compute_TAT_vals.py {sys.argv[1]} {sys.argv[2]} {sys.argv[3]}
"""

with open(f"./logs/{basename}.sh", "w") as f:
    f.write(header)
