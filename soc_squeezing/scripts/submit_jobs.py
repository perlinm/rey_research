#!/usr/bin/env python3

import os, sys


if len(sys.argv[1:]) != 3:
    print(f"usage: {sys.argv[0]} lattice_depth lattice_size method")
    exit()

assert(len(sys.argv[1]) == 3)
lattice_depth = float(sys.argv[1])
lattice_size = int(sys.argv[2])
method = sys.argv[3]

spin_num = lattice_depth**2
jump = "jump"
exact = "exact"

assert(method in [ jump, exact ])

if method == jump:
    memory_cap = "1G"
if method == exact:
    memory_cap = "2G"

basename = "_".join(sys.argv[1:][::-1])

header = f"""
#!/bin/bash

#SBATCH --partition=jila
#SBATCH --mem={memory_cap}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=./logs/{basename}.o
#SBATCH --error=./logs/{basename}.e
#SBATCH --time=01:00:00

module load python3

python3 compute_TAT_vals.py {sys.argv[1]} {sys.argv[2]} {sys.argv[3]}
"""

with open(f"./logs/{basename}.sh", "w") as f:
    f.write(header)
