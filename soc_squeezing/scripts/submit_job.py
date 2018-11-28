#!/usr/bin/env python3

import os, sys

if len(sys.argv[1:]) not in [ 3, 4 ]:
    print(f"usage: {sys.argv[0]} method lattice_depth lattice_size [rational]")
    exit()

method = sys.argv[1]
spin_num = int(sys.argv[3])**2
rational = ( len(sys.argv[1:]) == 4 )

if method == "jump":
    memory_cap = "4G"
    cores = 4
    time = "04-00"
elif method == "exact":
    memory_cap = "2G"
    cores = 1
    time = "01:00:00"
    if rational:
        time = "01-00"
        memory_cap = "4G"
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
#SBATCH --time={time}

module load python3

python3 compute_TAT_vals.py {" ".join(sys.argv[1:])}
"""

with open(f"./logs/{basename}.sh", "w") as f:
    f.write(header)
