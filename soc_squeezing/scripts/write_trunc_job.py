#!/usr/bin/env python3

import os, sys

if len(sys.argv[1:]) not in [ 3, 4 ]:
    print(f"usage: {sys.argv[0]} method lattice_depth lattice_size [rational]")
    exit()

method = sys.argv[1]
rational_correlators = ( len(sys.argv[1:]) == 4 )

if not rational_correlators:
    time = "01:00:00"
    memory_cap = "2G"
else:
    time = "01-00"
    memory_cap = "4G"

basename = "_".join(["trunc"]+sys.argv[1:])

header = f"""#!/bin/sh

#SBATCH --partition=nistq,jila
#SBATCH --mem={memory_cap}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name={basename}
#SBATCH --output=./logs/{basename}.o
#SBATCH --error=./logs/{basename}.e
#SBATCH --time={time}

module load python3

python3 compute_trunc_vals.py {" ".join(sys.argv[1:])}
"""

with open(f"./logs/{basename}.sh", "w") as f:
    f.write(header)
