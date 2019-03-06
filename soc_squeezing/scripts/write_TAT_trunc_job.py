#!/usr/bin/env python3

import os, sys

if len(sys.argv[1:]) not in [ 3, 4 ]:
    print(f"usage: {sys.argv[0]} lattice_depth lattice_size [rational]")
    exit()

method = sys.argv[1]
spin_num = int(sys.argv[2])**2
rational = ( len(sys.argv[1:]) == 3 )

if not rational:
    time = "01:00:00"
    memory_cap = "2G"
else:
    time = "01-00"
    memory_cap = "4G"

basename = "_".join(sys.argv[1:])

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

python3 compute_TAT_vals.py {" ".join(sys.argv[1:])}
"""

with open(f"./logs/trunc_{basename}.sh", "w") as f:
    f.write(header)
