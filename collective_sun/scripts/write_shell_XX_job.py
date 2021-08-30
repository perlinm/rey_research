#!/usr/bin/env python3

import os, sys

basename = "shell_XX_" + "_".join(sys.argv[1:])

time = "1-00"
memory_cap = "10G"
cpus = 8

script = f"""#!/bin/sh

#SBATCH --mem={memory_cap}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --job-name={basename}
#SBATCH --output=./logs/{basename}.o
#SBATCH --error=./logs/{basename}.e
#SBATCH --time={time}

module load intelpython3

python3 shell_model_XX.py {" ".join(sys.argv[1:])}
"""

with open(f"./logs/{basename}.sh", "w") as f:
    f.write(script)
