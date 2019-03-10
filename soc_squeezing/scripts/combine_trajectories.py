#!/usr/bin/env python3

import os, glob, re
import numpy as np

from squeezing_methods import squeezing_from_correlators

log10_N = 4
method = "TNT"

files = glob.glob(f"../data/squeezing/jumps/sqz_D_exact_logN{log10_N}_{method}_s???.txt")
data = sum([ np.loadtxt(f, dtype = complex, delimiter = ",")
             for f in files ]) / len(files)

sqz_name = os.path.basename(re.sub(r"_s[0-9]{3}", "", files[0]))
sqz_path = os.path.dirname(files[0]) + "/../" + sqz_name

header = ""
header += r"# first column: time in units of the OAT stregth \chi" + "\n"
header += r"# second column: squeezing as \xi^2" + "\n"

with open(files[0], "r") as f:
    for line in f:
        if "trajectories" in line:
            trajectories_per_run = int(line.split()[-1])
        elif "max_tau" in line:
            max_tau = int(line.split()[-1])
        elif "time_steps" in line:
            time_steps = int(line.split()[-1])
        elif "ops" in line:
            nums = np.array([ int(nn) for nn in re.findall(r"[0-9]+", line) ])
            nums = nums.reshape((nums.size//3,3))
            ops = [ tuple(nums[jj,:]) for jj in range(nums.shape[0]) ]
        elif line[0] != "#": break

header += f"# trajectories: {len(files)*trajectories_per_run}\n"

N = 10**log10_N
max_time = max_tau * N**(-2/3)
times = np.linspace(0, max_time, time_steps)

correlators = { ops[jj] : data[jj,:] for jj in range(len(ops)) }
sqz = squeezing_from_correlators(N, correlators)

with open(sqz_path, "w") as f:
    f.write(header)
    for tt in range(times.size):
        f.write(f"{times[tt]},{sqz[tt]}\n")
