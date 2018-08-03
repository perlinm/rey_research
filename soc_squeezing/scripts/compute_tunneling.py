#!/usr/bin/env python3

# FILE CONTENTS: computes tunneling rate at various lattice depths

import numpy as np
import pandas as pd

from mathieu_methods import mathieu_solution
from overlap_methods import tunneling_1D

data_dir = "../data/"

site_number = 100
bands = 1

shallow_min = 1
shallow_max = 20
shallow_step = 0.1

deep_min = 40
deep_max = 200
deep_step = 1

shallow = np.arange(shallow_min, shallow_max + shallow_step/2, shallow_step)
deep = np.arange(deep_min, deep_max + deep_step/2, deep_step, dtype = int)

J_0 = pd.DataFrame(data = np.zeros(len(shallow)), index = shallow)
J_T = pd.DataFrame(data = np.zeros(len(deep)), index = deep)

for depth in shallow:
    momenta, fourier_vecs, energies = mathieu_solution(depth, bands, site_number)
    J_0.at[depth] = tunneling_1D(depth, momenta, fourier_vecs)
for depth in deep:
    momenta, fourier_vecs, energies = mathieu_solution(depth, bands, site_number)
    J_T.at[depth] = tunneling_1D(depth, momenta, fourier_vecs)

header = "# first column = lattice depth\n"
header += "# values in units with the recoil energy"
header += r" E_R \approx 3.47 x 2\pi kHz equal to 1" + "\n"

for data, name in [ [ J_0, "J_0" ],
                    [ J_T, "J_T" ] ]:
    file_name = data_dir + name + ".txt"
    with open(file_name, "w") as f:
        f.write(header)
    data.to_csv(file_name, header = False, mode = "a")

