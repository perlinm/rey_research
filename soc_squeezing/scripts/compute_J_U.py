#!/usr/bin/env python3

# FILE CONTENTS: computes tunneling rates and interaction energies

import numpy as np
import pandas as pd
import itertools

from mathieu_methods import mathieu_solution
from overlap_methods import tunneling_1D, pair_overlap_1D
from sr87_olc_constants import g_int_LU

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

##########################################################################################
# compute tunneling rates and interaction energies
##########################################################################################

J_0 = pd.Series(data = np.zeros(shallow.size), index = shallow)
K_0 = pd.Series(data = np.zeros(shallow.size), index = shallow)

J_T = pd.Series(data = np.zeros(deep.size), index = deep)
K_T = pd.Series(data = np.zeros(deep.size), index = deep)

zero_frame = pd.DataFrame(data = np.zeros((shallow.size,deep.size)),
                          index = shallow, columns = deep)
U_int_1D = zero_frame.copy(deep = True)
U_int_2D = zero_frame.copy(deep = True)

for depth in shallow:
    momenta, fourier_vecs, energies = mathieu_solution(depth, bands, site_number)
    J_0.at[depth] = tunneling_1D(depth, momenta, fourier_vecs)
    K_0.at[depth] = pair_overlap_1D(momenta, fourier_vecs)
for depth in deep:
    momenta, fourier_vecs, energies = mathieu_solution(depth, bands, site_number)
    J_T.at[depth] = tunneling_1D(depth, momenta, fourier_vecs)
    K_T.at[depth] = pair_overlap_1D(momenta, fourier_vecs)

for V_0, V_T in itertools.product(shallow, deep):
    U_int_1D.at[V_0,V_T] = g_int_LU[1] * K_T.at[V_T]**2 * K_0.at[V_0]
    U_int_2D.at[V_0,V_T] = g_int_LU[1] * K_T.at[V_T] * K_0.at[V_0]**2


##########################################################################################
# save data to files
##########################################################################################

header_J = "# first column = lattice depth\n"
header_U = "# first column = primary lattice depth\n"
header_U += "# first row = confining lattice depth\n"
header_units = "# values in units with the recoil energy"
header_units += r" E_R \approx 3.47 x 2\pi kHz equal to 1" + "\n"

for data, name, header_JU in [ [ J_0, "J_0", header_J ],
                               [ J_0, "J_0", header_J ],
                               [ U_int_1D, "U_int_1D", header_U ],
                               [ U_int_2D, "U_int_2D", header_U ] ]:
    file_name = data_dir + name + ".txt"
    with open(file_name, "w") as f:
        f.write(header_JU + header_units)
    data.to_csv(file_name, mode = "a")


