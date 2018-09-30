#!/usr/bin/env python3

##########################################################################################
# FILE CONTENTS:
# computes optimal OAT and TAT squeezing parameters as a function of
# primary lattice depth and either confining lattice depth, system size, or SOC angle
##########################################################################################

import sys, itertools
import numpy as np
import pandas as pd
import scipy.optimize as optimize

from dicke_methods import spin_op_vec_mat_dicke
from squeezing_methods import squeezing_OAT, \
    get_optima_diagonalization, get_optima_simulation

from sr87_olc_constants import recoil_energy_NU


methods = [ "OAT", "OAT_dec", "TAT" ]
dependent_variables = [ "L", "T", "phi" ]

lattice_dim = 2
method = "OAT"
dependent_variable = "T"
assert(method in methods)
assert(dependent_variable in dependent_variables)

data_dir = "../data/"
save_dir = data_dir + "optimization/"

default_confinement = 60
default_length_1D = 30
site_number = 1000

lengths_1D = np.arange(10,101,1).astype(int)
SOC_angles = np.linspace(0,1/10,101)[1:] * np.pi

excited_lifetime_SI = 10 # seconds; lifetime of excited state (from e --> g decay)
decay_rate = 1/excited_lifetime_SI / recoil_energy_NU

max_tau = 2 # maximum value of reduced time in OAT squeezing minimization
max_time = lambda N : max_tau * N**(-2/3)

# value of N at which we switch methods for determining optimal TAT parameters
N_crossover = 900

# read in tunneling rates and interaction energies
def pd_read_1D(fname):
    return pd.read_csv(fname, comment = "#", squeeze = True, header = None, index_col = 0)
def pd_read_2D(fname):
    return pd.read_csv(fname, comment = "#", header = 0, index_col = 0)
J_0 = pd_read_1D(data_dir + "J_0.txt")
J_T = pd_read_1D(data_dir + "J_T.txt")
U_int = pd_read_2D(data_dir + f"U_int_{lattice_dim}D.txt")
phi_opt = pd_read_2D(data_dir + f"phi_opt_{lattice_dim}D.txt")
depths = J_0.index
confinements = J_T.index

# use the closest lattive depth available for the default confinement
default_confinement = confinements[abs(confinements-default_confinement).argmin()]

# set file names
suffix = f"{method}_{dependent_variable}_{lattice_dim}D"
base_name = f"{save_dir}{{}}_{suffix}.txt"
sqz_opt_fname = base_name.format("sqz_opt")
t_opt_fname = base_name.format("t_opt")


##########################################################################################
# compute squeezing parameters
##########################################################################################

def h_std(V_0,phi):
    return 2**(1+lattice_dim/2) * J_0.at[V_0] * np.sin(phi/2)

# get optimal squeezing parameter and time as a function of
# lattice geometry, primary lattice depth, confining lattice depth, SOC angle,
# and squeezing method
def get_optima(length, V_0, V_T, phi, method,
               soc_field_variance = {}, optima_TAT = {}):
    N = length**lattice_dim

    if phi == None: phi = phi_opt.at[V_0,str(V_T)]

    try:
        soc_field_variance[V_0,phi]
    except:
        soc_field_variance[V_0,phi] = h_std(V_0,phi)**2
    chi = soc_field_variance[V_0,phi] / ( (N-1) * U_int.at[V_0,str(V_T)] )

    if method[4:] == "dec":
        dec_rates = (0, 0, decay_rate / chi)
    else:
        dec_rates = (0,0,0)

    # return optimum squeezing value and time for OAT or TAT appropriately
    if method[:3] == "OAT":
        def sqz_OAT_nval(chi_t):
            return -squeezing_OAT(N, chi_t, dec_rates)
        optimum_OAT = optimize.minimize_scalar(sqz_OAT_nval, method = "bounded",
                                               bounds = (0, max_time(N)))
        sqz_opt, t_opt = -optimum_OAT.fun, optimum_OAT.x

    if method[:3] == "TAT":
        if optima_TAT.get(N) is None:

            S_op_vec, SS_op_mat = spin_op_vec_mat_dicke(N)
            S_op_vec = [ X[::2,::2] for X in S_op_vec ]
            SS_op_mat = [ [ X[::2,::2] for X in XS ] for XS in SS_op_mat ]

            H_TAT = 1/3 * ( SS_op_mat[2][2] - SS_op_mat[1][1] ).real

            state_nZ = np.zeros(N//2+1)
            state_nZ[0] = 1

            if N < N_crossover:
                get_optima = get_optima_diagonalization
            else:
                get_optima = get_optima_simulation

            optima_TAT[N] = \
                get_optima(N, H_TAT, S_op_vec, SS_op_mat, state_nZ, max_time(N))

        sqz_opt, t_opt = optima_TAT[N]

    return sqz_opt, t_opt / chi

# construst lists of simulation parameters
if dependent_variable == "L":
    length_vals = lengths_1D
    dep_vals = length_vals
    def dep_var(L, V_0, V_D, phi): return L
else:
    length_vals = [ default_length_1D ]

if dependent_variable == "T":
    T_vals = confinements
    dep_vals = T_vals
    def dep_var(L, V_0, V_D, phi): return V_T
else:
    T_vals = [ default_confinement ]

if dependent_variable == "phi":
    phi_vals = SOC_angles
    dep_vals = phi_vals
    def dep_var(L, V_0, V_D, phi): return phi
else:
    phi_vals = [ None ]

# compute everything!
zero_frame = pd.DataFrame(data = np.zeros((depths.size,dep_vals.size)),
                          index = depths, columns = dep_vals)
sqz_opt = zero_frame.copy(deep = True)
t_opt = zero_frame.copy(deep = True)
for length, V_0, V_T, phi in itertools.product(length_vals, depths, T_vals, phi_vals):
    print(length, V_0, V_T, phi)
    dep_vals = V_0, dep_var(length, V_0, V_T, phi)
    sqz_opt.at[dep_vals], t_opt.at[dep_vals] = get_optima(length, V_0, V_T, phi, method)


##########################################################################################
# save squeezing parameters
##########################################################################################

# construct headers
header = "# first column = primary lattice depths\n"
if dependent_variable == "L":
    header += "# first row = lattice size\n"
    header += f"# confining depth: {default_confinement}\n"
if dependent_variable == "T":
    header += "# first row = confining lattice depths\n"
    header += f"# lattice length: {default_length_1D}\n"
if dependent_variable == "phi":
    header += "# first row = SOC angle / pi\n"
    header += f"# confining depth: {default_confinement}\n"
    header += f"# lattice length: {default_length_1D}\n"

header += "# all dimensionful values are in units with the recoil energy"
header += r" E_R \approx 3.47 x 2\pi kHz equal to 1" + "\n"

header_sqz = r"# squeezing given in decibels: -10 log10(\xi^2)" + "\n"

# save all data
data_fname_headers = [ [ t_opt, t_opt_fname, header ] ]
if method[4:] == "dec":
    data_fname_headers += [ [ sqz_opt, sqz_opt_fname, header + header_sqz ] ]
for data, fname, hdr in data_fname_headers:
    with open(fname, "w") as f:
        f.write(hdr)
    data.to_csv(fname, mode = "a")
