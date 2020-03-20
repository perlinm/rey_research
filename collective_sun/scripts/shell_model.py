#!/usr/bin/env python3

import numpy as np
import os, sys, itertools, functools

from squeezing_methods import spin_squeezing
from dicke_methods import coherent_spin_state as coherent_state_PS
from multibody_methods import dist_method, get_multibody_states
from operator_product_methods import build_shell_operator

np.set_printoptions(linewidth = 200)
cutoff = 1e-10

if len(sys.argv) < 4:
    print(f"usage: {sys.argv[0]} [alpha] [max_manifold] [lattice_shape]")
    exit()

alpha = float(sys.argv[1]) # power-law couplings ~ 1 / r^\alpha
max_manifold = int(sys.argv[2])
lattice_shape = tuple(map(int, sys.argv[3:]))

# values of the ZZ coupling to inspect more closely
inspect_coupling_zz = [ -1, 0, 0.5, 1.5 ]
inspect_sim_time = 2

# values of the ZZ coupling to simulate in an XXZ model
sweep_coupling_zz = np.linspace(-1,3,41)

max_time = 10 # in units of J_\perp

plot_all_shells = False # plot the population for each shell?

ivp_tolerance = 1e-10 # error tolerance in the numerical integrator

data_dir = "../data/shells/"

def name_tag(coupling_zz = None):
    lattice_name = "_".join([ str(size) for size in lattice_shape ])
    base_tag = f"L{lattice_name}_M{max_manifold}_a{alpha}"
    if coupling_zz == None: return base_tag
    else: return base_tag + f"_z{coupling_zz}"

##################################################

lattice_dim = len(lattice_shape)
spin_num = np.product(lattice_shape)
if np.allclose(alpha, int(alpha)): alpha = int(alpha)

print("lattice shape:",lattice_shape)
##########################################################################################
# build SU(n) interaction matrix, and build generators of interaction eigenstates
##########################################################################################
print("builing generators of interaction energy eigenstates")

dist = dist_method(lattice_shape)

sunc = {}
sunc["mat"] = np.zeros((spin_num,spin_num))
for pp, qq in itertools.combinations(range(spin_num),2):
    sunc["mat"][pp,qq] = sunc["mat"][qq,pp] = -1/dist(pp,qq)**alpha
sunc["TI"] = True

# build generators of interaction eigenstates, compute energies, etc.
sunc["shells"], sunc["energies"], sunc_tensors \
    = get_multibody_states(lattice_shape, sunc["mat"], max_manifold, sunc["TI"])
sunc.update(sunc_tensors)
shell_num = len(sunc["energies"])

for manifold, shells in list(sunc["shells"].items()):
    if len(shells) == 0:
        del sunc["shells"][manifold]

##########################################################################################
# compute states and operators in the shell / Z-projection basis
##########################################################################################
print("building operators in the shell / Z-projection basis")
sys.stdout.flush()

# 1-local Z, X, Y operators
local_ops = { "Z" : np.array([[ 1,   0 ], [  0, -1 ]]),
              "X" : np.array([[ 0,   1 ], [  1,  0 ]]),
              "Y" : np.array([[ 0, -1j ], [ 1j,  0 ]]) }

# 2-local products of Z, X, Y operators
for op_lft, op_rht in itertools.product(local_ops.keys(), repeat = 2):
    mat_lft = local_ops[op_lft]
    mat_rht = local_ops[op_rht]
    local_ops[op_lft + op_rht] = np.kron(mat_lft,mat_rht).reshape((2,)*4)

# build the ZZ perturbation operator in the shell / Z-projection basis
print("building perturbation operator")
shell_coupling_mat \
    = build_shell_operator([sunc["mat"]], [local_ops["ZZ"]], sunc, sunc["TI"])

# collective spin vector, and its outer product with itself
def _pauli_mat(pauli):
    tensors = [np.ones(spin_num)]
    operators = [local_ops[pauli]]
    full_pauli_op = build_shell_operator(tensors, operators, sunc,
                                         sunc["TI"], shell_diagonal = True)
    full_pauli_op.shape = ( shell_num*(spin_num+1), )*2
    return full_pauli_op
print("building collective spin operators")
sys.stdout.flush()
S_op_vec = [ _pauli_mat(pauli)/2 for pauli in [ "Z", "X", "Y" ] ]
SS_op_mat = [ [ AA @ BB for BB in S_op_vec ] for AA in S_op_vec ]

# energies and energy eigenstates within each sector of fixed spin projection
def energies_states(zz_sun_ratio):
    energies = np.zeros( ( shell_num, spin_num+1 ) )
    eig_states = np.zeros( ( shell_num, shell_num, spin_num+1 ), dtype = complex )

    for spins_up in range(spin_num+1):
        # construct the Hamiltonian at this Z projection, from SU(n) + ZZ couplings
        _proj_hamiltonian = np.diag(sunc["energies"]) \
                          + zz_sun_ratio * shell_coupling_mat[:,spins_up,:,spins_up]

        # diagonalize the net Hamiltonian at this Z projection
        energies[:,spins_up], eig_states[:,:,spins_up] = np.linalg.eigh(_proj_hamiltonian)

        spins_dn = spin_num - spins_up
        if spins_up == spins_dn: break
        energies[:,spins_dn], eig_states[:,:,spins_dn] \
            = energies[:,spins_up], eig_states[:,:,spins_up]

    return energies, eig_states

# coherent spin state
def coherent_spin_state(vec):
    zero_shell = np.zeros(shell_num)
    zero_shell[0] = 1
    return np.outer(zero_shell, coherent_state_PS(vec,spin_num))

##########################################################################################
# simulate!
##########################################################################################

# note: extra factor of 1/2 in che_eff_bare for compatibility with Chunlei's work
chi_eff_bare = 1/4 * sunc["mat"].sum() / np.math.comb(spin_num,2)
state_X = coherent_spin_state([0,1,0])

def _states(initial_state, zz_sun_ratio, times):
    energies, eig_states = energies_states(zz_sun_ratio)
    init_state_eig = np.einsum("Ssz,Sz->sz", eig_states, initial_state)
    phases = np.exp(-1j * np.tensordot(times, energies, axes = 0))
    evolved_eig_states = phases * init_state_eig[None,:,:]
    return np.einsum("sSz,tSz->tsz", eig_states, evolved_eig_states)

def simulate(coupling_zz, sim_time = None, max_tau = 2, points = 500):
    print("coupling_zz:", coupling_zz)
    sys.stdout.flush()
    zz_sun_ratio = coupling_zz - 1
    assert(zz_sun_ratio != 0)

    # determine how long to simulate
    if sim_time is None:
        chi_eff = abs(zz_sun_ratio * chi_eff_bare)
        sim_time = min(max_time, max_tau * spin_num**(-2/3) / chi_eff)

    times = np.linspace(0, sim_time, points)

    # compute states at all times of interest
    # note: factor of 1/2 included for compatibility with Chunlei's work
    states = _states(state_X, zz_sun_ratio/2, times)

    # compute squeezing
    sqz = np.array([ spin_squeezing(spin_num, state.flatten(), S_op_vec, SS_op_mat)
                     for state in states ])

    # compute populations
    pops = np.einsum("tsz->ts", abs(states)**2)

    return times, sqz, pops

if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

##########################################################################################
print("running inspection simulations")
sys.stdout.flush()

for coupling_zz in inspect_coupling_zz:
    times, sqz, pops = simulate(coupling_zz, sim_time = inspect_sim_time)

    with open(data_dir + f"inspect_{name_tag(coupling_zz)}.txt", "w") as file:
        file.write("# times, squeezing, populations (within each shell)\n")
        for manifold, shells in sunc["shells"].items():
            file.write(f"# manifold {manifold} : ")
            file.write(" ".join([ str(shell) for shell in shells ]))
            file.write("\n")
        for tt in range(len(times)):
            file.write(f"{times[tt]} {sqz[tt]} ")
            file.write(" ".join([ str(pop) for pop in pops[tt,:] ]))
            file.write("\n")

##########################################################################################
if len(sweep_coupling_zz) == 0: exit()
print("running sweep simulations")
sys.stdout.flush()

sweep_coupling_zz = sweep_coupling_zz[sweep_coupling_zz != 1]
sweep_results = [ simulate(coupling_zz) for coupling_zz in sweep_coupling_zz ]
sweep_times, sweep_sqz, sweep_pops = zip(*sweep_results)

sweep_min_sqz = [ min(sqz) for sqz in sweep_sqz ]
min_sqz_idx = [ max(1,np.argmin(sqz)) for sqz in sweep_sqz ]

sweep_pops = [ np.array([ pops[:min_idx,shells].sum(axis = 1)
                          for shells in sunc["shells"].values() ]).T
               for pops, min_idx in zip(sweep_pops, min_sqz_idx) ]
sweep_min_pops = np.array([ pops.min(axis = 0) for pops in sweep_pops ])
sweep_max_pops = np.array([ pops.max(axis = 0) for pops in sweep_pops ])

with open(data_dir + f"sweep_{name_tag()}.txt", "w") as file:
    file.write("# coupling_zz, sqz_min, min_pop_0, max_pop (for manifolds > 0)\n")
    file.write("# manifolds : ")
    file.write(" ".join([ str(manifold) for manifold in sunc["shells"].keys() ]))
    file.write("\n")
    for zz in range(len(sweep_coupling_zz)):
        file.write(f"{sweep_coupling_zz[zz]} {sweep_min_sqz[zz]} {sweep_min_pops[zz,0]} ")
        file.write(" ".join([ str(val) for val in sweep_max_pops[zz,1:] ]))
        file.write("\n")

print("completed")
