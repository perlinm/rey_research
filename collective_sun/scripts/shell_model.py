#!/usr/bin/env python3

import numpy as np
import os, sys, itertools, functools, time

from squeezing_methods import squeezing_from_correlators
from dicke_methods import coherent_spin_state as coherent_state_PS
from multibody_methods import dist_method, get_multibody_states
from operator_product_methods import build_shell_operator

np.set_printoptions(linewidth = 200)
cutoff = 1e-10

if len(sys.argv) < 4:
    print(f"usage: {sys.argv[0]} [test?] [lattice_shape] [alpha] [max_manifold]")
    exit()

# determine whether this is a test run
if "test" in sys.argv:
    test_run = True
    sys.argv.remove("test")
else:
    test_run = False

lattice_shape = tuple(map(int, sys.argv[1].split("x")))
alpha = float(sys.argv[2]) # power-law couplings ~ 1 / r^\alpha
max_manifold = int(sys.argv[3])

# values of the ZZ coupling to simulate in an XXZ model
sweep_coupling_zz = np.arange(-3, +4.01, 0.1)

# values of the ZZ coupling to inspect more closely: half-integer values
inspect_coupling_zz = [ coupling for coupling in sweep_coupling_zz
                        if np.allclose(coupling % 0.5, 0) ]
inspect_sim_time = 2

max_time = 10 # in units of J_\perp
ivp_tolerance = 1e-10 # error tolerance in the numerical integrator

data_dir = "../data/shells/"

if np.allclose(alpha, int(alpha)): alpha = int(alpha)
lattice_name = "x".join([ str(size) for size in lattice_shape ])
name_tag = f"L{lattice_name}_a{alpha}_M{max_manifold}"

##################################################

lattice_dim = len(lattice_shape)
spin_num = np.product(lattice_shape)
if np.allclose(alpha, int(alpha)): alpha = int(alpha)

start_time = time.time()
def runtime():
    return f"[{int(time.time() - start_time)} sec]"

print("lattice shape:",lattice_shape)
##########################################################################################
# build SU(n) interaction matrix, and build generators of interaction eigenstates
##########################################################################################
print("builing generators of interaction energy eigenstates")
sys.stdout.flush()

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
print("building operators in the shell / Z-projection basis", runtime())
sys.stdout.flush()

# spin basis
dn = np.array([1,0])
up = np.array([0,1])

# 1-local Pauli operators
local_ops = { "Z" : ( np.outer(up,up) - np.outer(dn,dn) ) / 2,
              "+" : np.outer(up,dn),
              "-" : np.outer(dn,up) }

# 2-local tensor products of Pauli operators
for op_lft, op_rht in itertools.product(local_ops.keys(), repeat = 2):
    mat_lft = local_ops[op_lft]
    mat_rht = local_ops[op_rht]
    local_ops[op_lft + op_rht] = np.kron(mat_lft,mat_rht).reshape((2,)*4)

# # build the ZZ perturbation operator in the shell / Z-projection basis
print("building perturbation operator", runtime())
sys.stdout.flush()

shell_coupling_mat \
    = build_shell_operator([sunc["mat"]], [local_ops["ZZ"]], sunc, sunc["TI"])

print("building collective spin operators", runtime())
sys.stdout.flush()

# build collective spin operators
def _pauli_mat(pauli):
    tensors = [np.ones(spin_num)]
    operators = [local_ops[pauli]]
    diagonal = ( pauli in [ "Z", "ZZ" ] )
    full_pauli_op = build_shell_operator(tensors, operators, sunc, sunc["TI"],
                                         collective = True, shell_diagonal = diagonal)
    full_pauli_op.shape = ( shell_num*(spin_num+1), )*2
    return full_pauli_op

collective_ops = { "Z" : _pauli_mat("Z"),
                   "+" : _pauli_mat("+") }
collective_ops["ZZ"] = collective_ops["Z"] @ collective_ops["Z"]
collective_ops["++"] = collective_ops["+"] @ collective_ops["+"]
collective_ops["+Z"] = collective_ops["+"] @ collective_ops["Z"]
collective_ops["+-"] = collective_ops["+"] @ collective_ops["+"].conj().T

# if this is a test run, we can exit now
if test_run: exit()

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

    # compute collective spin correlators
    def val(mat, state):
        return state.conj() @ ( mat @ state ) / ( state.conj() @ state )
    correlators = { op : np.array([ val(mat,state.flatten()) for state in states ])
                    for op, mat in collective_ops.items() }

    # compute populations
    pops = np.einsum("tsz->ts", abs(states)**2)

    return times, correlators, pops

if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

##########################################################################################
print("running inspection simulations", runtime())
sys.stdout.flush()

str_ops = [ "Z", "+", "ZZ", "++", "+Z", "+-" ]
tup_ops = [ (0,1,0), (1,0,0), (0,2,0), (2,0,0), (1,1,0), (1,0,1) ]
def relabel(correlators):
    for str_op, tup_op in zip(str_ops, tup_ops):
        correlators[tup_op] = correlators.pop(str_op)
    return correlators

str_op_list = ", ".join(str_ops)

for coupling_zz in inspect_coupling_zz:
    times, correlators, pops = simulate(coupling_zz, sim_time = inspect_sim_time)
    sqz = squeezing_from_correlators(spin_num, relabel(correlators))

    with open(data_dir + f"inspect_{name_tag}_z{coupling_zz}.txt", "w") as file:
        file.write(f"# times, {str_op_list}, sqz, populations (within each shell)\n")
        for manifold, shells in sunc["shells"].items():
            file.write(f"# manifold {manifold} : ")
            file.write(" ".join([ str(shell) for shell in shells ]))
            file.write("\n")
        for tt in range(len(times)):
            file.write(f"{times[tt]} ")
            file.write(" ".join([ str(correlators[op][tt]) for op in tup_ops ]))
            file.write(f" {sqz[tt]} ")
            file.write(" ".join([ str(pop) for pop in pops[tt,:] ]))
            file.write("\n")

##########################################################################################
if len(sweep_coupling_zz) == 0: exit()
print("running sweep simulations", runtime())
sys.stdout.flush()

sweep_coupling_zz = sweep_coupling_zz[sweep_coupling_zz != 1]
sweep_results = [ simulate(coupling_zz) for coupling_zz in sweep_coupling_zz ]
sweep_times, sweep_correlators, sweep_pops = zip(*sweep_results)

print("computing squeezing values", runtime())
sys.stdout.flush()

sweep_sqz = [ squeezing_from_correlators(spin_num, relabel(correlators))
              for correlators in sweep_correlators ]
sweep_min_sqz = [ min(sqz) for sqz in sweep_sqz ]
min_sqz_idx = [ max(1,np.argmin(sqz)) for sqz in sweep_sqz ]
sweep_time_opt = [ sweep_times[zz][idx] for zz, idx in enumerate(min_sqz_idx) ]

sweep_pops = [ np.array([ pops[:min_idx,shells].sum(axis = 1)
                          for shells in sunc["shells"].values() ]).T
               for pops, min_idx in zip(sweep_pops, min_sqz_idx) ]
sweep_min_pops = np.array([ pops.min(axis = 0) for pops in sweep_pops ])
sweep_max_pops = np.array([ pops.max(axis = 0) for pops in sweep_pops ])

str_op_opt_list = ", ".join([ op + "_opt" for op in str_ops ])
correlators_opt = [ { op : correlator[op][min_idx] for op in tup_ops }
                    for correlator, min_idx in zip(sweep_correlators, min_sqz_idx) ]

print("saving results", runtime())
sys.stdout.flush()

with open(data_dir + f"sweep_{name_tag}.txt", "w") as file:
    file.write(f"# coupling_zz, time_opt, sqz_min, {str_op_opt_list}, min_pop_0,"
               + " max_pop (for manifolds > 0)\n")
    file.write("# manifolds : ")
    file.write(" ".join([ str(manifold) for manifold in sunc["shells"].keys() ]))
    file.write("\n")
    for zz in range(len(sweep_coupling_zz)):
        file.write(f"{sweep_coupling_zz[zz]} {sweep_time_opt[zz]} {sweep_min_sqz[zz]} ")
        file.write(" ".join([ str(correlators_opt[zz][op]) for op in tup_ops ]))
        file.write(f" {sweep_min_pops[zz,0]} ")
        file.write(" ".join([ str(val) for val in sweep_max_pops[zz,1:] ]))
        file.write("\n")

print("completed", runtime())
