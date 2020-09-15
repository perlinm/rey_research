#!/usr/bin/env python3

import sys, time
import numpy as np
import scipy.integrate

from dicke_methods import coherent_spin_state_angles, coherent_spin_state, spin_op_z_dicke

np.set_printoptions(linewidth = 200)

spin_dim = int(sys.argv[1])
spin_num = int(sys.argv[2])
init_state_str = sys.argv[3]

assert( spin_num % 2 == 0 )
assert( init_state_str in [ "X", "XX" ] )

# simulation parameters
log10_tun_vals = np.arange(-2,1.01,0.05)
soc_frac_vals = [ 0.1, 0.5, 0.9, 1.0 ]
periods = 1000

ivp_tolerance = 1e-10

data_dir = "../data/oscillations/"
sys_tag = f"n{spin_dim}_N{spin_num}_{init_state_str}"

##########################################################################################
# basic simulation objects and methods
##########################################################################################

# construct a boson MFT state from a quantum state
def boson_mft_state(bare_quantum_state):
    quantum_state = np.array(bare_quantum_state).T.copy().astype(complex)

    # if we were given a state of a single spin,
    #   construct uniform product state of all spins
    if quantum_state.ndim == 1:
        quantum_state /= np.sqrt(sum(abs(quantum_state)**2))
        quantum_state = np.repeat(quantum_state, spin_num)
        quantum_state.shape = (spin_dim,spin_num)
        return quantum_state

    # if we were given many states, construct the state of each spin independently
    if quantum_state.ndim == 2:
        quantum_state /= np.sqrt(np.sum(abs(quantum_state)**2, axis = 0))
        return quantum_state

# return a quantum "coherent" (polarized) state of a single spin
def spin_state(*direction):
    if len(direction) == 1:
        return coherent_spin_state(*direction, spin_dim-1)
    if len(direction) == 2:
        return coherent_spin_state_angles(*direction, spin_dim-1)

# return a polarized state of all spins
def polarized_state(*direction):
    return boson_mft_state(spin_state(*direction))

# method to construct a field tensor
def field_tensor(bare_field_data):
    field_data = np.array(bare_field_data)

    # if we were given a single spin operator, assume a homogeneous field
    if field_data.ndim == 2:
        tensor = np.repeat(field_data, spin_num)
        tensor.shape = (spin_dim,spin_dim,spin_num)
        return tensor

    # if we were given a list of spin operators, transpose data appropriately
    if field_data.ndim == 3:
        tensor = np.transpose(field_data, [ 1, 2, 0 ])
        return tensor

# method computing the time derivative of a mean-field bosonic state
def boson_time_deriv(state, field, coupling_op = None):
    state_triplet = ( state.conj(), state, state )
    if coupling_op is None:
        # uniform SU(n)-symmetric couplings
        vec = np.einsum("nk,ak,ni->ai", *state_triplet) / spin_num \
            - np.einsum("ni,ai,ni->ai", *state_triplet) / spin_num \
            + np.einsum("ani,ni->ai", field, state)
    elif type(coupling_op) is np.ndarray and coupling_op.ndim == 2:
        # inhomogeneous SU(n)-symmetric couplings
        vec = np.einsum("ik,rk,sk,ni->ai", coupling_op, *state_triplet) / spin_num \
            - np.einsum("ii,ri,si,ni->ai", coupling_op, *state_triplet) / spin_num \
            + np.einsum("ani,ni->ai", field, state)
    elif type(coupling_op) is np.ndarray and coupling_op.ndim == 4:
        # uniform asymmetric couplings
        vec = np.einsum("anrs,rk,sk,ni->ai", coupling_op, *state_triplet) / spin_num \
            - np.einsum("anrs,ri,si,ni->ai", coupling_op, *state_triplet) / spin_num \
            + np.einsum("ani,ni->ai", field, state)
    elif type(coupling_op) is np.ndarray and coupling_op.ndim == 6:
        # inhomogeneous asymmetric couplings
        vec = np.einsum("anirsk,rk,sk,ni->ai", coupling_op, *state_triplet) / spin_num \
            - np.einsum("anirsi,ri,si,ni->ai", coupling_op, *state_triplet) / spin_num \
            + np.einsum("ani,ni->ai", field, state)
    else:
        # couplings that factorize into an "operator" part and a "spatial" part
        vec = np.einsum("anrs,ik,rk,sk,ni->ai", *coupling_op, *state_triplet) / spin_num \
            - np.einsum("anrs,ii,ri,si,ni->ai", *coupling_op, *state_triplet) / spin_num \
            + np.einsum("ani,ni->ai", field, state)
    return -1j * vec

# wrapper for the numerical integrator, for dealing with multi-spin_dimensional state
def evolve(initial_state, sim_time, field, coupling_op = None,
           ivp_tolerance = ivp_tolerance):
    state_shape = initial_state.shape
    initial_state.shape = initial_state.size

    def time_deriv_flat(time, state):
        state.shape = state_shape
        vec = boson_time_deriv(state, field, coupling_op).ravel()
        state.shape = state.size
        return vec

    ivp_args = [ time_deriv_flat, (0, sim_time), initial_state ]
    ivp_kwargs = dict( rtol = ivp_tolerance, atol = ivp_tolerance )
    ivp_solution = scipy.integrate.solve_ivp(*ivp_args, **ivp_kwargs)

    times = ivp_solution.t
    states = ivp_solution.y
    states.shape = state_shape + (times.size,)
    states = states.transpose([2,0,1])

    initial_state.shape = state_shape
    return times, states

# get the angle associated with a given spin number <--> quasimomentum
def spin_angle(qq):
    return 2*np.pi*(qq+1/2)/spin_num

# construct an on-site external field
def bare_lattice_field(qq, soc_angle):
    return np.diag([ np.cos(spin_angle(qq) + mu * soc_angle) for mu in spin_vals ])


##########################################################################################
# set up objects for simulation
##########################################################################################

Sz = spin_op_z_dicke(spin_dim-1).todense()
spin_vals = np.diag(Sz)

def compute_spin_mat(state):
    return np.array([ state[mu,:].conj() @ state[nu,:] / spin_num
                      for mu, nu in zip(*np.triu_indices(spin_dim)) ], dtype = complex)
def compute_spin_mats(states):
    return np.array([ compute_spin_mat(state) for state in states ])

# construct initial state
if init_state_str == "X":
    init_state = polarized_state([0,1,0])
elif init_state_str == "XX":
    spin_states = [ spin_state([0,+1,0]) ] * (spin_num//2) \
                + [ spin_state([0,-1,0]) ] * (spin_num//2)
    init_state = boson_mft_state(spin_states)

##########################################################################################
# simulate!
##########################################################################################
sim_start = time.time()

for idx_soc, soc_frac in enumerate(soc_frac_vals):
    soc_tag = f"{soc_frac:.2f}"
    soc_angle = soc_frac * np.pi
    bare_field = field_tensor([ bare_lattice_field(qq, soc_angle)
                                for qq in range(spin_num) ])

    for idx_tun, log10_tun in enumerate(log10_tun_vals):
        tun_tag = f"{log10_tun:.2f}"
        file_tag = f"{sys_tag}_J{tun_tag}_p{soc_tag}"

        print(f"{idx_soc}/{len(soc_frac_vals)} " + \
              f"{idx_tun}/{len(log10_tun_vals)} " + \
              f"(soc_frac, log10_tun = {soc_tag}, {tun_tag})", end = "")
        sys.stdout.flush()

        this_start = time.time()

        tunneling = 10**log10_tun
        field = -tunneling * bare_field
        sim_time = 2*np.pi / np.sqrt(1 + tunneling)**2 * periods

        times, states = evolve(init_state, sim_time, field)
        spin_mats = compute_spin_mats(states)
        np.savetxt(data_dir + f"times_{file_tag}.txt", times)
        np.savetxt(data_dir + f"spin_mats_{file_tag}.txt", spin_mats.flatten())

        this_runtime = int( time.time() - this_start )
        print(f" {this_runtime} sec")

sim_runtime = int( time.time() - sim_start )
seconds = sim_runtime % 60
minutes = sim_runtime // 60 % 60
hours = sim_runtime // 60**2
print(f"total runtime: {hours:02d}:{minutes:02d}:{seconds:02d}")
