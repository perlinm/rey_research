#!/usr/bin/env python3

import numpy as np
import scipy.integrate

from dicke_methods import coherent_spin_state, spin_op_z_dicke

##########################################################################################
# basic objects and methods for boson mean-field theory simulations
##########################################################################################

# construct a boson MFT state from a quantum state
def boson_mft_state(bare_quantum_state, spin_dim, spin_num):
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
def spin_state(direction, spin_dim):
    return coherent_spin_state(direction, spin_dim-1)

# return a polarized state of all spins
def polarized_state(direction, spin_dim, spin_num):
    return boson_mft_state(spin_state(direction, spin_dim), spin_dim, spin_num)

# method to construct a field tensor
def field_tensor(field_ops):
    return np.array(field_ops).transpose([1,2,0])

# method computing the time derivative of a mean-field bosonic state
def boson_time_deriv(state, field, coupling_op = None):
    spin_dim, spin_num = state.shape
    state_triplet = ( state.conj(), state, state )
    if coupling_op is None:
        # uniform SU(n)-symmetric couplings
        vec = np.einsum("nk,ak,ni->ai", *state_triplet) / spin_num \
            + np.einsum("ani,ni->ai", field, state)
    elif type(coupling_op) is np.ndarray and coupling_op.ndim == 2:
        # inhomogeneous SU(n)-symmetric couplings
        vec = np.einsum("ik,rk,sk,ni->ai", coupling_op, *state_triplet) / spin_num \
            + np.einsum("ani,ni->ai", field, state)
    elif type(coupling_op) is np.ndarray and coupling_op.ndim == 4:
        # uniform asymmetric couplings
        vec = np.einsum("anrs,rk,sk,ni->ai", coupling_op, *state_triplet) / spin_num \
            + np.einsum("ani,ni->ai", field, state)
    elif type(coupling_op) is np.ndarray and coupling_op.ndim == 6:
        # inhomogeneous asymmetric couplings
        vec = np.einsum("anirsk,rk,sk,ni->ai", coupling_op, *state_triplet) / spin_num \
            + np.einsum("ani,ni->ai", field, state)
    else:
        # couplings that factorize into an "operator" part and a "spatial" part
        vec = np.einsum("anrs,ik,rk,sk,ni->ai", *coupling_op, *state_triplet) / spin_num \
            + np.einsum("ani,ni->ai", field, state)
    return -1j * vec

# wrapper for the numerical integrator, for dealing with multi-spin_dimensional state
def evolve_mft(initial_state, sim_time, field, coupling_op = None,
               ivp_tolerance = 1e-10):
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
def spin_angle(qq, spin_num):
    return 2*np.pi*(qq+1/2)/spin_num

# construct an on-site external field
def bare_lattice_field(qq, soc_angle, spin_dim, spin_num):
    spin_vals = np.diag(spin_op_z_dicke(spin_dim-1).todense())
    return np.diag([ -np.cos(spin_angle(qq, spin_num) + mu * soc_angle) for mu in spin_vals ])

def compute_mean_state(state):
    spin_dim, spin_num = state.shape
    return np.array([ state[mu,:] @ state[nu,:].conj() / spin_num
                      for mu, nu in zip(*np.triu_indices(spin_dim)) ], dtype = complex)
def compute_mean_states(states):
    return np.array([ compute_mean_state(state) for state in states ])
