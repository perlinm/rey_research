#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

from dicke_methods import coherent_spin_state_angles, coherent_spin_state, \
    spin_op_vec_mat_dicke

np.set_printoptions(linewidth = 200)

spin_dim = 2
spin_num = 10

sim_time = 2 * np.pi
ivp_tolerance = 1e-10

figsize = (4,3)
params = { "font.size" : 12,
           "text.usetex" : True,
           "text.latex.preamble" : [ r"\usepackage{braket}" ]}
plt.rcParams.update(params)

##########################################################################################
# basic simulation objects and methods
##########################################################################################

S_op_vec, _ = spin_op_vec_mat_dicke(spin_dim-1)

op_labels = [ "z", "x", "y" ]
spin_ops = { label : spin_op.todense() for label, spin_op in zip(op_labels, S_op_vec) }
Sz, Sx, Sy = spin_ops.values()
Sp = ( Sx + 1j * Sy ).real
max_spin = np.max(Sz) * spin_num

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
        vec = np.einsum("nk,ak,ni->ai", *state_triplet) / spin_num \
            - np.einsum("ni,ai,ni->ai", *state_triplet) / spin_num \
            + np.einsum("ani,ni->ai", field, state)
    elif coupling_op.ndim == 4:
        vec = np.einsum("anrs,rk,sk,ni->ai", coupling_op, *state_triplet) / spin_num \
            - np.einsum("anrs,ri,si,ni->ai", coupling_op, *state_triplet) / spin_num \
            + np.einsum("ani,ni->ai", field, state)
    elif coupling_op.ndim == 6:
        vec = np.einsum("anrsik,rk,sk,ni->ai", coupling_op, *state_triplet) / spin_num \
            - np.einsum("anrsii,ri,si,ni->ai", coupling_op, *state_triplet) / spin_num \
            + np.einsum("ani,ni->ai", field, state)
    return -1j * vec

# wrapper for the numerical integrator, for dealing with multi-spin_dimensional state
def evolve(initial_state, field, coupling_op = None,
           sim_time = sim_time, ivp_tolerance = ivp_tolerance):
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

##########################################################################################
# simulate and plot results
##########################################################################################

def collective_val(op, state):
    vals = np.einsum("mn,mj,nj->", op, state.conj(), state)
    if np.allclose(vals.imag, 0): vals = vals.real
    return vals
def collective_vals(op, states):
    return np.array([ collective_val(op, state) for state in states ])

alt_signs = np.ones(spin_num)
alt_signs[1::2] = -1

state_X = polarized_state([0,1,0])
field_Z = field_tensor(Sz)

state_X_alt = boson_mft_state([ spin_state([0,sign,0]) for sign in alt_signs ])
field_alt = field_tensor([ sign * Sz for sign in alt_signs ])

init_state = state_X_alt
bare_field = field_alt

scales = np.logspace(-2,0.5,21)
all_extrema = np.empty((4,scales.size))

for idx, scale in enumerate(scales):
    field = scale * bare_field
    time_step = np.pi / np.sqrt(1 + scale**2)

    state = init_state

    finished = False
    times = np.zeros(1)
    states = np.reshape(init_state, (1,) + init_state.shape)
    vals = collective_vals(Sp, states) / max_spin

    # simulate for one oscillation of the order parameter
    while True:
        new_times, new_states = evolve(states[-1,:,:], field, sim_time = time_step)
        new_vals = collective_vals(Sp, new_states) / max_spin
        times = np.concatenate([ times, times[-1] + new_times[1:] ])
        states = np.concatenate([ states, new_states[1:] ])
        vals = np.concatenate([ vals, new_vals[1:] ])

        def _num_peaks(values):
            return sum( ( values[1:-1] > values[:-2] ) &
                        ( values[1:-1] > values[+2:] ) )
        real_end = _num_peaks(vals.real) > 1 or np.allclose(vals.real, vals.real[0])
        imag_end = _num_peaks(vals.imag) > 1 or np.allclose(vals.imag, vals.imag[0])
        if real_end and imag_end: break

    all_extrema[:,idx] = max(vals.real), min(vals.real), max(vals.imag), min(vals.imag)

labels = [ r"$\max\mathrm{Re}(\Delta)$", r"$\min\mathrm{Re}(\Delta)$",
           r"$\max\mathrm{Im}(\Delta)$", r"$\min\mathrm{Im}(\Delta)$" ]

plt.figure(figsize = figsize)
plt.title(f"$n={spin_dim}$")
for extrema, label in zip(all_extrema, labels):
    if np.isclose(max(abs(extrema)), 0): continue
    plt.semilogx(scales, extrema, ".", label = label)
plt.xlabel(r"$\epsilon_0$")
plt.legend(loc = "best", handlelength = 0.5, framealpha = 1)
plt.tight_layout()

plt.show()
