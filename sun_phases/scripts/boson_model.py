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
# set up objects for simulation
##########################################################################################

S_op_vec, _ = spin_op_vec_mat_dicke(spin_dim-1)
op_labels = [ "z", "x", "y" ]
spin_ops = { label : spin_op.todense() for label, spin_op in zip(op_labels, S_op_vec) }
Sz, Sx, Sy = spin_ops.values()
Sp = ( Sx + 1j * Sy ).real
max_spin = np.max(Sz) * spin_num

def collective_val(op, state):
    vals = np.einsum("mn,mj,nj->", op, state.conj(), state)
    if np.allclose(vals.imag, 0): vals = vals.real
    return vals
def collective_vals(op, states):
    return np.array([ collective_val(op, state) for state in states ])

alt_signs = np.ones(spin_num)
alt_signs[spin_num//2:] = -1
bare_field = field_tensor([ sign * Sz for sign in alt_signs ])

def double_state(elevation, opening_angle):
    assert( spin_num % 2 == 0 )
    return boson_mft_state([ spin_state(np.pi/2-elevation, sign/2 * opening_angle)
                             for sign in alt_signs ])

##########################################################################################
# simulation method
##########################################################################################

def get_op_vals(field_scale, operator, end_cond, time_step = None, debug = False):
    field = field_scale * bare_field
    if time_step is None:
        time_step = np.pi / np.sqrt(1 + field_scale**2)

    times = np.zeros(1)
    states = np.reshape(init_state, (1,) + init_state.shape)
    op_vals = collective_vals(operator, states) / max_spin

    # simulate until the given terminal condition is satisfied
    while not end_cond(op_vals):
        new_times, new_states = evolve(states[-1,:,:], field, sim_time = time_step)
        new_op_vals = collective_vals(operator, new_states) / max_spin

        times = np.concatenate([ times, times[-1] + new_times[1:] ])
        states = np.concatenate([ states, new_states[1:] ])
        op_vals = np.concatenate([ op_vals, new_op_vals[1:] ])

    if debug:
        plt.figure(figsize = figsize)
        plt.title(debug)
        plt.plot(times/(2*np.pi), op_vals.real, label = r"$\mathrm{X}$")
        plt.plot(times/(2*np.pi), op_vals.imag, label = r"$\mathrm{Y}$")
        plt.xlabel("$t/2\pi$")
        plt.legend(loc = "best")
        plt.tight_layout()
        plt.show()

    return op_vals

def num_peaks(vals):
    return sum( ( vals[1:-1] > vals[:-2] ) &
                ( vals[1:-1] > vals[+2:] ) )
def end_cond(vals):
    if len(vals) < 3: return False
    if spin_dim == 2 or True:
        real_end = num_peaks(vals.real) > 1 or np.allclose(vals.real, vals.real[0])
        imag_end = num_peaks(vals.imag) > 1 or np.allclose(vals.imag, vals.imag[0])
        return real_end and imag_end

def get_all_extrema(field_scales, debug = False):
    all_extrema = np.empty((4,field_scales.size))

    for idx, field_scale in enumerate(field_scales):
        debug_title = f"${idx}/{len(field_scales)}$" if debug else None
        vals = get_op_vals(field_scale, Sp, end_cond, debug = debug_title)
        all_extrema[:,idx] = max(vals.real), min(vals.real), max(vals.imag), min(vals.imag)

    return all_extrema

def get_amplitudes(field_scales, debug = False):
    amplitudes = np.empty(field_scales.size)

    for idx, field_scale in enumerate(field_scales):
        debug_title = f"${idx}/{len(field_scales)}$" if debug else None
        vals = get_op_vals(field_scale, Sp, end_cond, debug = debug_title)
        real_amp = ( max(vals.real) - min(vals.real) ) / 2
        imag_amp = ( max(vals.imag) - min(vals.imag) ) / 2
        amplitudes[idx] = max(real_amp, imag_amp)

    return amplitudes

##########################################################################################
# simulate!
##########################################################################################

elevation, opening_angle = 0, np.pi
init_state = double_state(elevation, opening_angle)

field_scales = np.logspace(-2,1,21)
all_extrema = get_all_extrema(field_scales)

labels = [ r"$\braket{S_{\mathrm{x}}}_{\mathrm{max}}/S$",
           r"$\braket{S_{\mathrm{x}}}_{\mathrm{min}}/S$",
           r"$\braket{S_{\mathrm{y}}}_{\mathrm{max}}/S$",
           r"$\braket{S_{\mathrm{y}}}_{\mathrm{min}}/S$" ]

plt.figure(figsize = figsize)
plt.title(f"$n={spin_dim}$")
for extrema, label in zip(all_extrema, labels):
    if np.isclose(max(abs(extrema)), 0): continue
    plt.semilogx(field_scales, extrema, ".", label = label)

plt.xlim(field_scales[0], field_scales[-1])
plt.xlabel(r"$\epsilon_0$")
plt.legend(loc = "best", handlelength = 0.5, framealpha = 1)
plt.tight_layout()

plt.show()
