#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

from dicke_methods import coherent_spin_state_angles, coherent_spin_state, \
    spin_op_p_dicke, spin_op_z_dicke

np.set_printoptions(linewidth = 200)

spin_dim = 2
spin_num = 10

sim_time = 2 * np.pi
ivp_tolerance = 1e-10

color_map = "inferno"

figsize = (4,3)
params = { "font.size" : 12,
           "text.usetex" : True,
           "text.latex.preamble" : [ r"\usepackage{physics}",
                                     r"\usepackage{braket}" ]}
plt.rcParams.update(params)

assert( spin_dim % 2 == 0 )

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

Sz = spin_op_z_dicke(spin_dim-1).todense()
Sp = spin_op_p_dicke(spin_dim-1).todense()
max_spin = np.max(Sz) * spin_num

def state_polarization(state):
    Sz_val = np.einsum("mn,mj,nj->", Sz, state.conj(), state)
    Sp_val = np.einsum("mn,mj,nj->", Sp, state.conj(), state)
    return np.sqrt( abs(Sz_val)**2 + abs(Sp_val)**2 ) / max_spin
def state_polarizations(states):
    return np.array([ state_polarization(state) for state in states ])

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

def get_polarizations(init_state, field_scale, end_cond,
                      coupling_op = None, time_step = None, debug = False):
    field = field_scale * bare_field
    if time_step is None:
        if spin_dim == 2:
            time_step = np.pi / np.sqrt(1 + field_scale**2)
        else:
            time_step = 2*np.pi / min(field_scale, 1) * spin_num

    times = np.zeros(1)
    states = np.reshape(init_state, (1,) + init_state.shape)
    op_vals = state_polarizations(states)

    while not end_cond(op_vals):
        new_times, new_states = evolve(states[-1,:,:], field,
                                       coupling_op = coupling_op, sim_time = time_step)
        new_op_vals = state_polarizations(new_states)

        times = np.concatenate([ times, times[-1] + new_times[1:] ])
        states = np.concatenate([ states, new_states[1:] ])
        op_vals = np.concatenate([ op_vals, new_op_vals[1:] ])

    if debug:
        plt.figure(figsize = figsize)
        plt.title(debug)
        plt.plot(times/(2*np.pi), op_vals)
        plt.xlabel("$t/2\pi$")
        plt.tight_layout()
        plt.show()

    return op_vals

def num_peaks(vals):
    return sum( ( vals[1:-1] > vals[:-2] ) &
                ( vals[1:-1] > vals[+2:] ) )
def end_cond(vals):
    if len(vals) < 3: return False
    return num_peaks(vals) > 1 or np.allclose(vals, vals.real[0])

def get_amplitudes(init_state, field_scales, coupling_op = None,
                   time_step = None, debug = False):
    amplitudes = np.empty(field_scales.size)
    for idx, field_scale in enumerate(field_scales):
        debug_title = f"${idx}/{len(field_scales)}$" if debug else None
        vals = get_polarizations(init_state, field_scale, end_cond,
                                 coupling_op = coupling_op, time_step = time_step,
                                 debug = debug_title)
        amplitudes[idx] = max(vals) - min(vals)
    return amplitudes

##########################################################################################
# simulate!
##########################################################################################

field_fac = 1/2
field_scales = np.logspace(-2,1,51)
elevations = np.pi/2 * np.linspace(-1,1,51)

for opening_angle in [ 0, np.pi ]:
    print(round(opening_angle/np.pi))

    amplitudes = np.zeros((field_scales.size,elevations.size))
    for idx in range((elevations.size+1)//2):
        print(f" {idx}/{(elevations.size+1)//2}")
        init_state = double_state(elevations[idx], opening_angle)
        amplitudes[:,+idx] = get_amplitudes(init_state, field_scales * field_fac)
        amplitudes[:,-(idx+1)] = amplitudes[:,+idx]

    figure, axis = plt.subplots(figsize = figsize)
    if opening_angle == 0:
        axis.set_title(r"$\theta_0=0$")
    if opening_angle == np.pi:
        axis.set_title(r"$\theta_0=\pi$")

    dy = elevations[1] - elevations[0]
    elev_grid = np.concatenate([ elevations, [ elevations[-1] + dy ] ]) - dy/2

    log_scales = np.log(field_scales)
    dx = log_scales[1] - log_scales[0]
    log_scale_grid = np.concatenate([ log_scales, [ log_scales[-1] + dx ] ]) - dx/2
    scale_grid = np.exp(log_scale_grid)

    image = axis.pcolormesh(scale_grid, elev_grid/np.pi, amplitudes.T,
                            cmap = color_map, vmin = 0, vmax = 1)

    axis.set_xlabel(r"$\delta_E$")
    axis.set_ylabel(r"$\phi/\pi$")

    axis.set_xscale("log")
    axis.set_yticks([-1/2,-1/4,0,1/4,1/2])
    axis.set_yticklabels(["$-1/2$","$-1/4$","$0$","$1/4$","$1/2$"])

    figure.colorbar(image)
    image.set_rasterized(True)
    fig_name = f"../figures/BCS_osc_{round(opening_angle/np.pi)}.pdf"
    plt.tight_layout(pad = 0.1)
    figure.savefig(fig_name, dpi = 300)

    plt.close(figure)
