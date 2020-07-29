#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate, scipy.signal

from dicke_methods import coherent_spin_state_angles, coherent_spin_state, \
    spin_op_z_dicke, spin_op_p_dicke

np.set_printoptions(linewidth = 200)

spin_dim = int(sys.argv[1])
spin_num = int(sys.argv[2])

ivp_tolerance = 1e-10

color_map = "inferno"

data_dir = "../data/oscillations/"
fig_dir = "../figures/oscillations/"
sys_tag = f"n{spin_dim}_N{spin_num}"

figsize = (4,3)
params = { "font.size" : 12,
           "text.usetex" : True,
           "text.latex.preamble" : [ r"\usepackage{physics}",
                                     r"\usepackage{braket}" ]}
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

def get_params(param_method, initial_state, field, coupling_op = None,
               time_step = 2*np.pi, end_cond = None,
               ivp_tolerance = ivp_tolerance):

    evolve_args = [ time_step, field, coupling_op, ivp_tolerance ]
    times, states = evolve(initial_state, *evolve_args)
    params = param_method(states)
    if end_cond is None:
        return times, params

    while not end_cond(params):
        new_times, states = evolve(states[-1], *evolve_args)
        new_params = param_method(states)

        times = np.concatenate([ times, times[-1] + new_times[1:] ])
        params = np.concatenate([ params, new_params[1:] ])

    return times, params

##########################################################################################
# set up objects for simulation
##########################################################################################

Sz = spin_op_z_dicke(spin_dim-1).todense()
Sp = spin_op_p_dicke(spin_dim-1).todense()
spin_vals = np.diag(Sz)

def transition(mu,nu):
    op = np.zeros((spin_dim,)*2)
    op[mu,nu] = 1
    return op
swap = sum( np.kron(transition(mu,nu).T, transition(mu,nu))
            for mu in range(spin_dim) for nu in range(spin_dim) )
swap.shape = (spin_dim,)*4

def compute_SS_val(state):
    state_quad = ( state.conj(), state, state.conj(), state )
    SS = np.einsum("mnrs,mj,nj,rk,sk->", swap, *state_quad).real
    return SS / spin_num**2
def compute_SS_vals(states):
    return np.array([ compute_SS_val(state) for state in states ])

def get_interp_vals(xs, ys, kind = "cubic"):
    interp = scipy.interpolate.interp1d(xs, ys, kind = kind)
    new_xs = np.linspace(xs[0], xs[-1], xs.size)
    new_ys = interp(new_xs)
    return new_xs, new_ys

##########################################################################################
# simulate!
##########################################################################################

# simulation parameters
soc_angle = np.pi/2
log10_tun_vals = np.linspace(-2,1,11)

periods = 1000
freq_num = 1000
freq_scale = 2
plot_peaks = 100

# construct "bare" external field
def bare_lattice_field(qq):
    return np.diag([ np.cos(2*np.pi*qq/spin_num + mu * soc_angle)
                     for mu in spin_vals ])
bare_field = field_tensor([ bare_lattice_field(qq) for qq in range(spin_num) ])

# construct initial state
assert( spin_num % 2 == 0 )
theta_1, phi_1 = np.pi/2 + np.arcsin(1/3), + np.pi/3
theta_2, phi_2 = np.pi/2 + np.arcsin(1/3), - np.pi/3
eta = -(spin_dim-1)/3 * np.pi % (2*np.pi)
quantum_init_state = spin_state(theta_1, phi_1) * np.exp(+1j*eta) \
                   + spin_state(theta_2, phi_2) * np.exp(-1j*eta)
init_state = boson_mft_state(quantum_init_state)

# !!!!!!!!!!!!!!!!!!!!
if spin_dim == 2:
    alt_signs = np.ones(spin_num)
    alt_signs[spin_num//2:] = -1
    bare_field = field_tensor([ sign * Sz for sign in alt_signs ])
    init_state = polarized_state([0,1,0])
# !!!!!!!!!!!!!!!!!!!!

SS_ranges = np.zeros(log10_tun_vals.size)
tunneling_vals = 10**log10_tun_vals
for idx, ( log10_tun, tunneling ) in enumerate(zip(log10_tun_vals, tunneling_vals)):
    tun_tag = f"{log10_tun:.2f}"
    tun_title = r"$\log_{10}(J/U)=" + tun_tag + "$"
    print(f"{idx}/{log10_tun_vals.size} (log10_tun = {tun_tag})", end = "")
    sys.stdout.flush()

    field = -tunneling * bare_field
    sim_time = 2*np.pi / np.sqrt(1 + tunneling)**2 * periods
    try:
        times = np.loadtxt(data_dir + f"times_{sys_tag}_J{tun_tag}.txt")
        SS_vals = np.loadtxt(data_dir + f"SS_vals_{sys_tag}_J{tun_tag}.txt")
    except:
        times, states = evolve(init_state, sim_time, field)
        SS_vals = compute_SS_vals(states)
        np.savetxt(data_dir + f"times_{sys_tag}_J{tun_tag}.txt", times)
        np.savetxt(data_dir + f"SS_vals_{sys_tag}_J{tun_tag}.txt", SS_vals)
    SS_ranges[idx] = max(SS_vals) - min(SS_vals)

    peaks, _ = scipy.signal.find_peaks(SS_vals)
    print(f" {peaks.size} peaks")

    max_plot_time = times[peaks[plot_peaks]]
    max_plot_freq = np.sqrt(1 + tunneling)**2 * peaks.size / periods * freq_scale

    figure, axes = plt.subplots(2)
    axes[0].set_title(tun_title)

    # plot time-series data
    axes[0].plot(times/(2*np.pi), SS_vals)
    axes[0].set_xlabel(r"$t \times U/2\pi$")
    axes[0].set_ylabel(r"$\braket{S^2}$")

    # plot power spectrum
    times, SS_vals = get_interp_vals(times, SS_vals)
    freqs = 2*np.pi * np.fft.rfftfreq(times.size, times[1])
    SS_amps = np.fft.rfft(SS_vals-np.mean(SS_vals)) / times.size
    axes[1].plot(freqs, abs(SS_amps)**2)
    axes[1].set_xlabel(r"$\omega/U$")
    axes[1].set_ylabel(r"$P(\omega)$")
    axes[1].set_yticks([])

    axes[0].set_xlim(0, max_plot_time/(2*np.pi))
    axes[1].set_xlim(0, max_plot_freq)
    figure.tight_layout()
    figure.savefig(fig_dir + f"time_series/series_{sys_tag}_J{tun_tag}.pdf")
    plt.close(figure)

plt.figure(figsize = figsize)
plt.semilogx(tunneling_vals, SS_ranges, "k.")
plt.xlabel(r"$J/U$")
if spin_dim == 2: plt.xlabel(r"$\delta_E$")
plt.ylabel(r"$\Delta$")
plt.tight_layout(pad = 0.1)
plt.savefig(fig_dir + f"osc_{sys_tag}_cut.pdf")
