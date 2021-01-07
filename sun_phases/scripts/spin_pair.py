#!/usr/bin/env python3

import os, sys
import numpy as np
import scipy, scipy.integrate
import matplotlib.pyplot as plt
import qutip

from dicke_methods import spin_op_vec_dicke, plot_dicke_state

np.set_printoptions(linewidth = 200)

dim = int(sys.argv[1])
log10_field = float(sys.argv[2])

sim_time = 1000
time_step = 0.1
ivp_tolerance = 2.220446049250313e-14 # any smaller and we get a warning

fig_dir = f"../figures/spin_pair/zoom/"
if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

def fig_name(tag):
    return fig_dir + f"{tag}_d{dim:02d}_h{log10_field}.png"

####################

spin = (dim-1)/2
field = 10**log10_field / (2*spin)

# spin matrices
Sz, Sx, Sy = spin_op_vec_dicke(dim-1)

def rot_frame(state, time):
    phases = np.exp(-1j * time * field * Sz.diagonal())
    return phases.conj() * state

def rank(state):
    return int(np.round(np.log(state.size)/np.log(dim)))

# join two states into a 1-D array
def join(S0, S1):
    return np.array([ S0, S1 ]).ravel()

# split a 1-D array into two matrices
def split(state):
    S0 = state[:state.size//2]
    S1 = state[state.size//2:]
    S0.shape = S1.shape = (dim,) * rank(S0)
    return S0, S1

# time derivative of spin state
def time_deriv(state, field):
    B0, B1 = split(state)
    deriv_0 = B1 * ( B1.conj() @ B0 ) + field * ( Sz @ B0 )
    deriv_1 = B0 * ( B0.conj() @ B1 ) - field * ( Sz @ B1 )
    return -1j * join(deriv_0, deriv_1)

def evolve(initial_state, time_deriv, times, sim_time = sim_time):
    def ivp_deriv(time, state):
        return time_deriv(state, field)
    ivp_args = [ ivp_deriv, (0, sim_time), initial_state ]
    ivp_kwargs = dict( t_eval = times, rtol = ivp_tolerance, atol = ivp_tolerance )
    ivp_solution = scipy.integrate.solve_ivp(*ivp_args, **ivp_kwargs)
    return ivp_solution.y

####################

times = np.linspace(0, sim_time, int(sim_time/time_step+1))

# initialize states pointing up/down along Z
initial_B0 = np.zeros(dim, dtype = complex)
initial_B1 = np.zeros(dim, dtype = complex)
initial_B0[-1] = 1
initial_B1[-1] = 1

# rotate states to point up/down along X
pulse = scipy.linalg.expm(-1j * np.pi/2 * Sy.tocsc())
initial_B0 = pulse @ initial_B0
initial_B1 = pulse @ initial_B1

initial_state = join(initial_B0, initial_B1)

joined_states = evolve(initial_state, time_deriv, times)
pair_states = np.array([ split(joined_state) for joined_state in joined_states.T ])
states = pair_states[:,0,:] # 0 index for +Z spins

def spin_vec(state):
    spin_ops = np.array([ Sx, Sy, Sz ]) / spin
    return [ ( state.conj() @ ( Sj @ state ) ).real
             for Sj in spin_ops ]

spin_vecs = np.array([ spin_vec(state) for state in states ])

def make_bloch(spin_vecs, view = None):
    bloch = qutip.Bloch()
    bloch.add_points(spin_vecs.T)
    bloch.xlabel = [ "", "" ]
    bloch.ylabel = [ "", "" ]
    bloch.zlabel = [ "", "" ]
    if view: bloch.view = view
    return bloch

bloch = make_bloch(spin_vecs)
bloch.zlabel = [ r"$\log_{10}h=" + f"{log10_field:.4f}" + "$" , "" ]
bloch.zlpos[0] = 1.4
bloch.save(fig_name("bloch"))

bloch = make_bloch(spin_vecs, [0,90])
bloch.save(fig_name("top"))

bloch = make_bloch(spin_vecs, [0,0])
bloch.save(fig_name("side"))

### dim = 2,  hh = 0
### dim = 4,  hh = 0.3808, 0.3809
### dim = 6,  hh = 0.5408, 0.5409
### dim = 8,  hh = 0.6410, 0.6411
### dim = 10, hh = 0.7135, 0.7136
