#!/usr/bin/env python3

import os, sys, time
import numpy as np
import scipy, scipy.integrate

from dicke_methods import spin_op_vec_dicke

np.set_printoptions(linewidth = 200)

dim = sys.argv[1]
log10_field = sys.argv[2]

sim_time = 10**4
time_step = 0.1
ivp_tolerance = 2.220446049250313e-14 # smallest value allowed

data_dir = f"../data/spin_pair/"
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

data_file = data_dir + f"states_d{dim}_h{log10_field}.txt"

genesis = time.time()
####################

dim = int(dim)
log10_field = float(log10_field)

spin = (dim-1)/2
field = 10**log10_field / (2*spin)

# spin matrices
sz, sx, sy = spin_op_vec_dicke(dim-1)

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

def val(op, state):
    if rank(state) == 2:
        return np.trace(op @ state)
    else:
        return state.conj() @ ( op @ state )

# time derivative of spin state
def time_deriv(state, field):
    B0, B1 = split(state)
    deriv_0 = B1 * ( B1.conj() @ B0 ) + field * ( sz @ B0 )
    deriv_1 = B0 * ( B0.conj() @ B1 ) - field * ( sz @ B1 )
    return -1j * join(deriv_0, deriv_1)

def evolve(initial_state, time_deriv, times, tol = ivp_tolerance):
    def ivp_deriv(time, state): return time_deriv(state, field)
    ivp_args = [ ivp_deriv, (times[0], times[-1]), initial_state ]
    ivp_kwargs = dict( rtol = tol, atol = tol, t_eval = times )
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
pulse = scipy.linalg.expm(-1j * np.pi/2 * sy.todense())
initial_B0 = pulse @ initial_B0
initial_B1 = pulse @ initial_B1

initial_state = join(initial_B0, initial_B1)
states = evolve(initial_state, time_deriv, times)

data = np.vstack([ times, states ]).T
np.savetxt(data_file, data, header = "time, state")

print("runtime:", time.time()-genesis, "seconds")
