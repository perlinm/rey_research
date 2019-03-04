#!/usr/bin/env python3

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

import itertools, scipy
from correlator_methods import compute_correlators, invert_vals

from random import random, seed
seed(0)

I2 = qt.qeye(2)
up = qt.basis(2,0)
dn = qt.basis(2,1)

### basic simulation options / parameters

N = 3
order_cap = 20
max_time = 0.03
init_state = "+X"

chi_times = np.linspace(0,max_time,100)
ivp_tolerance = 1e-10
mu = -1

### construct a random system

def rand(magnitude = 1):
    return ( random() + 1j*random() ) * magnitude

mag = 0
dec_rates = [ (rand(mag), 0, 0), (rand(mag), 0, 0) ]
dec_mat = np.array([ [ random() for jj in range(3) ] for kk in range(3) ])

h_rand = { (ll,mm,nn) : random()
           for ll, mm, nn in itertools.product(range(3), repeat = 3)
           if ll + mm + nn < 3 }

track_ops = [ (ll,mm,nn) for ll, mm, nn in itertools.product(range(3), repeat = 3)
              if ll + mm + nn < 3 ]

### construct operators for exact simulation

II = qt.tensor([I2]*N)
ZZ = 0 * II

s_z_j = []
s_p_j = []
s_m_j = []
for jj in range(N):
    s_z_j.append(qt.tensor([ I2 ] * jj + [ qt.sigmaz()/2 ] + [ I2 ] * (N-jj-1)))
    s_p_j.append(qt.tensor([ I2 ] * jj + [ qt.sigmap() ] + [ I2 ] * (N-jj-1)))
    s_m_j.append(qt.tensor([ I2 ] * jj + [ qt.sigmam() ] + [ I2 ] * (N-jj-1)))

S_p = sum(s_p_j)
S_m = sum(s_m_j)
S_z = sum(s_z_j)
S_x =   1/2 * ( S_p + S_m )
S_y = -1j/2 * ( S_p - S_m )

# Hamiltonian
H = II
for op, val in h_rand.items():
    H += S_z**op[0] * S_x**op[1] * S_y**op[2] * val

# decoherence vectors
dec_vecs_g = []
dec_vecs_G = []
for jj in range(3):
    dec_vec_g = dec_mat[:,jj] * np.sqrt(dec_rates[0][jj])
    dec_vec_G = dec_mat[:,jj] * np.sqrt(dec_rates[1][jj])
    if not max(dec_vec_g) == 0:
        dec_vecs_g.append(dec_vec_g)
    if not max(dec_vec_G) == 0:
        dec_vecs_G.append(dec_vec_G)

# jump operators
gams_g = [ vec[0] * s_p_j[jj] + vec[1] * s_z_j[jj] + vec[2] * s_m_j[jj]
           for jj in range(N) for vec in dec_vecs_g ]
gams_G = [ vec[0] * S_p + vec[1] * S_z + vec[2] * S_m
           for vec in dec_vecs_G ]
gams = gams_g + gams_G
ggs = [ gam.dag() * gam for gam in gams ]

# initial conditions
vec_Z = qt.tensor([ dn ] * N)
vec_X = qt.tensor([ up + dn ] * N) / 2**(N/2)

state_Z = vec_Z * vec_Z.dag()
state_X = vec_X * vec_X.dag()

if init_state == "-Z":
    init_state_mat = state_Z
if init_state == "+X":
    init_state_mat = state_X

# convert qutip objects to ndarrays
init_state_vec = init_state_mat.data.toarray().flatten()
H = H.data.toarray()
gams = [ gam.data.toarray() for gam in gams ]
ggs = [ gg.data.toarray() for gg in ggs ]

# time evolution
def comm(A,B): return A @ B - B @ A
def acomm(A,B): return A @ B + B @ A

def ham(state_vec):
    state_mat = state_vec.reshape((2**N,2**N))
    ham_mat = -1j * comm(H, state_mat)
    return ham_mat.flatten()

def dec(state_vec):
    if gams == []: return np.zeros(state_vec.size)
    state_mat = state_vec.reshape((2**N,2**N))
    dec_mat = sum([ gam @ state_mat @ gam.conj().T
                    - 1/2 * acomm(ggs[jj], state_mat)
                    for jj, gam in enumerate(gams) ])
    return dec_mat.flatten()

def time_deriv(time,state):
    return ham(state) + dec(state)

states = scipy.integrate.solve_ivp(time_deriv, (0,chi_times[-1]), init_state_vec,
                                   t_eval = chi_times,
                                   rtol = ivp_tolerance, atol = ivp_tolerance).y
states = [ states[:,tt].reshape((2**N,2**N)) for tt in range(chi_times.size) ]

# operators that we track
def mat(op):
    if mu == +1: S_mu, S_bmu = S_p, S_m
    if mu == -1: S_mu, S_bmu = S_m, S_p
    return S_mu**op[0] * (mu*S_z)**op[1] * S_bmu**op[2]
op_mats = { op : mat(op).data.toarray() for op in track_ops }

# exact correlators
correlators_exact = { op : np.zeros(len(chi_times), dtype = complex)
                      for op in track_ops }
for op in correlators_exact.keys():
    for tt in range(chi_times.size):
        correlators_exact[op][tt] = (op_mats[op] @ states[tt]).trace()

# compute correlators using taylor expansion method and compare
correlators = {}
for nu in [ +1, -1 ]:
    correlators[nu] = compute_correlators(N, order_cap, chi_times, init_state, h_rand,
                                          dec_rates, dec_mat, mu = nu)

if mu == -1:
    correlators[+1] = invert_vals(correlators[+1])
    correlators[-1] = invert_vals(correlators[-1])

for op in correlators[1].keys():
    plt.figure()
    plt.title(op)

    plt.plot(np.real(correlators_exact[op]))
    plt.plot(np.real(correlators[+1][op]), "--")
    plt.plot(np.real(correlators[-1][op]), ":")
    plt.plot(np.imag(correlators_exact[op]))
    plt.plot(np.imag(correlators[+1][op]), "--")
    plt.plot(np.imag(correlators[-1][op]), ":")

    ylim = abs(correlators_exact[op]).max() * 1.1
    plt.ylim(-ylim,ylim)

plt.show()
