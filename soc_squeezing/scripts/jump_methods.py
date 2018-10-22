#!/usr/bin/env python3

# FILE CONTENTS: methods monte carlo simulations via the quantum jump method

import numpy as np
import scipy.sparse as sparse
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from math import factorial

from dicke_methods import spin_op_z_dicke, spin_op_m_dicke, coherent_spin_state
from squeezing_methods import squeezing_from_correlators, squeezing_OAT

from dicke_methods import plot_dicke_state

np.random.seed(0)

time_steps = 100
trajectories = 1000
ivp_tolerance = 1e-10

N = 10
h_pzm = { (0,2,0) : 1 }
dec_rates = [ (2,0,0), (0,2,0), (0,0,2) ]

dec_ops = [ (np.sqrt(a),np.sqrt(b),np.sqrt(c)) for a,b,c in dec_rates ]

sqz_ops = [ (0,1,0), (0,2,0), (1,0,0), (2,0,0), (1,1,0), (1,0,1) ]
max_time = 2 * N**(-2/3)
save_times = np.linspace(0,max_time,time_steps)
save_times_idx = np.arange(time_steps)

s_i = np.array([[1,0],[0,1]])
s_z = np.array([[1,0],[0,-1]]) / 2
s_p = np.array([[0,1],[0,0]])
s_m = np.array([[0,0],[1,0]])

def base_ops(J, base_ops = {}):
    try:
        return base_ops[J]
    except:
        base_ops[J] =  [ spin_op_z_dicke(int(2*J)), spin_op_m_dicke(int(2*J)) ]
        return base_ops[J]

def op_mat(J, pows, ops = {}):
    try:
        return ops[J,pows]
    except:
        S_z, S_m = base_ops(J)
        ops[J,pows] = S_m.T**pows[0] * S_z**pows[1] * S_m**pows[2]
        return ops[J,pows]

def g_op(dec):
    return dec[0] * s_p + dec[1] * s_z + dec[2] * s_m

def gg_mat(dec):
    g = g_op(dec)
    return g.conj().T @ g

def jumps(J):
    S_z, S_m = base_ops(J)
    S_i = sparse.identity(int(2*J)+1)
    return [ sum( np.trace(op.conj().T @ gg_mat(dec)) * OP
                  for op, OP in [ (s_i/2,N*S_i), (s_z,2*S_z), (s_p,S_m.T), (s_m,S_m) ] )
             for dec in dec_ops ]

def build_H_eff(J, h_pzm):
    return ( sum( val * op_mat(J, pows) for pows, val in h_pzm.items() )
             - 1j/2 * sum( jump for jump in jumps(J) ) )

def P_JM(J, M, d_J, d_M):
    assert(d_J == 0 or d_J == -1 or d_J == 1)
    assert(d_M == 0 or d_M == -1 or d_M == 1)
    if J == 0 and d_J != 1: return 0

    if d_J == 0:
        if d_M == 0: coeff = M
        else: coeff = np.sqrt((J-d_M*M) * (J+d_M*M+1))
        return np.sqrt( (2+N) / (4*J*(J+1)) ) * coeff

    if d_J == -1:
        if d_M == 0: coeff = (J-M)*(J+M)
        else: coeff = (J-d_M*M) * (J-d_M*M-1)
        sign = -1 if d_M == -1 else 1
        return sign * np.sqrt( (N+2*J+2) / (4*J*(2*J+1)) * coeff )

    if d_J == 1:
        if d_M == 0: coeff = (J-M+1)*(J+M+1)
        else: coeff = (J+d_M*M+1) * (J+d_M*M+2)
        sign = -1 if d_M == 1 else 1
        return sign * np.sqrt( (N-2*J) / (4*(J+1)*(2*J+1)) * coeff )

def P_J(J, d_J, d_M):
    return np.array([ P_JM(J, -J+mm, d_J, d_M) for mm in range(int(2*J+1)) ])

def transform(state, d_J, dec_op, P_J_mat):
    J = (state.size-1)/2 + d_J
    if J == 0: return np.array([1], dtype = complex)

    state_new = np.zeros(int(2*J+1), dtype = complex)
    if d_J == 0:
        if dec_op[1] != 0: state_new      += dec_op[1] * P_J_mat[d_J,0]      * state
        if dec_op[0] != 0: state_new[1:]  += dec_op[0] * P_J_mat[d_J,1][:-1] * state[:-1]
        if dec_op[2] != 0: state_new[:-1] += dec_op[2] * P_J_mat[d_J,-1][1:] * state[1:]
    elif d_J == 1:
        if dec_op[1] != 0: state_new[1:-1] += dec_op[1] * P_J_mat[d_J,0]  * state
        if dec_op[0] != 0: state_new[2:]   += dec_op[0] * P_J_mat[d_J,1]  * state
        if dec_op[2] != 0: state_new[:-2]  += dec_op[2] * P_J_mat[d_J,-1] * state
    elif d_J == -1:
        if dec_op[1] != 0: state_new += dec_op[1] * P_J_mat[d_J,0][1:-1] * state[1:-1]
        if dec_op[0] != 0: state_new += dec_op[0] * P_J_mat[d_J,1][:-2]  * state[:-2]
        if dec_op[2] != 0: state_new += dec_op[2] * P_J_mat[d_J,-1][2:]  * state[2:]

    return state_new / np.sqrt(np.dot(state_new.conj(),state_new))

def sqr_norm(vec):
    return abs(np.dot(vec.conj(),vec))

def correlator(J, op, state):
    return state.conj() @ op_mat(J,op) @  state / sqr_norm(state)

def choose_index(probabilities):
    probabilities /= probabilities.sum()
    choice = np.random.rand()
    idx = 0
    while choice > probabilities[:idx+1].sum():
        idx += 1
    return idx

correlator_mat = np.zeros((trajectories,len(sqz_ops),time_steps), dtype = complex)

for trajectory in range(trajectories):
    J = N/2
    state = coherent_spin_state([0,1,0], N)
    H_eff = build_H_eff(J, h_pzm)

    time = 0
    while time < max_time:

        # construct time derivative operator
        def time_derivative(time, state):
            return -1j * H_eff @ state

        # construct dec event that returns zero when we do a jump
        jump_norm = np.random.random()
        def dec_event(time, state):
            return abs(state.conj() @ state) - jump_norm
        dec_event.terminal = True
        dec_event.direction = 0

        # simulate until a jump occurs
        ivp_solution = solve_ivp(time_derivative, (time, max_time), state,
                                 events = dec_event, rtol = ivp_tolerance)
        times = ivp_solution.t
        states = ivp_solution.y

        # set current time and state
        time = times[-1]
        state = states[:,-1] / np.sqrt(sqr_norm(states[:,-1]))

        # compute correlators at save points
        correlators = { op : np.array([ correlator(J, op, states[:,tt])
                                        for tt in range(times.size) ])
                        for op in sqz_ops }

        # determine indices times at which we wish to save correlators
        interp_time_idx = (save_times >= times[0]) & (save_times <= times[-1])
        interp_times = save_times[interp_time_idx]

        # compute correlators at desired times by interpolation
        for op_idx in range(len(sqz_ops)):
            op_interp = interpolate.interp1d(times,correlators[sqz_ops[op_idx]])
            correlator_mat[trajectory,op_idx,interp_time_idx] = op_interp(interp_times)

        # if there are no decoherence operators
        #   or the ivp solver terminated by reaching the maximum time, skip the jump
        if len(dec_ops) == 0 or ivp_solution.status == 0: continue

        # compute jump probabilities for each operater and change in net spin J
        P_J_mat = { (d_J,d_M) : P_J(J,d_J,d_M) for d_J in [1,0,-1] for d_M in [1,0,-1] }
        probs = np.zeros((len(dec_ops),3))
        for dec_idx in range(len(dec_ops)):
            for d_J in [ 1, 0, -1 ]:
                dec = dec_rates[dec_idx]
                probs[dec_idx,1-d_J] \
                    = sum( dec[1-d_M] * sqr_norm(P_J_mat[d_J,d_M] * state)
                           for d_M in [1,0,-1] if dec[1-d_M] != 0 )

        # choose which decoherence operator to use
        dec_choice = choose_index(probs.sum(1))
        dec_op = dec_ops[dec_choice]

        # choose which branch of J --> J+1, J, J-1 to follow
        d_J = 1 - choose_index(probs[dec_choice,:])
        J += d_J

        # transform state according to jump, and rebuild Hamiltonian if necessary
        state = transform(state, d_J, dec_op, P_J_mat)
        if d_J != 0: H_eff = build_H_eff(J, h_pzm)

correlators = { sqz_ops[op_idx] : correlator_mat[:,op_idx,:].mean(0)
                for op_idx in range(len(sqz_ops)) }
sqz_test = squeezing_from_correlators(N, correlators)
plt.plot(sqz_test, "k.")

sqz_OAT = squeezing_OAT(N, save_times)
sqz_OAT_D = squeezing_OAT(N, save_times, (2,2,2))

plt.plot(sqz_OAT, "b-")
plt.plot(sqz_OAT_D, "r-")


###################################################################################################
tat_file = "/home/perlinm/Workspace/MATLAB/correlators_TAT.txt"
times, Z, X, Y, ZZ, XX, YY, ZX, ZY, XY = np.loadtxt(tat_file, delimiter = ",", unpack = True)
with open(tat_file, "r") as f:
    line = f.readline()
    if "spin_num" in line:
        spin_num = int(line.split()[-1])

correlators = { (1,0,0) : Z,
                (0,1,0) : X,
                (0,0,1) : Y,
                (2,0,0) : ZZ,
                (0,2,0) : XX,
                (0,0,2) : YY,
                (1,1,0) : ZX,
                (1,0,1) : ZY,
                (0,1,1) : XY }

sqz_test = squeezing_from_correlators(spin_num, correlators, zxy_basis = True)
plt.plot(sqz_test, ".c")
###################################################################################################


plt.show()
