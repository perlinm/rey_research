#!/usr/bin/env python3

# FILE CONTENTS: methods monte carlo simulations via the quantum jump method

import numpy as np
import scipy.sparse as sparse
import scipy.linalg as linalg
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from math import factorial

from dicke_methods import spin_op_z_dicke, spin_op_m_dicke, coherent_spin_state
from squeezing_methods import squeezing_from_correlators, squeezing_OAT

from dicke_methods import plot_dicke_state

np.random.seed(0)

time_steps = 100
trajectories = 100
ivp_tolerance = 1e-10

N = 10
h_pzm = { (0,2,0) : 1 }
dec_rates = [ (0,0,2) ]

dec_ops = [ (np.sqrt(a/2),np.sqrt(b),np.sqrt(c)) for a,b,c in dec_rates ]

sqz_ops = [ (0,1,0), (0,2,0), (1,0,0), (2,0,0), (1,1,0), (1,0,1) ]
max_time = 2 * N**(-2/3)
save_times = np.linspace(0,max_time,time_steps)
save_times_idx = np.arange(time_steps)

s_i = np.array([[1,0],[0,1]])
s_z = np.array([[1,0],[0,-1]])
s_p = np.array([[0,1],[0,0]])
s_m = np.array([[0,0],[1,0]])
s_x = s_p + s_m
s_y = -1j * ( s_p - s_m )

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
    return dec[0] * s_z + dec[1] * s_p + dec[2] * s_m

def gg_mat(dec):
    g = g_op(dec)
    return g.conj().T @ g

def jumps(J):
    S_z, S_m = base_ops(J)
    S_x = ( S_m + S_m.T ) / 2
    S_y = ( S_m - S_m.T ) * 1j / 2
    S_i = sparse.identity(int(2*J)+1)
    return [ sum( np.trace(op.conj().T @ gg_mat(dec)) * OP
                  for op, OP in [ (s_i,J*S_i), (s_z,S_z), (s_x,S_x), (s_y,S_y) ] )
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
        if dec_op[0] != 0: state_new      += dec_op[0] * P_J_mat[d_J,0]      * state
        if dec_op[1] != 0: state_new[1:]  += dec_op[1] * P_J_mat[d_J,1][:-1] * state[:-1]
        if dec_op[2] != 0: state_new[:-1] += dec_op[2] * P_J_mat[d_J,-1][1:] * state[1:]
    elif d_J == 1:
        if dec_op[0] != 0: state_new[1:-1] += dec_op[0] * P_J_mat[d_J,0]  * state
        if dec_op[1] != 0: state_new[2:]   += dec_op[1] * P_J_mat[d_J,1]  * state
        if dec_op[2] != 0: state_new[:-2]  += dec_op[2] * P_J_mat[d_J,-1] * state
    elif d_J == -1:
        if dec_op[0] != 0: state_new += dec_op[0] * P_J_mat[d_J,0][1:-1] * state[1:-1]
        if dec_op[1] != 0: state_new += dec_op[1] * P_J_mat[d_J,1][:-2]  * state[:-2]
        if dec_op[2] != 0: state_new += dec_op[2] * P_J_mat[d_J,-1][2:]  * state[2:]

    return state_new / np.sqrt(np.dot(state_new.conj(),state_new))

def sqr_norm(vec):
    return abs(np.dot(vec.conj(),vec))

def choose_index(probabilities):
    probabilities /= probabilities.sum()
    choice = np.random.rand()
    idx = 0
    while choice > probabilities[:idx+1].sum():
        idx += 1
    return idx

correlator_mat = np.zeros((trajectories,len(sqz_ops),time_steps), dtype = complex)

zzz = 0
upp = 0
dnn = 0
ttt = 0

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

        # simulate until a jump occurs, saving states along the way
        save_idx = (save_times >= time)
        save_points = solve_ivp(time_derivative, (time, max_time), state,
                                t_eval = save_times[save_idx], events = dec_event,
                                rtol = ivp_tolerance)
        save_time = save_points.t[-1] # time at last save point
        states = save_points.y

        # compute all correlators at the desired times
        time_idx_list = save_times_idx[save_idx & (save_times <= save_time)]
        for tt in range(len(time_idx_list)):
            time_idx = time_idx_list[tt]
            state = states[:,tt]
            state_conj = state.conj()
            state_sqr_norm = np.dot(state_conj,state)
            for op_idx in range(len(sqz_ops)):
                correlator_mat[trajectory,op_idx,time_idx] \
                    = state_conj @ op_mat(J,sqz_ops[op_idx]) @ state / state_sqr_norm

        if len(dec_ops) == 0 or save_time == max_time:
            time = save_time
            continue

        time = save_time

        # # simulate from the last save point up to the time of the jump
        # jump_time = save_points.t_events[0][0]
        # jump_point = solve_ivp(time_derivative, (save_time, jump_time), state,
        #                        rtol = ivp_tolerance)
        # time = jump_point.t[-1]
        # state = jump_point.y[:,-1]

        # compute jump probabilities for each operater and change in net spin J
        P_J_mat = { (d_J,d_M) : P_J(J,d_J,d_M) for d_J in [1,0,-1] for d_M in [1,0,-1] }
        probs = np.zeros((len(dec_ops),3))
        for dec_idx in range(len(dec_ops)):
            for d_J in [ -1, 0, 1 ]:
                dec = dec_ops[dec_idx]
                probs[dec_idx,-d_J+1] \
                    = sum( sqr_norm(dec[-d_M+1] * P_J_mat[d_J,d_M] * state)
                           for d_M in [1,0,-1] if dec[-d_M+1] != 0 )

        # choose which decoherence operator to use
        dec_choice = choose_index(probs.sum(1))
        dec_op = dec_ops[dec_choice]

        # choose which branch of J --> J+1, J, J-1 to follow
        d_J = -choose_index(probs[dec_choice,:]) + 1
        J += d_J
        if d_J == 1: upp += 1
        if d_J == 0: zzz += 1
        if d_J == -1: dnn += 1

        # transform state according to jump, and rebuild Hamiltonian if necessary
        state = transform(state, d_J, dec_op, P_J_mat)
        if d_J != 0: H_eff = build_H_eff(J, h_pzm)

print(upp,zzz,dnn,upp+zzz+dnn)
exit()

correlators = { sqz_ops[op_idx] : correlator_mat[:,op_idx,:].mean(0)
                for op_idx in range(len(sqz_ops)) }
sqz_test = squeezing_from_correlators(N, correlators)

sqz_OAT = squeezing_OAT(N, save_times)
sqz_OAT_D = squeezing_OAT(N, save_times, dec_rates[0])

plt.plot(sqz_OAT, "b-")
plt.plot(sqz_OAT_D, "r-")
plt.plot(sqz_test, "k.")


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
