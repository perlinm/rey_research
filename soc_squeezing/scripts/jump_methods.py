#!/usr/bin/env python3

# FILE CONTENTS: methods monte carlo simulations via the quantum jump method

import sys
import numpy as np
import scipy.sparse as sparse
import scipy.interpolate as interpolate

from scipy.integrate import solve_ivp

from dicke_methods import spin_op_z_dicke, spin_op_m_dicke
from correlator_methods import get_dec_vecs, vec_zxy_to_pzm, squeezing_ops


# squared norm of vector
def sqr_norm(vec):
    return abs(vec.conj() @ vec)

# single-spin pauli operators
s_i = np.array([[1,0],[0,1]])
s_z = np.array([[1,0],[0,-1]]) / 2
s_p = np.array([[0,1],[0,0]])
s_m = np.array([[0,0],[1,0]])

# collective S_z and S_m operators for net spin J
def base_ops(J, base_ops = {}):
    if base_ops.get(J) is None:
        base_ops[J] = [ spin_op_z_dicke(int(2*J)),
                        spin_op_m_dicke(int(2*J)) ]
    return base_ops[J]

# collective operators of the form S_+^ll S_z^mm S_-^nn
def op_mat(J, op):
    S_z, S_m = base_ops(J)
    return S_m.T**op[0] * S_z**op[1] * S_m**op[2]

# matrix representation of sinple-spin jump operator \gamma
def decoherence_op(dec_vec):
    return dec_vec[0] * s_p + dec_vec[1] * s_z + dec_vec[2] * s_m

# \gamma^\dag \gamma for jump operator \gamma
def gg_mat(dec):
    g_vec = decoherence_op(dec)
    return g_vec.conj().T @ g_vec

# \sum_\gamma \gamma^\dag \gamma over all jump operators \gamma
def eff_jump_ops(N, J, dec_vecs):
    S_z, S_m = base_ops(J)
    I = sparse.identity(int(2*J)+1)
    return ( sum( np.trace(op.conj().T @ gg_mat(dec_vec)) * OP
                  for op, OP in [ (s_i/2,N*I), (2*s_z,S_z),
                                  (s_p,S_m.T), (s_m,S_m) ] )
             for dec_vec in dec_vecs )

# build Hamiltonian for coherent evolution
def build_H(op_mats, h_pzm):
    return sum( val * op_mats[op] for op, val in h_pzm.items() )

# build effective Hamiltonian
def build_H_eff(N, J, op_mats, h_pzm, dec_vecs):
    H_eff = ( build_H(op_mats, h_pzm)
              - 1j/2 * sum( op for op in eff_jump_ops(N, J, dec_vecs) ) )
    return H_eff.todia()

# jump transition amplitudes
def P_JM(N, J, M, d_J, d_M):
    assert(d_J == 0 or d_J == -1 or d_J == 1)
    assert(d_M == 0 or d_M == -1 or d_M == 1)
    if J == 0 and d_J != 1: return 0

    if d_J == 0:
        if d_M == 0: coeff = M
        else: coeff = np.sqrt((J-d_M*M) * (J+d_M*M+1))
        P_JM = np.sqrt( (2+N) / (4*J*(J+1)) ) * coeff

    if d_J == -1:
        if d_M == 0: coeff = (J-M)*(J+M)
        else: coeff = (J-d_M*M) * (J-d_M*M-1)
        sign = -1 if d_M == -1 else 1
        P_JM = sign * np.sqrt( (N+2*J+2) / (4*J*(2*J+1)) * coeff )

    if d_J == 1:
        if d_M == 0: coeff = (J-M+1)*(J+M+1)
        else: coeff = (J+d_M*M+1) * (J+d_M*M+2)
        sign = -1 if d_M == 1 else 1
        P_JM = sign * np.sqrt( (N-2*J) / (4*(J+1)*(2*J+1)) * coeff )

    return P_JM

# jump transition amplitude vectors
def P_J(N, J, d_J, d_M):
    return np.array([ P_JM(N, J, -J+mm, d_J, d_M) for mm in range(int(2*J+1)) ])

# transform a state according to a quantum jump
def jump_state(state, d_J, dec_op, P_J_mat):
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

    return state_new / np.sqrt(state_new.conj() @ state_new)

# compute expectation value of given operator with respect to a given state
def correlator(op_mat, state):
    J = (state.size-1)/2
    return state.conj() @ ( op_mat @  state ) / sqr_norm(state)

# choose an index at random, given some probability distribution
def choose_index(probs):
    probs /= probs.sum()
    choice = np.random.rand()
    idx = 0
    while choice > probs[:idx+1].sum():
        idx += 1
    return idx

# compute correlators via the quantum jump method
def correlators_from_trajectories(spin_num, trajectories, chi_times, initial_state, h_vec,
                                  dec_rates = [], dec_mat = None, seed = 0,
                                  default_savepoints = True, print_updates = True,
                                  ivp_tolerance = 1e-15, solver = None):
    np.random.seed(seed)
    if solver is None: ivp_solver = "RK45"
    else: ivp_solver = solver

    if print_updates:
        from time import time as current_time

    max_time = chi_times[-1]
    if dec_mat is None: dec_mat = np.eye(3)
    dec_vecs = get_dec_vecs(dec_rates, dec_mat)
    for dec_vec in dec_vecs:
        if np.sum(dec_vec[1]) != 0:
            print("quantum jumps with collective decoherence not yet implemented!")
            exit()
    dec_vecs = [ vecs[0] for vecs in dec_vecs ]

    all_ops = set(squeezing_ops).union(set(h_vec.keys())) # all operators we care about

    correlator_mat_shape = (trajectories,len(squeezing_ops),chi_times.size)
    correlator_mat = np.zeros(correlator_mat_shape, dtype = complex)

    for trajectory in range(trajectories):

        if print_updates:
            print(f"{trajectory}/{trajectories}")
            sys.stdout.flush()
            start_time = current_time()

        J = spin_num/2
        state = initial_state
        op_mats = { op : op_mat(J, op) for op in all_ops }
        H_eff = build_H_eff(spin_num, J, op_mats, h_vec, dec_vecs)

        time = 0
        jumps = 0
        while time < max_time:

            # construct time derivative operator
            def time_derivative(time, state):
                return -1j * ( H_eff @ state )

            # construct dec event that returns zero when we do a jump
            jump_norm = np.random.random()
            def dec_event(time, state):
                return sqr_norm(state) - jump_norm
            dec_event.terminal = True
            dec_event.direction = -1

            # simulate until a jump occurs
            if default_savepoints:
                ivp_solution = solve_ivp(time_derivative, (time, max_time), state,
                                         events = dec_event, method = ivp_solver,
                                         rtol = ivp_tolerance, atol = ivp_tolerance)
                times = ivp_solution.t
                states = ivp_solution.y

                # set current time and state
                time = times[-1]
                state = states[:,-1] / np.sqrt(sqr_norm(states[:,-1]))

                # compute correlators at save points
                correlators = np.zeros((len(squeezing_ops),times.size), dtype = complex)
                for tt in range(times.size):
                    for op_idx, sqz_op in enumerate(squeezing_ops):
                        correlators[op_idx,tt] = correlator(op_mats[sqz_op], states[:,tt])

                # determine indices times at which we wish to save correlators
                interp_time_idx = (chi_times >= times[0]) & (chi_times <= times[-1])
                interp_times = chi_times[interp_time_idx]

                # compute correlators at desired times by interpolation
                for op_idx in range(len(squeezing_ops)):
                    op_interp = interpolate.interp1d(times,correlators[op_idx,:])
                    correlator_mat[trajectory,op_idx,interp_time_idx] \
                        = op_interp(interp_times)

            else: # not default savepoints
                eval_times = chi_times[chi_times >= time]
                add_first_time = time not in eval_times
                if add_first_time:
                    eval_times = [ time ] + list(eval_times)
                ivp_solution = solve_ivp(time_derivative, (time, max_time), state,
                                         events = dec_event, method = ivp_solver,
                                         rtol = ivp_tolerance, atol = ivp_tolerance,
                                         t_eval = eval_times)
                times = ivp_solution.t[add_first_time:]
                states = ivp_solution.y[:,add_first_time:]

                # set current time and state
                last_time = ivp_solution.t[-1]
                last_state = ivp_solution.y[:,-1]
                if ivp_solution.status != 0:
                    time = ivp_solution.t_events[0][0]
                    state = solve_ivp(time_derivative, (last_time, time), last_state,
                                      events = dec_event, method = ivp_solver,
                                      rtol = ivp_tolerance, atol = ivp_tolerance,
                                      t_eval = (last_time, time)).y[:,-1]
                    state /= np.sqrt(sqr_norm(state))

                # compute correlators at save points
                for tt in range(times.size):
                    time_idx = np.argmax(chi_times == times[tt])
                    for op_idx, sqz_op in enumerate(squeezing_ops):
                        correlator_mat[trajectory,op_idx,time_idx] \
                            = correlator(op_mats[sqz_op], states[:,tt])

            # print status info
            if print_updates:
                print(" ", jumps, time, time/max_time, current_time() - start_time)
                sys.stdout.flush()

            # if the ivp solver terminated by reaching the maximum time, then we're done
            if ivp_solution.status == 0: break
            else: jumps += 1 # otherwise, we're doing another jump

            # compute jump probabilities for each operater and change in net spin J
            P_J_mat = { (d_J,d_M) : P_J(spin_num,J,d_J,d_M)
                        for d_J in [1,0,-1] for d_M in [1,0,-1] }
            probs = np.zeros((len(dec_vecs),3))
            for dec_idx, dec_vec in enumerate(dec_vecs):
                for d_J in [ 1, 0, -1 ]:
                    probs[dec_idx,1-d_J] \
                        = sum( sqr_norm(dec_vec[1-d_M] * P_J_mat[d_J,d_M] * state)
                               for d_M in [1,0,-1] if dec_vec[1-d_M] != 0 )

            # choose which decoherence operator to use
            dec_choice = choose_index(probs.sum(1))
            dec_op = dec_vecs[dec_choice]

            # choose which branch of J --> J+1, J, J-1 to follow
            d_J = 1 - choose_index(probs[dec_choice,:])
            J += d_J

            # recompute matrix representations of all relevant operators
            op_mats = { op : op_mat(J, op) for op in all_ops }

            # transform state according to jump, and rebuild Hamiltonian if necessary
            state = jump_state(state, d_J, dec_op, P_J_mat)
            if d_J != 0: H_eff = build_H_eff(spin_num, J, op_mats, h_vec, dec_vecs)

    # average over all trajectories
    correlators = { sqz_op : correlator_mat[:,op_idx,:].mean(0)
                    for op_idx, sqz_op in enumerate(squeezing_ops) }

    if print_updates:
        print(f"{trajectories}/{trajectories}", current_time() - start_time)

    return correlators
