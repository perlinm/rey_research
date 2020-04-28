#!/usr/bin/env python3

import numpy as np
import itertools, numpy.matlib, scipy.optimize
from squeezing_methods import squeezing_from_correlators

# correlators and squeezing behavior of the Ising model with power-law interactions

def ising_spin_correlators(times, sunc_mat, TI = False):
    spin_num = sunc_mat.shape[0]
    np.fill_diagonal(sunc_mat,0)

    type_times = type(times)
    if not type_times is np.ndarray:
        times = np.array(times, ndmin = 1)

    Sz = np.zeros(times.shape)
    Sz_Sz = spin_num/4 * np.ones(times.shape)

    angles = np.einsum("t,ij->tij", times, sunc_mat)
    _cos_angles = np.cos(angles)
    _cos_prod = np.prod(_cos_angles, axis = -1)
    if not TI:
        Sp = np.einsum("ti->t", _cos_prod) / 2

        _tan_angles = np.tan(angles)
        sp_sz_sum = np.einsum("ti->t", np.einsum("tij->ti", _tan_angles) * _cos_prod)

        _angles_ext = np.einsum("tij,o->tioj", angles, np.ones(spin_num))
        _angles_mod = np.einsum("tioj->toij", _angles_ext)
        _cos_angles_plus = np.cos(_angles_ext + _angles_mod)
        _cos_angles_diff = np.cos(_angles_ext - _angles_mod)
        _cos_angles_sqr = _cos_angles**2
        _cos_prod_plus = np.prod(_cos_angles_plus, axis = -1) / _cos_angles_sqr
        _cos_prod_diff = np.prod(_cos_angles_diff, axis = -1) / _cos_angles_sqr

        sp_sp_sum = ( + np.einsum("tij->t", _cos_prod_plus)
                      - np.einsum("tii->t", _cos_prod_plus) )
        sp_sm_sum = ( + np.einsum("tij->t", _cos_prod_diff)
                      - np.einsum("tii->t", _cos_prod_diff) )

    else:
        Sp = spin_num/2 * _cos_prod[:,0]

        _tan_angles = np.tan(angles[:,:,0])
        sp_sz_sum = spin_num * np.einsum("ti->t", _tan_angles * _cos_prod)

        _angles_mod = np.einsum("tj,o->toj", angles[:,:,0], np.ones(spin_num))
        _cos_angles_plus = np.cos(angles + _angles_mod)
        _cos_angles_diff = np.cos(angles - _angles_mod)
        _cos_angles_sqr = _cos_angles[:,:,0]**2
        _cos_prod_plus = np.prod(_cos_angles_plus, axis = -1) / _cos_angles_sqr
        _cos_prod_diff = np.prod(_cos_angles_diff, axis = -1) / _cos_angles_sqr

        sp_sp_sum = spin_num * ( np.einsum("ti->t", _cos_prod_plus) - _cos_prod_plus[:,0] )
        sp_sm_sum = spin_num * ( np.einsum("ti->t", _cos_prod_diff) - _cos_prod_diff[:,0] )

    Sp_Sz = -1/2 * Sp + 1j/4 * sp_sz_sum
    Sp_Sp = 1/4 * sp_sp_sum
    Sp_Sm = 1/4 * sp_sm_sum + Sz + spin_num/2

    correlators = { (0,1,0) : Sz,
                    (0,2,0) : Sz_Sz,
                    (1,0,0) : Sp,
                    (2,0,0) : Sp_Sp,
                    (1,1,0) : Sp_Sz,
                    (1,0,1) : Sp_Sm }

    if type_times is np.ndarray:
        return correlators
    else:
        return { op : val[0] for op, val in correlators.items() }

def ising_squeezing_function(sunc_mat, TI = False):
    spin_num = sunc_mat.shape[0]
    def correlators(time):
        return ising_spin_correlators(time, sunc_mat, TI)
    def squeezing(time):
        return squeezing_from_correlators(spin_num, correlators(time))
    return np.vectorize(squeezing)

def ising_squeezing_optimum(sunc_mat, TI = False):
    spin_num = sunc_mat.shape[0]
    squeezing = ising_squeezing_function(sunc_mat, TI)

    time_guess = spin_num**(-2/3)
    minimum = scipy.optimize.minimize(squeezing, time_guess)
    opt_time = minimum.x[0]
    min_sqz = minimum.fun
    return opt_time, min_sqz

def ising_minimal_SS(sunc_mat, TI = False, trials = 10):
    spin_num = sunc_mat.shape[0]
    opt_time, _ = ising_squeezing_optimum(sunc_mat, TI)

    def SS(time):
        correlators = ising_spin_correlators(time, sunc_mat, TI)
        return ( + correlators[(0,2,0)].real
                 + correlators[(1,0,1)].real
                 - correlators[(0,1,0)].real )
    SS = np.vectorize(SS)

    trial_times = np.linspace(0, opt_time, trials)
    trial_values = SS(trial_times)
    time_guess = trial_times[np.argmin(trial_values)]

    bound = (0,opt_time)
    minimum = scipy.optimize.minimize(SS, time_guess, bounds = [bound])
    min_time = minimum.x[0]
    min_SS = minimum.fun[0]
    return min_time, min_SS
