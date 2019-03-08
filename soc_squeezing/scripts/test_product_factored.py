#!/usr/bin/env python3

import qutip as qt
import numpy as np
import itertools

from math import factorial
from scipy.special import binom

N = 3
op_cap = 3

I2 = qt.qeye(2)
sz = qt.sigmaz()/2
sp = qt.sigmap()
sm = qt.sigmam()
II = qt.tensor([ I2 ] * N)

def partition_1D(indices, partition_sizes):
    return [ indices[partition_sizes[:mu].sum() :
                     partition_sizes[:mu].sum() + partition_sizes[mu] ]
                     for mu in range(partition_sizes.size) ]

def partition_2D(indices, partition_sizes):
    split_indices = partition_1D(indices, partition_sizes.sum(1))
    return [ partition_1D(indices, partition_sizes[ii,:])
             for ii, indices in enumerate(split_indices) ]

def partition(indices, partition_sizes):
    assert(np.sum(partition_sizes) == len(indices))
    if partition_sizes.ndim == 1:
        return partition_1D(indices, partition_sizes)
    if partition_sizes.ndim == 2:
        return partition_2D(indices, partition_sizes)
    else:
        print("dimension of partition matrix too large")
        exit()

def spin_op(mu):
    if mu == 0: return sp
    if mu == 1: return sz
    if mu == 2: return sm
    if mu == 3: return I2

def PP(jj, mm):
    mm = np.array(mm)
    jj_vals = partition(jj,mm)
    op_list = [ I2 ] * N
    for mu in range(mm.size):
        for jj_mu in jj_vals[mu]:
            op_list[jj_mu] = spin_op(mu)
    return qt.tensor(op_list)

def SS(mm):
    op = 0 * II
    for jj in itertools.permutations(range(N), int(np.sum(mm))):
        op += PP(jj,mm)
    return op

def poch(nn, kk):
    return np.prod([ nn - cc for cc in range(kk) ])

def eta_val(mu, nu, rho):
    if spin_op(rho) == I2: fac = 1/2
    elif spin_op(rho) == sz: fac = 2
    else: fac = 1
    return ( spin_op(mu)*spin_op(nu) * spin_op(rho).dag() ).tr() * fac

eta = np.array([ [ [ eta_val(mu,nu,kk) for kk in range(4) ]
                   for nu in range(3) ]
                 for mu in range(3) ])
eta_mnk = np.array([ [ [ eta[mu,nu,kk] not in [ 0, 1 ] for kk in range(4) ]
                       for nu in range(3) ]
                     for mu in range(3) ])
eta_terms = eta[eta_mnk]

def r_mats(mm,nn,ss):
    return ( np.array([ [0,    r_pz, r_pm],
                        [r_zp, r_zz, r_zm],
                        [r_mp, r_mz, 0   ] ])
             for r_zz in range(ss+1)
             for r_zp in range(ss-r_zz+1)
             for r_pz in range(ss-r_zz-r_zp+1)
             for r_pm in range(min(ss-r_zz-r_zp,mm[0])-r_pz+1)
             for r_mp in range(min(ss-r_zz-r_pz-r_pm,nn[0])-r_zp+1)
             for r_zm in range(min(ss-r_zz-r_zp-r_pz-r_pm-r_mp,mm[1]-r_zz-r_zp,nn[2]-r_pm)+1)
             for r_mz in [ ss-r_zz-r_zp-r_pz-r_pm-r_mp-r_zm ]
             if r_mz + r_mp <= mm[2] and r_mz+r_pz+r_zz <= nn[1] )

def rho_mats(rr):
    return ( np.array([ [ [0,0,0,0],           [rr[0,1],0,0,0], [0,c_02_1,0,c_02_3] ],
                        [ [rr[1,0],0,0,0],     [0,0,0,rr[1,1]], [0,0,rr[1,2],0]     ],
                        [ [0,c_20_1,0,c_20_3], [0,0,rr[2,1],0], [0,0,0,0]           ] ])
             for c_02_1 in range(rr[0,2]+1)
             for c_20_1 in range(rr[2,0]+1)
             for c_02_3 in [ rr[0,2] - c_02_1 ]
             for c_20_3 in [ rr[2,0] - c_20_1 ] )

def product(mm,nn):
    op_prod = {}
    mn_ops = int(mm.sum()+nn.sum())
    min_overlap = max(mn_ops-N, 0)
    max_overlap = min(mn_ops, N)

    for ss in range(min_overlap, max_overlap+1):
        for rr in r_mats(mm,nn,ss):

            mnr_op_nums = mm + nn - rr.sum(0) - rr.sum(1)
            mnr_fac = np.prod([ poch(mm[mu],rr[mu,:].sum()) *
                                poch(nn[mu],rr[:,mu].sum())
                                for mu in range(3) ])

            for rho in rho_mats(rr):

                rho_kk = rho.sum((0,1))
                rho_op_nums = rho_kk[:-1]
                op_nums = mnr_op_nums + rho_op_nums

                id_ops = rho_kk[-1]
                ops = int(op_nums.sum())

                rho_fac = 1 / np.prod([ factorial(val) for val in rho.flatten() ])
                eta_fac = np.prod(eta_terms**rho[eta_mnk])
                id_fac = poch(N-ops,id_ops)
                fac = mnr_fac * rho_fac * eta_fac * id_fac

                try: op_prod[tuple(op_nums)] += fac
                except: op_prod[tuple(op_nums)] = fac

    return op_prod

def vec_to_mat(vec):
    terms = [ val * SS(op) for op, val in vec.items() ]
    if terms == []: return 0 * II
    else: return np.sum(terms)

for mm in itertools.product(range(op_cap+1), repeat = 3):
    mm = np.array(mm)
    SS_mm = SS(mm)
    for nn in itertools.product(range(op_cap+1), repeat = 3):
        nn = np.array(nn)
        op_full = SS_mm * SS(nn)
        op = vec_to_mat(product(mm,nn))
        print(mm, nn, op_full == op)
        if op_full != op: exit()
