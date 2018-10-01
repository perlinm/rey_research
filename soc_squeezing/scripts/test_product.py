#!/usr/bin/env python3

import qutip as qt
import numpy as np
import itertools

from math import factorial
from scipy.special import binom

N = 3

I2 = qt.qeye(2)
sz = qt.sigmaz()
sp = qt.sigmap()
sm = qt.sigmam()
II = qt.tensor([ I2 ] * N)

def partition_1D(indices, partition_sizes):
    return [ indices[partition_sizes[:mu].sum() :
                     partition_sizes[:mu].sum() + partition_sizes[mu] ]
                     for mu in range(3) ]

def partition_2D(indices, partition_sizes):
    split_indices = partition_1D(indices, partition_sizes.sum(1))
    return [ partition_1D(split_indices[ii], partition_sizes[ii,:])
             for ii in range(len(split_indices)) ]

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

def poch(nn, kk, vals = {}):
    return np.prod([ nn - cc for cc in range(kk) ])

def f(mm,nn,rr):
    mnb_fac = np.prod([ poch(mm[mu],rr[mu,:].sum()) * poch(nn[mu],rr[:,mu].sum())
                       for mu in range(3) ])
    rr_fac = np.prod([ factorial(rr[mu,nu]) for mu in range(3) for nu in range(3) ])
    return mnb_fac / rr_fac

def g(cc,rr_mu_nu):
    return factorial(rr_mu_nu) / np.prod([ factorial(cc_jj) for cc_jj in cc  ])

def eta(mu, nu, rho, vals = {}):
    try: return vals[mu,nu,rho]
    except:
        fac = 2 if rho in [1,3] else 1
        vals[mu,nu,rho] = ( spin_op(mu)*spin_op(nu) * spin_op(rho).dag() ).tr() / fac
        return vals[mu,nu,rho]

def PP(jj,mm):
    jj_vals = partition(jj,mm)
    op_list = [ I2 ] * N
    for mu in range(3):
        for jj_mu in jj_vals[mu]:
            op_list[jj_mu] = spin_op(mu)
    return qt.tensor(op_list)

def KK_cc(KK,mu,nu,cc,kk):
    return KK[mu][nu][cc[:kk].sum():cc[:kk].sum()+cc[kk]]

def cc_vals(rr_mu_nu):
    return ( np.array([c_0,c_1,c_2,rr_mu_nu-c_0-c_1-c_2])
             for c_0 in range(rr_mu_nu+1)
             for c_1 in range(rr_mu_nu-c_0+1)
             for c_2 in range(rr_mu_nu-c_0-c_1+1) )

def QQ(KK,rr):
    KK_mat = partition(KK,rr)

    QQ = II
    for mu, nu in itertools.product(range(3), repeat = 2):
        QQ_mu_nu = 0 * II
        for cc in cc_vals(rr[mu,nu]):
            cc_ops = [ I2 ] * N
            g_fac = g(cc,rr[mu,nu])
            eta_fac = 1
            for kk in range(4):
                eta_fac *= eta(mu,nu,kk)**cc[kk]
                for idx in KK_cc(KK_mat,mu,nu,cc,kk):
                    cc_ops[idx] = spin_op(kk)
            QQ_mu_nu += g_fac * eta_fac * qt.tensor(cc_ops)
        QQ *= QQ_mu_nu

    return QQ

    # op_list = [ I2 ] * N
    # for mu, nu in itertools.product(range(3), repeat = 2):
    #     for KK_mu_nu in KK_mat[mu][nu]:
    #         op_list[KK_mu_nu] = spin_op(mu) * spin_op(nu)

    # QQ = qt.tensor(op_list)
    # return QQ

def SS(mm):
    op = 0 * II
    for jj in itertools.permutations(range(N), int(np.sum(mm))):
        op += PP(jj,mm)
    return op

def r_mats(mm,nn,ss):
    return ( np.array([ [0,    r_01, r_02],
                        [r_10, r_11, r_12],
                        [r_20, r_21, 0   ] ])
             for r_11 in range(ss+1)
             for r_01 in range(ss-r_11+1)
             for r_10 in range(ss-r_11-r_01+1)
             for r_12 in range(min(ss-r_01,mm[1])-r_10-r_11+1)
             for r_21 in range(min(ss-r_10-r_12,nn[1])-r_11-r_01+1)
             for r_02 in range(min(ss-r_11-r_01-r_10-r_12-r_21,mm[0]-r_01,nn[2]-r_12)+1)
             for r_20 in [ ss-r_11-r_01-r_10-r_12-r_21-r_02 ]
             if r_20 + r_10 <= nn[0] and r_20 + r_21 <= mm[2] )

op_cap = 2

for mm in itertools.product(range(op_cap+1), repeat = 3):
    if mm == (0,0,0): continue
    mm = np.array(mm)
    SS_mm = SS(mm)
    for nn in itertools.product(range(op_cap+1), repeat = 3):
        if nn == (0,0,0): continue
        nn = np.array(nn)
        op = SS_mm * SS(nn)
        op_test = 0 * II

        mn_ops = int(mm.sum()+nn.sum())
        min_overlap = max(mn_ops-N, 0)
        max_overlap = mn_ops
        for ss in range(min_overlap, max_overlap+1):
            ops = max_overlap - ss

            for rr in r_mats(mm,nn,ss):
                f_val = f(mm,nn,rr)

                for JK in itertools.permutations(range(N), ops):
                    mn = mm + nn - rr.sum(0) - rr.sum(1)

                    if mn.sum() > 0:
                        JJ = JK[:mn.sum()]
                        PP_JJ = PP(JJ,mn)
                    else:
                        PP_JJ = II
                    if mn.sum() < len(JK):
                        KK = JK[mn.sum():]
                        QQ_KK = QQ(KK,rr)
                    else:
                        QQ_KK = II

                    op_test += f_val * PP_JJ * QQ_KK

        print(mm, nn, op == op_test)
        if op != op_test: exit()
