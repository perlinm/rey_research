#!/usr/bin/env python3

import qutip as qt
import numpy as np
import itertools

from math import factorial
from scipy.special import binom

N = 4

I2 = qt.qeye(2)
sz = qt.sigmaz()
sp = qt.sigmap()
sm = qt.sigmam()
II = qt.tensor([ I2 ] * N)

def spin_op(mu):
    if mu == 0: return sp
    if mu == 1: return sz
    if mu == 2: return sm

def ss(comb,ll,mm,nn,jj):
    for kk in range(ll+mm+nn):
        if jj == comb[kk]:
            if kk < ll:
                return sp
            if kk < ll + mm:
                return sz
            return sm
    return I2

def SS(ll,mm,nn):
    op = 0 * II
    if ll < 0 or mm < 0 or nn < 0: return op
    for comb in itertools.permutations(range(N), ll+mm+nn):
        op += qt.tensor([ ss(comb,ll,mm,nn,jj) for jj in range(N) ])
    return op

def PP(mm,jj):
    assert(mm.sum() == len(jj))
    jj_vals = [ jj[np.sum(mm[:mu]):np.sum(mm[:mu])+mm[mu]] for mu in range(3) ]
    op_list = []
    for nn in range(N):
        found = False
        for mu in range(3):
            if nn in jj_vals[mu]:
                op_list += [ spin_op(mu) ]
                found = True
                break
        if not found: op_list += [ I2 ]
    return qt.tensor(op_list)

def QQ(rr,jj):
    assert(rr.sum() == len(jj))
    jj_vals = [ [ jj[rr[:mu,:].sum()+rr[mu,:nu].sum():rr[:mu,:].sum()+rr[mu,:nu].sum()+rr[mu,nu]]
                  for nu in range(3) ]
                for mu in range(3) ]
    op_list = []
    for nn in range(N):
        for mu, nu in itertools.product(range(3), repeat = 2):
            found = False
            if nn in jj_vals[mu][nu]:
                op_list += [ spin_op(mu) * spin_op(nu) ]
                found = True
                break
        if not found: op_list += [ I2 ]
    return qt.tensor(op_list)

def SS(mm):
    op = 0 * II
    for jj in itertools.permutations(range(N), int(np.sum(mm))):
        op += PP(mm,jj)
    return op

def poch(nn, kk, vals = {}):
    return np.prod([ nn - cc for cc in range(kk) ])

def g(mm,nn,rr):
    mnb_fac = np.prod([ poch(mm[mu],rr[mu,:].sum()) * poch(nn[mu],rr[:,mu].sum())
                       for mu in range(3) ])
    rr_fac = np.prod([ factorial(rr[mu,nu]) for mu in range(3) for nu in range(3) ])
    return mnb_fac / rr_fac

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
                g_val = g(mm,nn,rr)

                for jj in itertools.permutations(range(N), ops):
                    ll = mm + nn - rr.sum(0) - rr.sum(1)
                    jj_single = jj[:ll.sum()]
                    jj_double = jj[ll.sum():]
                    assert(len(jj_single) == mm.sum()+nn.sum()-2*ss)
                    assert(len(jj_double) == ss)

                    op_single = PP(ll,jj_single)
                    op_double = QQ(rr,jj_double)

                    op_test += g_val * op_single * op_double

        print(mm, nn, op == op_test)
        if op != op_test: exit()
