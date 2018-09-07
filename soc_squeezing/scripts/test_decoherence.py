#!/usr/bin/env python3

from pylab import *
import qutip as qt
import itertools

from random import random
from scipy.special import factorial, binom

N = 6
S = N/2

I2 = qt.qeye(2)
II = qt.tensor([I2]*N)

s_z_j = []
s_p_j = []
s_m_j = []
for jj in range(N):
    s_z_j.append(qt.tensor([ I2 ] * jj + [ qt.sigmaz() ] + [ I2 ] * (N-jj-1)))
    s_p_j.append(qt.tensor([ I2 ] * jj + [ qt.sigmap() ] + [ I2 ] * (N-jj-1)))
    s_m_j.append(qt.tensor([ I2 ] * jj + [ qt.sigmam() ] + [ I2 ] * (N-jj-1)))

S_p = sum(s_p_j)
S_m = sum(s_m_j)
S_z = sum(s_z_j) / 2

def acomm(A,B):
    return A*B + B*A

def D_full(jump_ops, op_in):
    return sum([ jump.dag() * op_in * jump - 1/2 * acomm(jump.dag() * jump, op_in)
                 for jump in jump_ops ])

def ops(mu):
    if mu == 1:
        return S_p, S_m, s_p_j, s_m_j
    else:
        return S_m, S_p, s_m_j, s_p_j

def check(X):
    if not X:
        print("FAIL")
        exit()


g_z, g_p, g_m = int(random()*(N-2)), int(random()*(N-2)), int(random()*(N-2))
g_z += 1
g_p += 1
g_m += 1

g_mp = np.conj(g_m) * g_p
g_zp = np.conj(g_z) * g_p
g_mz = np.conj(g_m) * g_z
gg_p = ( g_zp + g_mz ) / 2
gg_m = ( g_zp - g_mz ) / 2

print(g_z, g_p, g_m)
print()


for l, m, n in itertools.product(range(N), repeat = 3):

    for mu in [ 1, -1 ]:
        print(l,m,n,f"{mu:2d}")

        S_mu, S_nu, s_mu_j, s_nu_j = ops(mu)

        op = S_mu**l * S_z**m * S_nu**n
        g_ops = [ g_z * s_z_j[jj] + g_p * s_mu_j[jj] + g_m * s_nu_j[jj] for jj in range(N) ]
        G_op = g_z * S_z + g_p * S_mu + g_m * S_nu

        a = D_full(g_ops, op)
        A = D_full([G_op], op)

        ##############################

        d_p = S_mu**l * ( (S-l-n-mu*S_z) * sum( mu**(m-k) * binom(m,k) * S_z**k
                                                for k in range(m) )
                          - 1/2*(l+n) * S_z**m ) * S_nu**n
        if l >= 1 and n >= 1:
            d_p += l*n*(2*S-l-n+2) * S_mu**(l-1) * S_z**m * S_nu**(n-1)
        if l >= 2 and n >= 2:
            d_p += ( l*n*(l-1)*(n-1) * S_mu**(l-2) * (S+mu*S_z)
                     * sum( (-mu)**(m-k) * binom(m,k) * S_z**k for k in range(m+1) )
                     * S_nu**(n-2) )

        ##############################

        d_m = S_mu**l * ( (S+mu*S_z) * sum( (-mu)**(m-k) * binom(m,k) * S_z**k
                                            for k in range(m) )
                          - 1/2 * (l+n) * S_z**m ) * S_nu**n

        ##############################

        d_z = -2*(l+n) * S_mu**l * S_z**m * S_nu**n
        if l >= 1 and n >= 1:
            d_z += ( 4*l*n * S_mu**(l-1) * (S+mu*S_z)
                     * sum( (-mu)**(m-k) * binom(m,k) * S_z**k for k in range(m+1) )
                     * S_nu**(n-1) )

        ##############################

        def K(l,m,n,mu):
            return mu/2 * S_mu**(l+1) * ( (mu+S_z)**m - S_z**m ) * S_nu**n

        def L(l,m,n,mu):
            L = 0*II
            if n >= 1:
                L += mu*n * S_mu**l * (-2*S+2*l+3/2*(n-1)+mu*S_z) * S_z**m * S_nu**(n-1)
            if n >= 2 and l >= 1:
                L += -2*mu*l*n*(n-1) * S_mu**(l-1) * (S+mu*S_z) * (-mu+S_z)**m * S_nu**(n-2)
            return L

        def M(l,m,n,mu):
            M = 0*II
            if n >= 1:
                M += mu*n * ( S_mu**l
                              * ( 2*(S+mu*S_z) * (-mu+S_z)**m - ((n-1)/2+mu*S_z) * S_z**m)
                              * S_nu**(n-1) )
            return M

        def P(l,m,n,mu):
            P = 0 * II
            if n >= 1:
                P += n * S_mu**(l+1) * S_z**m * S_nu**(n-1)
            if n >= 2:
                P += -n*(n-1) * S_mu**l * (S+mu*S_z) * (-mu+S_z)**m * S_nu**(n-2)
            return P

        def Q(l,m,n,mu):
            gg_P = np.conj(g_m) * g_p
            gg_L = np.conj(g_z) * g_p
            gg_M = np.conj(g_m) * g_z
            gg_K = gg_L + gg_M
            return ( gg_P * P(l,m,n,mu)
                     + gg_K * K(l,m,n,mu)
                     + gg_L * L(l,m,n,mu)
                     + gg_M * M(l,m,n,mu) )

        ##############################

        b = abs(g_z)**2 * d_z + abs(g_p)**2 * d_p + abs(g_m)**2 * d_m
        b += Q(l,m,n,mu) + Q(n,m,l,mu).dag()

        check(a == b)

        ############################################################

        D_p = S_mu**(l+1) * ((2*mu+S_z)**m-(mu+S_z)**m) * S_nu**(n+1)
        D_p += -S_mu**l * ( l*(l+1) + n*(n+1) + 2*mu*(l+n+1)*S_z ) * (mu+S_z)**m * S_nu**n
        D_p += 1/2 * S_mu**l * ( l*(l+1) + n*(n+1) + 2*mu*(l+n+2)*S_z ) * S_z**m * S_nu**n
        if l >= 1 and n >= 1:
            D_p += ( l*n * S_mu**(l-1)
                     * ( (l-1)*(n-1) + 2*mu*(l+n-2)*S_z + 4*S_z**2 )
                     * S_z**m * S_nu**(n-1) )

        D_m = -S_mu**(l+1) * ((mu+S_z)**m-S_z**m) * S_nu**(n+1)
        D_m += S_mu**l * ( 1/2*l*(l-1) + 1/2*n*(n-1) + mu*(l+n)*S_z ) * S_z**m * S_nu**n

        D_z = -1/2 * (l-n)**2 * S_mu**l * S_z**m * S_nu**n

        ##############################

        def L(l,m,n,mu):
            L = mu * ( (l-n+1/2) * gg_p + (l+1/2) * gg_m ) * S_mu**(l+1) * (mu+S_z)**m * S_nu**n
            L += -mu * ( (l-n+1/2) * gg_p + (n+1/2) * gg_m ) * S_mu**(l+1) * S_z**m * S_nu**n
            L += gg_m * S_mu**(l+1) * S_z * ( (mu+S_z)**m - S_z**m ) * S_nu**n
            return L

        def M(l,m,n,mu):
            if n == 0: return 0
            M = ( mu * (n-1) * ( (l-n+1/2) * gg_p + (l-1/2) * gg_m )
                  * S_mu**l * S_z**m * S_nu**(n-1) )
            M += ( 2 * ( (l-n+1/2) * gg_p + (l+n/2-1) * gg_m )
                   * S_mu**l * S_z**(m+1) * S_nu**(n-1) )

            M += 2 * mu * gg_m * S_mu**l * S_z**(m+2) * S_nu**(n-1)
            return M

        def P(l,m,n,mu):
            P = -1/2 * S_mu**(l+2) * ( (2*mu+S_z)**m - 2*(mu+S_z)**m + S_z**m ) * S_nu**n
            if n >= 1:
                P += n * ( S_mu**(l+1)
                           * ( (n+2*mu*S_z) * (mu+S_z)**m - (n-1+2*mu*S_z) * S_z**m ) *
                           S_nu**(n-1) )
            if n >= 2:
                P += -n*(n-1) * ( S_mu**l
                                  * ( 1/2*(n-1)*(n-2) + mu*(2*n-3)*S_z + 2*S_z**2 )
                                  * S_z**m * S_nu**(n-2) )
            return P

        def Q(l,m,n,mu):
            return L(l,m,n,mu) - n * M(l,m,n,mu) + np.conj(g_m) * g_p * P(l,m,n,mu)

        ##############################

        B = abs(g_z)**2 * D_z + abs(g_p)**2 * D_p + abs(g_m)**2 * D_m
        B += Q(l,m,n,mu) + Q(n,m,l,mu).dag()

        check(A == B)
