#!/usr/bin/env python3

import os, sys, time
import numpy as np
import functools, scipy, sympy, itertools
from scipy import sparse

from operator_product_methods import multi_Z_op
from multibody_methods import get_multibody_states, sym_state, embed_operator

np.set_printoptions(linewidth = 200)
cutoff = 1e-10

if len(sys.argv) < 3:
    print(f"usage: {sys.argv[0]} [site_num] [manifolds]")
    exit()

site_num = int(sys.argv[1])
manifolds = [ int(mm) for mm in sys.argv[2:] ]

assert(site_num <= 16)
assert(max(manifolds) <= 4)

data_dir = "../data/projectors/"

lattice_shape = (site_num,)
sunc_mat = -np.ones((site_num,site_num))
sunc_mat -= np.diag(np.diag(sunc_mat))

if len(manifolds) == 1:
    manifolds = range(manifolds[0]+1)

manifolds = [ manifold for manifold in manifolds if manifold <= site_num/2 ]

print("site_num:",site_num)
print("manifolds:",manifolds)
##########################################################################################
print("computing eigenstates")
sys.stdout.flush()

manifold_shells, energies, tensors \
    = get_multibody_states(lattice_shape, sunc_mat, manifolds, TI = False)
shell_num = len(energies)

##########################################################################################
print("-"*50)
print("building projectors")
sys.stdout.flush()

projs = { manifold : np.zeros((2**site_num,)*2, dtype = complex)
          for manifold in manifolds }

multi_Z_ops = { manifold : multi_Z_op(manifold, diag = True)
                for manifold in manifolds }
def _multi_Z_op(manifold, sites):
    return embed_operator(multi_Z_ops[manifold], sites, site_num)

# build full projectors
for net_up in range(site_num+1):
    print(f" net_up: {net_up}/{site_num+1}")
    sys.stdout.flush()
    time_start = time.time()

    _sym_state = sym_state(net_up, site_num, normalize = False)
    for manifold, shells in manifold_shells.items():
        print(f"  manifold: {manifold}")
        sys.stdout.flush()
        for shell in shells:
            op = sum( tensors[shell][sites] * _multi_Z_op(manifold, sites)
                      for sites in itertools.combinations(range(site_num), manifold) )
            state = op * _sym_state
            norm = state.conj() @ state
            if not np.isclose(norm,0):
                projs[manifold] += np.outer(state, state.conj()) / norm

# clean up projectors and make them sparse
for manifold, proj in projs.items():
    zero = np.zeros(proj.shape)
    proj[np.isclose(proj,zero)] = 0
    if np.allclose(proj.imag,zero):
        proj = proj.real
    projs[manifold] = scipy.sparse.csr_matrix(proj)

##########################################################################################
print("-"*50)
print(f"writing projectors")
sys.stdout.flush()

if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

for manifold, proj in projs.items():
    if np.allclose(abs(proj).sum(), 0): break
    print(f" manifold: {manifold}")
    sys.stdout.flush()
    with open(data_dir + f"projector_N{site_num}_M{manifold}.txt", "w") as f:
        f.write(f"# projector onto collective shell number {manifold} of {site_num} spins\n")
        f.write(f"# represented by a matrix in the standard basis of {site_num} qubits\n")
        f.write("# row, column, value\n")
        for row in range(proj.shape[0]):
            for ind in range(proj.indptr[row], proj.indptr[row+1]):
                col, val = proj.indices[ind], proj.data[ind]
                if abs(val) < cutoff: continue
                f.write(f"{row}, {col}, {val}\n")
