#!/usr/bin/env python3

import os, sys
import numpy as np
import tensorflow as tf
import functools, itertools, scipy

from scipy import sparse
from scipy.integrate import solve_ivp
from squeezing_methods import spin_squeezing
from tensorflow_extension import tf_outer_product
from multibody_methods import dist_method

np.set_printoptions(linewidth = 200)

if len(sys.argv) < 4:
    print(f"usage: {sys.argv[0]} [alpha] [max_manifold] [lattice_shape] [proj?]")
    exit()

# determine whether to project operators onto the relevant manifolds
if "proj" in sys.argv:
    project = True
    sys.argv.remove("proj")
else:
    project = False

alpha = float(sys.argv[1]) # power-law couplings ~ 1 / r^\alpha
max_manifold = int(sys.argv[2])
lattice_shape = tuple(map(int, sys.argv[3:]))

# values of the ZZ coupling to inspect more closely
inspect_coupling_zz = [ -1, 0, 0.5, 1.5 ]
inspect_sim_time = 2

# values of the ZZ coupling to simulate in an XXZ model
sweep_coupling_zz = np.linspace(-1,3,41)

periodic = True # use periodic boundary conditions?

max_time = 10 # in units of J_\perp
ivp_tolerance = 1e-10 # error tolerance in the numerical integrator

data_dir = "../data/spins/"
proj_dir = "../data/projectors/"

def name_tag(coupling_zz = None):
    lattice_name = "_".join([ str(size) for size in lattice_shape ])
    base_tag = f"L{lattice_name}_M{max_manifold}_a{alpha}"
    if coupling_zz == None: return base_tag
    else: return base_tag + f"_z{coupling_zz}"

##################################################

lattice_dim = len(lattice_shape)
spin_num = np.product(lattice_shape)
manifolds = [ manifold for manifold in range(max_manifold+1) if manifold <= spin_num/2 ]
if np.allclose(alpha, int(alpha)): alpha = int(alpha)

assert(spin_num <= 12)
print("lattice shape:",lattice_shape)
##########################################################################################
print("reading in projectors onto manifolds of fixed net spin")
sys.stdout.flush()

projs = {}
for manifold in manifolds:
    proj_path = proj_dir + f"projector_N{spin_num}_M{manifold}.txt"
    if not os.path.isfile(proj_path):
        print(f"projector not found: {proj_path}")
        exit(1)
    projs[manifold] = scipy.sparse.dok_matrix((2**spin_num,)*2)
    with open(proj_path, "r") as f:
        for line in f:
            if "#" in line: continue
            row, col, val = line.split(", ")
            projs[manifold][int(row),int(col)] = float(val)
    projs[manifold] = projs[manifold].tocsr()

##########################################################################################
# define basic objects / operators

# qubit states and operators
up = np.array([1,0])
dn = np.array([0,1])
up_z, dn_z = up, dn
up_x = ( up + dn ) / np.sqrt(2)
dn_x = ( up - dn ) / np.sqrt(2)
up_y = ( up + 1j * dn ) / np.sqrt(2)
dn_y = ( up - 1j * dn ) / np.sqrt(2)

base_ops = {}
base_ops["I"] = np.outer(up_z,up_z.conj()) + np.outer(dn_z,dn_z.conj())
base_ops["Z"] = np.outer(up_z,up_z.conj()) - np.outer(dn_z,dn_z.conj())
base_ops["X"] = np.outer(up_x,up_x.conj()) - np.outer(dn_x,dn_x.conj())
base_ops["Y"] = np.outer(up_y,up_y.conj()) - np.outer(dn_y,dn_y.conj())

# act with a multi-local operator `op` on the spin specified by `indices`
def spin_op(op, indices = None):
    # make `op` and `indices` correspond to a single spin operator,
    #   make them single-item lists
    if type(op) is str:
        op = [ op ]
    if type(indices) is int:
        indices = [ indices ]

    # convert an array into a sparse tensor
    def _to_sparse_tensor(array):
        if type(array) is tf.sparse.SparseTensor: return array
        if scipy.sparse.issparse(array):
            _array_dok = array.todok()
            _indices, _values = zip(*_array_dok.items())
            _values = np.array(_values)
            return tf.SparseTensor(indices = _indices, values = _values,
                                   dense_shape = array.shape)
        if type(array) is np.ndarray:
            _indices = np.array(list(np.ndindex(array.shape)))
            return tf.SparseTensor(indices = _indices[array.flatten()!=0],
                                   values = array[array!=0].flatten(),
                                   dense_shape = array.shape)

    # convert a list of strings into a sparse tensor in which the tensor factors
    #   associated with each qubit are collected together and flattened
    if type(op) is list:
        op_spins = len(op)
        op = functools.reduce(tf_outer_product,
                              [ _to_sparse_tensor(base_ops[tag].flatten()) for tag in op ])
    else:
        op_spins = int(np.log2(np.prod(op.shape)) + 1/2) // 2
        op = tf.sparse.reshape(_to_sparse_tensor(op), (2,2,)*op_spins)

        fst_half = range(op_spins)
        snd_half = range(op_spins,2*op_spins)
        perm = np.array(list(zip(list(fst_half),list(snd_half)))).flatten()
        op = tf.sparse.transpose(op, perm)
        op = tf.sparse.reshape(op, (4,)*op_spins)

    if indices is None:
        # if we were not given indices, just return the multi-local operator
        indices = list(range(op_spins))
        total_spins = op_spins
    else:
        # otherwise, return an operator that acts on the indexed spins
        indices = list(indices)
        total_spins = spin_num
        for _ in range(total_spins - op_spins):
            op = tf_outer_product(op, _to_sparse_tensor(base_ops["I"].flatten()))

    # rearrange tensor factors according to the desired qubit order
    old_order = indices + [ jj for jj in range(total_spins) if jj not in indices ]
    new_order = np.arange(total_spins)[np.argsort(old_order)]
    op = tf.sparse.transpose(op, new_order)
    op = tf.sparse.reshape(op, (2,2,)*total_spins)

    # un-flatten the tensor factors, and flatten the tensor into
    #   a matrix that acts on the joint Hilbert space of all qubits
    evens = range(0,2*total_spins,2)
    odds = range(1,2*total_spins,2)
    op = tf.sparse.transpose(op, list(evens)+list(odds))
    op = tf.sparse.reshape(op, (2**total_spins,2**total_spins))

    # convert matrix into scipy's sparse matrix format
    return scipy.sparse.csr_matrix((op.values.numpy(), op.indices.numpy().T))

##########################################################################################
print("building operators")
sys.stdout.flush()

dist = dist_method(lattice_shape)
couplings_sun = { (pp,qq) : -1/dist(pp,qq)**alpha
                  for qq in range(spin_num) for pp in range(qq) }

swap = np.array([[ 1, 0, 0, 0 ],
                 [ 0, 0, 1, 0 ],
                 [ 0, 1, 0, 0 ],
                 [ 0, 0, 0, 1 ]])
H_0 = sum( coupling * spin_op(swap, pp_qq).real
           for pp_qq, coupling in couplings_sun.items() )

# note: factor of 1/2 included for compatibility with Chunlei's work
ZZ = sum( coupling * spin_op(["Z","Z"], pp_qq).real / 2
          for pp_qq, coupling in couplings_sun.items() )

def col_op(op):
    return sum( spin_op(op,idx) for idx in range(spin_num) )
Sz = col_op("Z") / 2
Sx = col_op("X") / 2
Sy = col_op("Y") / 2

S_op_vec = [ Sz, Sx, Sy ]
SS_op_mat = [ [ AA @ BB for BB in S_op_vec ] for AA in S_op_vec ]

if project:
    P_0 = sum( proj for proj in projs.values() )
    H_0 = P_0 @ H_0 @ P_0
    ZZ = P_0 @ ZZ @ P_0

##########################################################################################

# note: factor of 1/2 included for compatibility with Chunlei's work
chi_eff_bare = 1/4 * np.mean(list(couplings_sun.values()))
state_X = functools.reduce(np.kron, [up_x]*spin_num).astype(complex)

def simulate(coupling_zz, sim_time = None, max_tau = 2):
    print("coupling_zz:", coupling_zz)
    sys.stdout.flush()
    zz_sun_ratio = coupling_zz - 1
    assert(zz_sun_ratio != 0)

    H = H_0 + zz_sun_ratio * ZZ
    def _time_derivative(time, state):
        return -1j * ( H @ state.T )

    # determine how long to simulate
    if sim_time is None:
        chi_eff = abs(zz_sun_ratio * chi_eff_bare)
        sim_time = min(max_time, max_tau * spin_num**(-2/3) / chi_eff)

    # simulate!
    ivp_solution = solve_ivp(_time_derivative, (0, sim_time), state_X,
                             rtol = ivp_tolerance, atol = ivp_tolerance)

    times = ivp_solution.t
    states = ivp_solution.y
    sqz = np.array([ spin_squeezing(spin_num, state, S_op_vec, SS_op_mat)
                     for state in states.T ])

    # compute populations
    pops = np.array([ [ abs(states[:,tt].conj() @ proj @ states[:,tt])
                        for proj in projs.values() ]
                      for tt in range(len(times))])

    return times, sqz, pops

if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

if project: data_dir += "proj_"

##########################################################################################
print("running inspection simulations")
sys.stdout.flush()

def _is_zero(array):
    return np.allclose(array, np.zeros(array.shape))

for coupling_zz in inspect_coupling_zz:
    times, sqz, pops = simulate(coupling_zz, sim_time = inspect_sim_time)

    _manifolds = [ manifold for manifold in manifolds
                   if not _is_zero(pops[:,manifold]) ]

    with open(data_dir + f"inspect_{name_tag(coupling_zz)}.txt", "w") as file:
        file.write("# times, squeezing, populations (within each manifold)\n")
        for idx, manifold in enumerate(_manifolds):
            file.write(f"# manifold {manifold} : {idx}\n")
        for tt in range(len(times)):
            file.write(f"{times[tt]} {sqz[tt]} ")
            file.write(" ".join([ str(pops[tt,manifold]) for manifold in _manifolds ]))
            file.write("\n")

##########################################################################################
if len(sweep_coupling_zz) == 0: exit()
print("running sweep simulations")
sys.stdout.flush()

sweep_coupling_zz = sweep_coupling_zz[sweep_coupling_zz != 1]
sweep_results = [ simulate(coupling_zz) for coupling_zz in sweep_coupling_zz ]
sweep_times, sweep_sqz, sweep_pops = zip(*sweep_results)

sweep_min_sqz = [ min(sqz) for sqz in sweep_sqz ]
min_sqz_idx = [ max(1,np.argmin(sqz)) for sqz in sweep_sqz ]
sweep_time_opt = [ sweep_times[zz][idx] for zz, idx in enumerate(min_sqz_idx) ]

sweep_pops = [ pops[:min_idx,:] for pops, min_idx in zip(sweep_pops, min_sqz_idx) ]
sweep_min_pops = np.array([ pops.min(axis = 0) for pops in sweep_pops ])
sweep_max_pops = np.array([ pops.max(axis = 0) for pops in sweep_pops ])

_manifolds = [ manifold for manifold in manifolds
               if not _is_zero(sweep_max_pops[:,manifold]) ]

with open(data_dir + f"sweep_{name_tag()}.txt", "w") as file:
    file.write("# coupling_zz, sqz_min, time_opt, min_pop_0, max_pop (for manifolds > 0)\n")
    file.write("# manifolds : ")
    file.write(" ".join([ str(manifold) for manifold in manifolds ]))
    file.write("\n")
    for zz in range(len(sweep_coupling_zz)):
        file.write(f"{sweep_coupling_zz[zz]} {sweep_min_sqz[zz]} ")
        file.write(f"{sweep_time_opt[zz]} {sweep_min_pops[zz,0]} ")
        file.write(" ".join([ str(sweep_max_pops[zz,manifold])
                              for manifold in _manifolds[1:] ]))
        file.write("\n")

print("completed")
