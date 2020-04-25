#!/usr/bin/env python3

import os, sys
import numpy as np
import tensorflow as tf
import functools, itertools, scipy

from scipy import sparse
from scipy.integrate import solve_ivp
from squeezing_methods import squeezing_from_correlators
from tensorflow_extension import tf_outer_product
from multibody_methods import dist_method

np.set_printoptions(linewidth = 200)

if len(sys.argv) < 4:
    print(f"usage: {sys.argv[0]} [proj?] [lattice_shape] [alpha] [max_manifold]")
    exit()

# determine whether to project operators onto the relevant manifolds
if "proj" in sys.argv:
    project = True
    sys.argv.remove("proj")
else:
    project = False

lattice_shape = tuple(map(int, sys.argv[1].split("x")))
alpha = float(sys.argv[2]) # power-law couplings ~ 1 / r^\alpha
max_manifold = int(sys.argv[3])

# values of the ZZ coupling to simulate in an XXZ model
zz_couplings = np.arange(-3, +4.01, 0.1)

# values of the ZZ coupling to inspect more closely: half-integer values
inspect_zz_couplings = [ zz_coupling for zz_coupling in zz_couplings
                         if np.allclose(zz_coupling % 0.5, 0)
                         or np.allclose(zz_coupling % 0.5, 0.5) ]
inspect_sim_time = 2

periodic = True # use periodic boundary conditions?

max_time = 10 # in units of J_\perp
ivp_tolerance = 1e-10 # error tolerance in the numerical integrator

data_dir = "../data/spins/"
proj_dir = "../data/projectors/"

if np.allclose(alpha, int(alpha)): alpha = int(alpha)
lattice_name = "x".join([ str(size) for size in lattice_shape ])
name_tag = f"L{lattice_name}_a{alpha}_M{max_manifold}"

##################################################

lattice_dim = len(lattice_shape)
spin_num = np.product(lattice_shape)
manifolds = [ manifold for manifold in range(max_manifold+1) if manifold <= spin_num/2 ]
if np.allclose(alpha, int(alpha)): alpha = int(alpha)

zz_couplings = [ zz for zz in zz_couplings if not np.allclose(zz,1) ]
inspect_zz_couplings = [ zz for zz in inspect_zz_couplings if not np.allclose(zz,1) ]

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
dn = np.array([1,0])
up = np.array([0,1])

base_ops = {}
base_ops["I"] = np.outer(up,up.conj()) + np.outer(dn,dn.conj())
base_ops["Z"] = np.outer(up,up.conj()) - np.outer(dn,dn.conj())
base_ops["+"] = np.outer(up,dn.conj())

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
    return scipy.sparse.csr_matrix((op.values.numpy(), op.indices.numpy().T),
                                   shape = (2**spin_num,)*2)

##########################################################################################
print("building operators")
sys.stdout.flush()

dist = dist_method(lattice_shape)
sunc_mat = np.zeros((spin_num,spin_num))
for pp, qq in itertools.combinations(range(spin_num),2):
    sunc_mat[pp,qq] = sunc_mat[qq,pp] = -1/dist(pp,qq)**alpha

swap = np.array([[ 1, 0, 0, 0 ],
                 [ 0, 0, 1, 0 ],
                 [ 0, 1, 0, 0 ],
                 [ 0, 0, 0, 1 ]])
H_0 = sum( sunc_mat[pp_qq] * spin_op(swap, pp_qq).real
           for pp_qq in itertools.combinations(range(spin_num),2) )

ZZ = sum( sunc_mat[pp_qq] * spin_op(["Z","Z"], pp_qq).real / 2
          for pp_qq in itertools.combinations(range(spin_num),2) )

def col_op(op):
    return sum( spin_op(op,idx) for idx in range(spin_num) )
collective_ops = { "Z" : col_op("Z"),
                   "+" : col_op("+") }
collective_ops["ZZ"] = collective_ops["Z"] @ collective_ops["Z"]
collective_ops["++"] = collective_ops["+"] @ collective_ops["+"]
collective_ops["+Z"] = collective_ops["+"] @ collective_ops["Z"]
collective_ops["+-"] = collective_ops["+"] @ collective_ops["+"].conj().T

if project:
    P_0 = sum( proj for proj in projs.values() )
    H_0 = P_0 @ H_0 @ P_0
    ZZ = P_0 @ ZZ @ P_0

##########################################################################################

chi_eff_bare = sunc_mat.sum() / (spin_num * (spin_num-1))
state_X = functools.reduce(np.kron, [up+dn]*spin_num).astype(complex) / 2**(spin_num/2)

def simulate(zz_coupling, sim_time = None, max_tau = 2):
    print("zz_coupling:", zz_coupling)
    sys.stdout.flush()
    zz_sun_ratio = zz_coupling - 1
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

    # compute collective spin correlators
    def val(mat, state):
        return state.conj() @ ( mat @ state ) / ( state.conj() @ state )
    correlators = { op : np.array([ val(mat,state) for state in states.T ])
                    for op, mat in collective_ops.items() }

    # compute populations
    pops = np.array([ [ abs(states[:,tt].conj() @ proj @ states[:,tt])
                        for proj in projs.values() ]
                      for tt in range(len(times))])

    return times, correlators, pops

if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

if project: data_dir += "proj_"

##########################################################################################
print("running inspection simulations")
sys.stdout.flush()

def _is_zero(array):
    return np.allclose(array, np.zeros(array.shape))

str_ops = [ "Z", "+", "ZZ", "++", "+Z", "+-" ]
tup_ops = [ (0,1,0), (1,0,0), (0,2,0), (2,0,0), (1,1,0), (1,0,1) ]
def relabel(correlators):
    return { tup_op : correlators[str_op] for tup_op, str_op in zip(tup_ops, str_ops) }

str_op_list = ", ".join(str_ops)

for zz_coupling in inspect_zz_couplings:
    times, correlators, pops = simulate(zz_coupling, sim_time = inspect_sim_time)
    sqz = squeezing_from_correlators(spin_num, relabel(correlators), pauli_ops = True)

    _manifolds = [ manifold for manifold in manifolds
                   if not _is_zero(pops[:,manifold]) ]

    with open(data_dir + f"inspect_{name_tag}_z{zz_coupling:.1f}.txt", "w") as file:
        file.write(f"# times, {str_op_list}, sqz, populations (within each shell)\n")
        for idx, manifold in enumerate(_manifolds):
            file.write(f"# manifold {manifold} : {idx}\n")
        for tt in range(len(times)):
            file.write(f"{times[tt]} ")
            file.write(" ".join([ str(correlators[op][tt]) for op in str_ops ]))
            file.write(f" {sqz[tt]} ")
            file.write(" ".join([ str(pops[tt,manifold]) for manifold in _manifolds ]))
            file.write("\n")

##########################################################################################
if len(zz_couplings) == 0: exit()
print("running sweep simulations")
sys.stdout.flush()

results = [ simulate(zz_coupling) for zz_coupling in zz_couplings ]
sweep_times, sweep_correlators, sweep_pops = zip(*results)

print("computing squeezing values")
sys.stdout.flush()

sweep_sqz = [ squeezing_from_correlators(spin_num, relabel(correlators), pauli_ops = True)
              for correlators in sweep_correlators ]
min_sqz_vals = [ min(sqz) for sqz in sweep_sqz ]
min_sqz_idx = [ max(1,np.argmin(sqz)) for sqz in sweep_sqz ]
opt_time_vals = [ sweep_times[zz][tt] for zz, tt in enumerate(min_sqz_idx) ]

min_SS_vals = [ min( correlators["ZZ"][:tt].real/4 + correlators["+-"][:tt].real )
                for correlators, tt in zip(sweep_correlators, min_sqz_idx) ]

pop_vals = [ pops[:tt,:] for pops, tt in zip(sweep_pops, min_sqz_idx) ]
min_pop_vals = np.array([ pops.min(axis = 0) for pops in pop_vals ])
max_pop_vals = np.array([ pops.max(axis = 0) for pops in pop_vals ])

str_op_opt_list = ", ".join([ op + "_opt" for op in str_ops ])
opt_correlators = [ { op : correlator[op][tt] for op in str_ops }
                    for correlator, tt in zip(sweep_correlators, min_sqz_idx) ]

_manifolds = [ manifold for manifold in manifolds
               if not _is_zero(max_pop_vals[:,manifold]) ]

print("saving results")
sys.stdout.flush()

with open(data_dir + f"sweep_{name_tag}.txt", "w") as file:
    file.write(f"# zz_coupling, time_opt, sqz_min, {str_op_opt_list}, SS_min, "
               + "min_pop_0, max_pop (for manifolds > 0)\n")
    file.write("# manifolds : ")
    file.write(" ".join([ str(manifold) for manifold in _manifolds ]))
    file.write("\n")
    for zz in range(len(zz_couplings)):
        file.write(f"{zz_couplings[zz]} {opt_time_vals[zz]} ")
        file.write(f"{min_sqz_vals[zz]} {min_SS_vals[zz]} ")
        file.write(" ".join([ str(opt_correlators[zz][op]) for op in str_ops ]))
        file.write(f" {min_pop_vals[zz,0]} ")
        file.write(" ".join([ str(max_pop_vals[zz,manifold])
                              for manifold in _manifolds[1:] ]))
        file.write("\n")

print("completed")
