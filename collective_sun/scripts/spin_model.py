#!/usr/bin/env python3

import os
import numpy as np
import tensorflow as tf
import functools, itertools, scipy
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.integrate import solve_ivp
from squeezing_methods import spin_squeezing
from tensorflow_extension import tf_outer_product
from multibody_methods import dist_method

np.set_printoptions(linewidth = 200)

lattice_shape = (2,4)
alpha = 3 # power-law couplings ~ 1 / r^\alpha
periodic = True
project_hamiltonian = True

# values of the ZZ coupling to simulate in an XXZ model
sweep_coupling_zz = np.linspace(-2,4,25)

# values of the ZZ coupling to inspect more closely
inspect_coupling_zz = [ -1 ]

max_time = 10 # in units of J_\perp

ivp_tolerance = 1e-10 # error tolerance in the numerical integrator

data_dir = "../data/projectors/"
fig_dir = "../figures/spins/"

figsize = (5,4)
params = { "font.size" : 16,
           "text.usetex" : True,
           "text.latex.preamble" : [ r"\usepackage{amsmath}",
                                     r"\usepackage{braket}" ]}
plt.rcParams.update(params)

##################################################

if type(lattice_shape) is int:
    lattice_shape = (lattice_shape,)
lattice_dim = len(lattice_shape)
spin_num = np.product(lattice_shape)
assert(spin_num <= 12)

##########################################################################################
print("reading in projectors onto manifolds of fixed net spin")

projs = {}
for manifold in range(3):
    proj_path = data_dir + f"projector_N{spin_num}_M{manifold}.txt"
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

dist = dist_method(lattice_shape)

# qubit states and operators
up = np.array([1,0])
dn = np.array([0,1])
up_z, dn_z = up, dn
up_x = ( up + dn ) / np.sqrt(2)
dn_x = ( up - dn ) / np.sqrt(2)
up_y = ( up + 1j * dn ) / np.sqrt(2)
dn_y = ( up - 1j * dn ) / np.sqrt(2)

base_ops = {}
base_ops["I"] = np.outer(up_z,up_z) + np.outer(dn_z,dn_z)
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

if project_hamiltonian:
    P_0 = sum( proj for proj in projs.values() )
    H_0 = P_0 @ H_0 @ P_0
    ZZ = P_0 @ ZZ @ P_0

def col_op(op):
    return sum( spin_op(op,idx) for idx in range(spin_num) )
Sz = col_op("Z") / 2
Sx = col_op("X") / 2
Sy = col_op("Y") / 2

S_op_vec = [ Sz, Sx, Sy ]
SS_op_mat = [ [ AA @ BB for BB in S_op_vec ] for AA in S_op_vec ]

##########################################################################################

# note: factor of 1/2 included for compatibility with Chunlei's work
chi_eff_bare = 1/4 * np.mean(list(couplings_sun.values()))
state_X = functools.reduce(np.kron, [up_x]*spin_num).astype(complex)

def simulate(coupling_zz, max_tau = 2, overshoot_ratio = 1.5):
    print("coupling_zz:", coupling_zz)
    zz_sun_ratio = coupling_zz - 1

    H = H_0 + zz_sun_ratio * ZZ
    def _time_derivative(time, state):
        return -1j * ( H @ state )

    # determine how long to simulate
    if zz_sun_ratio != 0:
        chi_eff = abs(zz_sun_ratio * chi_eff_bare)
        sim_time = min(max_time, max_tau * spin_num**(-2/3) / chi_eff)
    else:
        sim_time = max_time

    # simulate!
    ivp_solution = solve_ivp(_time_derivative, (0, sim_time), state_X,
                             rtol = ivp_tolerance, atol = ivp_tolerance)

    times = ivp_solution.t
    states = ivp_solution.y
    sqz = np.array([ spin_squeezing(spin_num, state, S_op_vec, SS_op_mat)
                     for state in states.T ])

    # don't look too far beyond the maximum squeezing time
    max_tt = int( np.argmin(sqz) * overshoot_ratio )
    if max_tt == 0:
        max_tt = len(times)
    else:
        max_tt = min(max_tt, len(times))

    times = times[:max_tt]
    sqz = sqz[:max_tt]

    # compute populations
    pops = { manifold : np.array([ abs(states[:,tt].conj() @ proj @ states[:,tt])
                               for tt in range(len(times)) ])
             for manifold, proj in projs.items() }
    pops["\mathrm{ext}"] = 1 - sum( pop for pop in pops.values() )

    return times, sqz, pops

def name_tag(coupling_zz = None):
    base_tag = f"N{spin_num}_D{lattice_dim}_a{alpha}"
    if coupling_zz == None: return base_tag
    else: return base_tag + f"_z{coupling_zz}"

def pop_label(manifold, prefix = None):
    label = r"$\braket{\mathcal{P}_{" + str(manifold) + r"}}$"
    if prefix == None:
        return label
    else:
        return prefix + " " + label

def to_dB(sqz):
    return 10*np.log10(np.array(sqz))

if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)
if project_hamiltonian: fig_dir += "proj_"

##########################################################################################
print("running inspection simulations")

for coupling_zz in inspect_coupling_zz:
    times, sqz, pops = simulate(coupling_zz)

    title_text = f"$N={spin_num},~D={lattice_dim},~\\alpha={alpha}," \
               + f"~J_{{\mathrm{{z}}}}/J_\perp={coupling_zz}$"

    plt.figure(figsize = figsize)
    plt.title(title_text)
    plt.plot(times, to_dB(sqz), "k")
    plt.ylim(plt.gca().get_ylim()[0], 0)
    plt.xlabel(r"time ($J_\perp t$)")
    plt.ylabel(r"$\xi_{\mathrm{min}}^2$ (dB)")
    plt.tight_layout()

    plt.savefig(fig_dir + f"squeezing_{name_tag(coupling_zz)}.pdf")

    plt.figure(figsize = figsize)
    plt.title(title_text)
    for manifold, pops in pops.items():
        plt.plot(times, pops, label = pop_label(manifold))
    plt.axvline(times[np.argmin(sqz)], color = "gray", linestyle  = "--")
    plt.xlabel(r"time ($J_\perp t$)")
    plt.ylabel("population")
    plt.legend(loc = "best")
    plt.tight_layout()

    plt.savefig(fig_dir + f"populations_{name_tag(coupling_zz)}.pdf")

##########################################################################################
print("running sweep simulations")

sweep_coupling_zz = sweep_coupling_zz[sweep_coupling_zz != 1]
sweep_results = [ simulate(coupling_zz) for coupling_zz in sweep_coupling_zz ]
sweep_times, sweep_sqz, sweep_pops = zip(*sweep_results)

sweep_min_sqz = [ min(sqz) for sqz in sweep_sqz ]
min_sqz_idx = [ np.argmin(sqz) for sqz in sweep_sqz ]

title_text = f"$N={spin_num},~D={lattice_dim},~\\alpha={alpha}$"

plt.figure(figsize = figsize)
plt.title(title_text)
plt.plot(sweep_coupling_zz, to_dB(sweep_min_sqz), "ko")
plt.ylim(plt.gca().get_ylim()[0], 0)
plt.xlabel(r"$J_{\mathrm{z}}/J_\perp$")
plt.ylabel(r"$\xi_{\mathrm{min}}^2$ (dB)")
plt.tight_layout()
plt.savefig(fig_dir + f"squeezing_N{spin_num}_D{lattice_dim}_a{alpha}.pdf")

manifolds = sweep_pops[0].keys()
sweep_points = len(sweep_coupling_zz)

sweep_pops = { manifold : [ sweep_pops[jj][manifold][:min_sqz_idx[jj]]
                            for jj in range(sweep_points) ]
               for manifold in manifolds }
sweep_min_pops = { manifold : [ min(sweep_pops[manifold][jj])
                                for jj in range(sweep_points) ]
                   for manifold in manifolds }
sweep_max_pops = { manifold : [ max(sweep_pops[manifold][jj])
                                for jj in range(sweep_points) ]
                   for manifold in manifolds }

plt.figure(figsize = figsize)
plt.title(title_text)
plt.plot(sweep_coupling_zz, sweep_min_pops[0], "o", label = pop_label(0,"min"))
for manifold, max_pops in sweep_max_pops.items():
    if manifold == 0: continue
    plt.plot(sweep_coupling_zz, max_pops, "o", label = pop_label(manifold,"max"))
plt.xlabel(r"$J_{\mathrm{z}}/J_\perp$")
plt.ylabel("population")
plt.legend(loc = "best")
plt.tight_layout()
plt.savefig(fig_dir + f"populations_N{spin_num}_D{lattice_dim}_a{alpha}.pdf")
