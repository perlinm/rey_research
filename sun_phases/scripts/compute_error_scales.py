#!/usr/bin/env python3

import os, sys, functools, time
import scipy, scipy.linalg
import sympy, sympy.physics.quantum
import numpy as np

import multiprocessing
try:
    cpus = multiprocessing.cpu_count()
except NotImplementedError:
    cpus = 4 # default number of cpus to use for multithreading

np.set_printoptions(linewidth = 200)

write_data = False

parallelize = False
if "par" in sys.argv:
    parallelize = True
    sys.argv.remove("par")

min_dim = int(sys.argv[1])
try: max_dim = int(sys.argv[2])
except: max_dim = min_dim
try: seed = int(sys.argv[3])
except: seed = 0
np.random.seed(seed)

data_dir = "../data/error_scale/"
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

par_tag = f"c{cpus}" if parallelize else "serial"
data_file = data_dir + f"times_d{min_dim}-{max_dim}_{par_tag}.txt"

genesis = time.time()

##########################################################################################
# general simulation options / methods

sample_cap = 100 # maximum number of times we choose a random set of measurement axes
time_cap = 300 # maximum time to run, in seconds

# generate random axes on the sphere by uniform sampling
def random_axes(axis_num):
    axes = np.random.rand(axis_num,2)
    axes[:,0] = np.arccos(2*axes[:,0]-1) # polar angles
    axes[:,1] *= 2*np.pi # azimuthal angles
    return axes

# get squared singular values of a matrix
def get_svdvals(matrix):
    # return scipy.linalg.svdvals(matrix) # for some reason this is slower...
    if matrix.shape[0] < matrix.shape[1]:
        MM = matrix @ matrix.conj().T
    else:
        MM = matrix.conj().T @ matrix
    return np.sqrt(scipy.linalg.eigvalsh(MM))

# run a batch of independent jobs, in parallel by default
def compute_batch(pool, function, values):
    if parallelize:
        return pool.map(function, values)
    else:
        return list(map(function, values))

##########################################################################################
# methods to compute a vector of D^L_{m,0}(v) for all |m| <= L,
# where: D^L_{mn} is a (Wigner) rotation matrix element, and
#        v = (alpha,beta) is a axis on the sphere at azimuth/polar angles (alpha,beta)
#
# basically, we decompose D^L_{m,0}(alpha,beta) into:
#     e^{-i alpha S_z} * exp(+i \pi/2 S_y) * e^{-i beta S_z} * exp(-i \pi/2 S_y) |L,0>
# where: S_z, S_y are spin-z and spin_y operators
#        |L,0> is the state of a spin-L particle with spin projection 0 onto the z axis

# pre-compute pi/2 rotation matrices
def _spin_vals(LL):
    return np.arange(-LL, LL+1)
def _roz_z_to_x(LL):
    MM = _spin_vals(LL)[:-1]
    diag_vals = np.sqrt(LL*(LL+1)-MM*(MM+1))
    S_m = np.diag(diag_vals, 1)
    S_y = ( S_m - S_m.T ) * 1j/2
    return scipy.linalg.expm(-1j * np.pi/2 * S_y)
pool = multiprocessing.Pool(processes = cpus)
spin_vals = compute_batch(pool, _spin_vals, range(max_dim-1,-1,-1))[::-1]
rot_z_to_x = compute_batch(pool, _roz_z_to_x, range(max_dim-1,-1,-1))[::-1]

# pre-compute the vector exp(-i \pi/2 S_y) |L,0> for all (relevant) L
# every other entry in this vector is zero, so remove it preemptively
rot_zero_vecs = [ rot_z_to_x[LL][:,LL][::2] for LL in range(max_dim) ]

# pre-compute exp(+i \pi/2 S_y), skipping every other column
pulse_mats = [ rot_z_to_x[LL].T[:,::2] for LL in range(max_dim) ]

# construct the vector of D^L_{m,0}(v) for all |m| <= L
def rot_mid(LL, axis):
    phases_0 = np.exp(-1j * axis[0] * spin_vals[LL][::2])
    phases_1 = np.exp(-1j * axis[1] * spin_vals[LL])
    return phases_1 * ( pulse_mats[LL] @ ( phases_0 * rot_zero_vecs[LL] ) )

##########################################################################################
# pre-compute weights on fixed-degree variances

def get_gamma(dim, LL):
    II = ( sympy.S(dim)-1 ) / 2
    cg_coef = lambda *args : float(sympy.physics.quantum.cg.CG(*args).doit())
    cg_vals = [ cg_coef(II, -II+mm, LL, 0, II,-II+mm)
                for mm in range(dim-1) ]
    return ( max(cg_vals) - min(cg_vals) ) / 2
variance_weight = { ( dim, LL ) : get_gamma(dim, LL)**2 * (2*LL+1)/dim
                    for dim in range(min_dim, max_dim+1)
                    for LL in range(1,dim) }

##########################################################################################
# compute the measurement matrix, its singular values, and the error scale

# construct a fixed-degree measurement matrix
def axes_trans_ops(LL, axes):
    return np.array([ rot_mid(LL, axis) for axis in axes ])

# get singular values ("norms") of a single fixed-degree measurement matrix
def degree_norms(LL, axes):
    return get_svdvals(axes_trans_ops(LL, axes))

# get singular values ("norms") of all fixed-degree measurement matrix
pool = multiprocessing.Pool(processes = cpus)
def meas_norms(dim, axes):
    _degree_norms = functools.partial(degree_norms, axes = axes)
    return compute_batch(pool, _degree_norms, range(dim-1,0,-1))

# compute the error scale for the given axes
def error_scale(dim, axes):
    axes_mat_shape = (axes.size//2,2)
    axes_mat = axes.reshape(axes_mat_shape)
    norms = meas_norms(dim, axes_mat)
    weights = [ variance_weight[dim,LL] for LL in range(dim-1,0,-1) ]
    return np.sqrt(sum([ weight * sum(1/degree_norms**2)
                         for weight, degree_norms in zip(weights, norms) ]))

##########################################################################################
# simulate!

if write_data:
    with open(data_file, "w") as file:
        file.write(f"# sample_cap: {sample_cap}\n")
        file.write(f"# time_cap: {time_cap} sec\n")
        file.write(f"# seed: {seed}\n")
        file.write("# dim, mean_time, min_error_scale\n")

for dim in range(min_dim, max_dim+1):
    print(dim, end = " ")

    start = time.time()
    min_error_scale = np.inf

    axis_num = 2*dim-1 # minimum number of axes
    for sample in range(sample_cap):
        rnd_error_scale = error_scale(dim, random_axes(axis_num))
        min_error_scale = min(min_error_scale, rnd_error_scale)
        if time.time() - start > time_cap: break

    mean_time = ( time.time() - start ) / (sample+1)
    if write_data:
        with open(data_file, "a") as file:
            file.write(f"{dim} {mean_time} {min_error_scale}\n")

    print(mean_time, min_error_scale)
    sys.stdout.flush()

runtime = time.time() - genesis
runtime_text = f"total runtime: {runtime}"
print(runtime_text)
if write_data:
    with open(data_file, "a") as file:
        file.write(f"# {runtime_text}\n")
