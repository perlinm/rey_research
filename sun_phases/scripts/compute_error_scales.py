#!/usr/bin/env python3

import os, sys, time, functools
import scipy, scipy.linalg
import numpy as np

import wigner.lib, py3nj

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

# get diagonals and anti-diagonals of a matrix
def diagonals(mat):
    rows, cols = mat.shape
    fill = np.zeros(((cols - 1), cols), dtype = mat.dtype)
    stacked = np.vstack((mat, fill, mat))
    major_stride, minor_stride = stacked.strides
    strides = major_stride, minor_stride * (cols + 1)
    shape = (rows + cols - 1, cols)
    diags = np.lib.stride_tricks.as_strided(stacked, shape, strides)
    return np.roll(np.flipud(diags), 1, axis = 0)

# "invert" a matrix or vector
def invert(mat, axis = 0):
    ll = (mat.shape[0]-1)//2
    signs = np.array([ (-1)**mm for mm in range(-ll,ll+1) ])
    if mat.ndim == 1: return signs * np.flip(mat)
    if axis == 0:
        return signs[:,None] * np.flip(mat, axis)
    else:
        return signs[None,:] * np.flip(mat, axis)

# run a batch of independent jobs, possibly in parallel
def compute_batch(pool, function, args):
    args_list = list(args)
    if parallelize:
        values = pool.map(function, args_list)
    else:
        values = map(function, args_list)
    return { arg : value for arg, value in zip(args_list, values) }

##########################################################################################
# pre-compute matrices of Wigner-3j coefficients,
# which are used to build matrices of structure constants

def wigner_3j_mat(labels):
    LL, ll = labels
    matrix = np.zeros((2*ll+1,)*2)
    for mm in range(-ll,ll+1):
        start, end, vals = wigner.lib.wigner_3j_m(ll, ll, LL, mm)
        matrix[mm+ll, int(start)+ll : int(end)+ll+1] = vals
    return matrix
pool = multiprocessing.Pool(processes = cpus)
degree_labels = ( (LL,ll)
                  for ll in range(max_dim-1,-1,-1)
                  for LL in range(min(max_dim-1,2*ll),-1,-1) )
wigner_3j_mats = compute_batch(pool, wigner_3j_mat, degree_labels)

# compute a structure factor matrix
def struct_mat(dim, LL, ll):
    wigner_6j_args = [ 2*ll, 2*ll, 2*LL, dim-1, dim-1, dim-1 ]
    wigner_6j_factor = py3nj.wigner6j(*wigner_6j_args)
    prefactor = (-1)**(dim-1+LL) * (2*ll+1) * np.sqrt(2*LL+1) * wigner_6j_factor
    return prefactor * wigner_3j_mats[LL,ll]

##########################################################################################
# methods to compute a vector of D^L_{m,0}(v) for all |m| <= L,
# where: D^L_{mn}(v) = <Lm |R(v)|Ln> is a (Wigner) rotation matrix element
#        v = (alpha,beta) is a point on the sphere at azimuth/polar angles (alpha,beta)
#        R(v) is an Euler-angle rotation operator: exp(-i alpha S_z) * exp(-i beta S_y)
#        S_z and S_y are respectively spin-z and spin_y operators
#        |Lm> is a state of a spin-L particle with spin projection m onto the z axis
#
# the basic idea behind our methods is to decompose D^L_{m,0}(alpha,beta) into:
#     e^{-i alpha S_z} * exp(+i pi/2 S_y) * e^{-i beta S_z} * exp(-i pi/2 S_y) |L,0>

# pre-compute pi/2 rotation matrices exp(-i pi/2 S_y) for all L
def _spin_vals(LL):
    return np.arange(-LL, LL+1)
def _roz_z_to_x(LL):
    MM = _spin_vals(LL)[:-1]
    diag_vals = np.sqrt(LL*(LL+1)-MM*(MM+1))
    S_m = np.diag(diag_vals, 1)
    S_y = ( S_m - S_m.T ) * 1j/2
    return scipy.linalg.expm(-1j * np.pi/2 * S_y)
pool = multiprocessing.Pool(processes = cpus)
spin_vals = compute_batch(pool, _spin_vals, range(max_dim-1,-1,-1))
rot_z_to_x = compute_batch(pool, _roz_z_to_x, range(max_dim-1,-1,-1))

# pre-compute the vector `exp(-i pi/2 S_y) |L,0>` for all (relevant) L
# every other entry in this vector is zero, so remove it preemptively
rot_zero_vecs = [ rot_z_to_x[LL][:,LL][::2] for LL in range(max_dim) ]

# collect all exp(+i pi/2 S_y), skipping every other column because we don't need it
pulse_mats = [ rot_z_to_x[LL].T[:,::2] for LL in range(max_dim) ]

# construct the vector of D^L_{m,0}(v) for all |m| <= L,
# i.e. the "middle column" of the rotation matrix R(v) for a spin-L particle
def rot_mid(LL, axis):
    phases_0 = np.exp(-1j * axis[0] * spin_vals[LL][::2])
    phases_1 = np.exp(-1j * axis[1] * spin_vals[LL])
    return phases_1 * ( pulse_mats[LL] @ ( phases_0 * rot_zero_vecs[LL] ) )

##########################################################################################
# compute the measurement matrix, its singular values, and error scales

# construct a fixed-degree measurement matrix
def degree_meas_mat(LL, axes):
    return np.array([ rot_mid(LL, axis) for axis in axes ])

# get singular values ("norms") of a single fixed-degree measurement matrix
def degree_meas_norms(LL, axes):
    return get_svdvals(degree_meas_mat(LL, axes))

# get singular values ("norms") of all fixed-degree measurement matrices
pool = multiprocessing.Pool(processes = cpus)
def meas_norms(dim, axes):
    _degree_meas_norms = functools.partial(degree_meas_norms, axes = axes)
    return compute_batch(pool, _degree_meas_norms, range(dim-1,-1,-1))

# compute the classical error scale for a given set of axes
def classical_error_scale(dim, axes):
    norms = meas_norms(dim, axes)
    return np.sqrt(sum( sum(1/degree_norms**2)
                        for degree_norms in norms.values() ))

# squared "gamma" factors that appear in the quantum error scale
def sqr_gamma(dim, ll):
    ss = (dim-1)/2
    mm_min, mm_max, wigner_3j_vals = wigner.lib.wigner_3j_m(ll, ss, ss, 0)
    signs = np.array([ (-1)**(2*ll+ss-mm) for mm in np.arange(mm_min, mm_max+1) ])
    signed_vals = signs * wigner_3j_vals
    half_span = ( max(signed_vals) - min(signed_vals) ) / 2
    return (2*ll+1) * half_span**2

# compute the quantum error scale for a given set of axes
def quantum_error_scale(dim, axes):
    norms = meas_norms(dim, axes)
    return np.sqrt(sum( sqr_gamma(dim, ll) * sum(1/degree_norms**2)
                        for ll, degree_norms in norms.items() ))

##########################################################################################
# compute root-mean-squared reconstruction error

# compute the fixed-degree noise matrix
def degree_noise_mat(LL, axes):
    mat = degree_meas_mat(LL, axes).T
    _, vals, vecs = scipy.linalg.svd(mat, full_matrices = False)
    diag_vals = np.sum(abs(vecs/vals[:,None])**2, axis = 0)
    return ( mat * diag_vals[None,:] ) @ mat.conj().T

# compute all noise matrices
pool = multiprocessing.Pool(processes = cpus)
def noise_mats(axes):
    _degree_noise_mat = functools.partial(degree_noise_mat, axes = axes)
    return compute_batch(pool, _degree_noise_mat, range(dim-1,-1,-1))

# compute the fixed-degree "chi vector" in the "degree-order" basis
def degree_chi_state(LL, noise_mats):
    min_ll = (LL+1)//2
    dim = max(noise_mats.keys()) + 1
    chi_state = np.zeros(2*LL+1, dtype = complex)
    for ll, noise_mat in noise_mats.items():
        if ll < min_ll: continue
        mat = noise_mat.T * invert(struct_mat(dim, LL, ll))
        diag_mat = np.roll(diagonals(mat), LL, axis = 0)[:2*LL+1,:]
        chi_state += diag_mat.sum(axis = 1)
    return chi_state

# compute the full "chi vector" in the "degree-order" basis
pool = multiprocessing.Pool(processes = cpus)
def chi_state(noise_mats):
    _degree_chi_state = functools.partial(degree_chi_state, noise_mats = noise_mats)
    return compute_batch(pool, _degree_chi_state, range(dim-1,-1,-1))

# compute contribution to squared reconstruction error from a single degree
def sqr_degree_error(LL, state, chi_state, noise_mats):
    pos = chi_state[LL].conj() @ state[LL]
    neg = state[LL].conj() @ ( noise_mats[LL] @ state[LL] )
    return pos - neg

# compute the reconstruction error for a state in the "degree-order" basis
pool = multiprocessing.Pool(processes = cpus)
def error(state, axes):
    _noise_mats = noise_mats(axes)
    _chi_state = chi_state(_noise_mats)

    dim = max(state.keys()) + 1
    kwargs = dict( state = state, chi_state = _chi_state, noise_mats = _noise_mats )
    _sqr_degree_error = functools.partial(sqr_degree_error, **kwargs)
    sqr_degree_errors = compute_batch(pool, _sqr_degree_error, range(dim-1,-1,-1))

    return np.sqrt(sum( sqr_error for sqr_error in sqr_degree_errors.values() ))

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
        rnd_error_scale = classical_error_scale(dim, random_axes(axis_num))
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
