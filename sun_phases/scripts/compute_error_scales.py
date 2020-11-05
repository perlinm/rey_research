#!/usr/bin/env python3

import sys, functools, time
import scipy, scipy.linalg
import numpy as np

import multiprocessing
try:
    cpus = multiprocessing.cpu_count()
except NotImplementedError:
    cpus = 4 # default number of cpus to use for multithreading

np.set_printoptions(linewidth = 200)

parallelize = True
if "serial" in sys.argv:
    parallelize = False
    sys.argv.remove("serial")

dim = int(sys.argv[1])
try: seed = int(sys.argv[2])
except: seed = 0
np.random.seed(seed)

##########################################################################################
# general simulation options / methods

max_samples = 100 # maximum number of times we choose a random set of measurement axes
max_time = 300 # maximum time to run, in seconds

axis_num = 2*dim-1 # number of measurement axes
degrees = np.arange(dim-1,-1,-1) # degrees

# generate random points on the sphere by uniform sampling
def random_points(axis_num = axis_num):
    points = np.random.rand(axis_num,2)
    points[:,0] = np.arccos(2*points[:,0]-1)
    points[:,1] *= 2*np.pi
    return points

# get squared singular values of a matrix
def get_sqr_svdvals(matrix):
    # return scipy.linalg.svdvals(matrix)**2 # for some reason this is slower...
    if matrix.shape[0] < matrix.shape[1]:
        MM = matrix @ matrix.conj().T
    else:
        MM = matrix.conj().T @ matrix
    return scipy.linalg.eigvalsh(MM)

# run a batch of independent jobs, in parallel by default
def compute_batch(pool, function, values):
    if parallelize:
        return pool.map(function, values)
    else:
        return list(map(function, values))

##########################################################################################
# methods to compute a vector of D^L_{m,0}(v) for all |m| <= L,
# where: D^L_{mn} is a (Wigner) rotation matrix element, and
#        v = (alpha,beta) is a point on the sphere at azimuth/polar angles (alpha,beta)
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
spin_vals = compute_batch(pool, _spin_vals, degrees)[::-1]
rot_z_to_x = compute_batch(pool, _roz_z_to_x, degrees)[::-1]

# pre-compute the vector exp(-i \pi/2 S_y) |L,0> for all (relevant) L.
# every other entry in this vector is zero, so remove it preemptively
rot_zero_vecs = { degree : rot_z_to_x[degree][:,degree][::2] for degree in range(dim) }

# pre-compute exp(+i \pi/2 S_y), skipping every other column
pulse_mats = { degree : rot_z_to_x[degree].T[:,::2] for degree in range(dim) }

# construct the vector of D^L_{m,0}(v) for all |m| <= L
def rot_mid(degree, point):
    phases_0 = np.exp(-1j * point[0] * spin_vals[degree][::2])
    phases_1 = np.exp(-1j * point[1] * spin_vals[degree])
    return phases_1 * ( pulse_mats[degree] @ ( phases_0 * rot_zero_vecs[degree] ) )

##########################################################################################
# compute the measurement matrix and its singular values

# construct a fixed-degree measurement matrix
def axes_trans_ops(degree, points):
    return np.array([ rot_mid(degree, point) for point in points ])

# get (squared) singular values ("norms") of a fixed-degree measurement matrix
def degree_norms(degree, points):
    return get_sqr_svdvals(axes_trans_ops(degree, points))

# get (squared) singular values ("norms") of the full measurement matrix
pool = multiprocessing.Pool(processes = cpus)
def meas_norms(points):
    _degree_norms = functools.partial(degree_norms, points = points)
    return np.concatenate(compute_batch(pool, _degree_norms, degrees))

# compute the (squared) error scale
def error_scale(points):
    points_mat_shape = (points.size//2,2)
    points_mat = points.reshape(points_mat_shape)
    norms = meas_norms(points_mat)
    return abs(sum(1/norms))

##########################################################################################
# simulate!

samples = 0
start = time.time()
min_error_scale = np.inf

while samples < max_samples and time.time() - start < max_time:
    rnd_error_scale = error_scale(random_points())
    min_error_scale = min(min_error_scale, rnd_error_scale)
    samples += 1

mean_time = ( time.time() - start ) / samples
print(samples)
print(mean_time)
print(min_error_scale)
