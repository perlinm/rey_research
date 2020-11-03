#!/usr/bin/env python3

import os, sys, functools
import numpy as np
import matplotlib.pyplot as plt
import scipy, scipy.optimize

from dicke_methods import spin_op_z_dicke, spin_op_y_dicke
from multilevel_methods import transition_op

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

def parallel(pool, function, values):
    if parallelize:
        return pool.map(function, values)
    else:
        return list(map(function, values))

####################

samples = 100
axis_num = 2*dim-1
degrees = np.arange(dim-1,-1,-1)

zhat = [1,0,0]
xhat = [0,1,0]
yhat = [0,0,1]

def to_axis(point):
    theta, phi = point
    return [ np.cos(theta),
             np.sin(theta) * np.cos(phi),
             np.sin(theta) * np.sin(phi) ]
def to_angles(vec):
    return [ np.arccos( vec[0] / np.sqrt(vec @ vec) ),
             np.arctan2( vec[2], vec[1] ) ]

# spin projections and pi/2 rotation matrices
def _spin_vals(degree):
    return spin_op_z_dicke(2*degree).diagonal()
def _roz_z_to_x(degree):
    Sy = spin_op_y_dicke(2*degree)
    return scipy.linalg.expm(-1j*np.pi/2*Sy.todense()).T
pool = multiprocessing.Pool(processes = cpus)
spin_vals = parallel(pool, _spin_vals, degrees)[::-1]
rot_z_to_x = parallel(pool, _roz_z_to_x, degrees)[::-1]

# construct the middle column of a rotation matrix
def rot_z(angle, degree):
    return np.exp(-1j*angle*spin_vals[degree])
def rot_y_mid(angle, degree):
    return rot_z_to_x[degree].T @ ( rot_z(angle, degree) * rot_z_to_x[degree][:,degree] )
def rot_mid(point, degree):
    return rot_z(point[1], degree) * rot_y_mid(point[0], degree)

# compute all rotated (order-0) transition operators of a given degree
def axes_trans_ops(degree, points):
    return np.array([ rot_mid(point, degree) for point in points ])

# get squared singular values of a matrix, in decreasing order
def get_sqr_svdvals(matrix):
    # return scipy.linalg.svdvals(matrix)**2 # for some reason this is slower...
    if matrix.shape[0] < matrix.shape[1]:
        MM = matrix @ matrix.conj().T
    else:
        MM = matrix.conj().T @ matrix
    return scipy.linalg.eigvalsh(MM)[::-1]

# get squared singular values ("norms") of a fixed-degree measurement matrix
def degree_sqr_norms(degree, points):
    return get_sqr_svdvals(axes_trans_ops(degree, points))

# get squared singular values ("norms") of the full measurement matrix
pool = multiprocessing.Pool(processes = cpus)
def meas_sqr_norms(points):
    _degree_sqr_norms = functools.partial(degree_sqr_norms, points = points)
    return np.concatenate(parallel(pool, _degree_sqr_norms, degrees))

# compute the squared error norm
def sqr_error_norm(points):
    points_mat_shape = (points.size//2,2)
    points_mat = points.reshape(points_mat_shape)
    sqr_norms = meas_sqr_norms(points_mat)
    return abs(sum(1/sqr_norms))

# generate random points on the sphere by uniform sampling
def random_points(axis_num = axis_num):
    points = np.random.rand(axis_num,2)
    points[:,0] = np.arccos(2*points[:,0]-1)
    points[:,1] *= 2*np.pi
    return points

####################

rnd_point_sets = [ random_points() for _ in range(samples) ]
rnd_points = min(rnd_point_sets, key = sqr_error_norm)
rnd_norms = meas_sqr_norms(rnd_points)
print(sqr_error_norm(rnd_points))
print(np.sort(rnd_norms[rnd_norms < 1]))

min_optimum = scipy.optimize.minimize(sqr_error_norm, random_points().ravel())
min_points = min_optimum.x.reshape(rnd_points.shape)
min_norms = meas_sqr_norms(min_points)
print()
print(sqr_error_norm(min_points))
print(np.sort(min_norms[min_norms < 1]))

####################

def distance(point_fst, point_snd):
    axis_fst = to_axis(point_fst)
    axis_snd = to_axis(point_snd)
    angle = np.arccos(np.dot(axis_fst,axis_snd)) / np.pi
    return min(angle,1-angle)

def rotate_point(point, rot_axis, rot_angle):
    cos_angle = np.cos(rot_angle)
    sin_angle = np.sin(rot_angle)
    axis = np.array(to_axis(point))
    new_axis = axis * cos_angle \
             + np.cross(rot_axis,axis) * sin_angle \
             + np.array(rot_axis) * np.dot(rot_axis,axis) * (1-cos_angle)
    return to_angles(new_axis)

def organize_points(points, operation = None, track = False):
    def array_map(function,points):
        return np.array(list(map(function,points)))

    if operation is not None:
        new_points = array_map(operation,points)

    else:
        # anchor the point farthest from any other point to zhat
        anchor_idx = np.argmax( min( distance(point,other)
                                     for other in points if np.all(other != point) )
                                for point in points )
        anchor_axis = to_axis(points[anchor_idx])
        if np.allclose(anchor_axis, zhat):
            def anchor(point): return point
        else:
            rot_axis = np.cross(anchor_axis,zhat)
            rot_axis /= scipy.linalg.norm(rot_axis)
            rot_angle = np.arccos( np.dot(anchor_axis,zhat) )
            anchor = functools.partial(rotate_point,
                                       rot_axis = rot_axis, rot_angle = rot_angle)
        new_points = array_map(anchor,points)

        # orient the post closest to the equator along xhat
        orient_idx = np.argmin(abs(new_points[:,0]-np.pi/2))
        orient_axis = to_axis(new_points[orient_idx])
        rot_angle = -np.arctan2( np.dot(orient_axis,yhat),
                                    np.dot(orient_axis,xhat) )
        orient = functools.partial(rotate_point,
                                   rot_axis = zhat, rot_angle = rot_angle)
        new_points = array_map(orient,new_points)

        # place all points in upper hemisphere
        def northernize(point):
            theta, phi = point
            if theta <= np.pi/2:
                return theta, phi
            else:
                return np.pi-theta, np.pi+phi
        new_points = array_map(northernize,new_points)

        # if the first azimuthal gap is larger than the last,
        #   then change chirality (invert all azimuthal angles)
        new_points[anchor_idx,1] = new_points[orient_idx,1] = 0
        azimuths = new_points[:,1] % (2*np.pi)
        azimuths = azimuths[azimuths != 0]
        if np.min(azimuths) > np.min(2*np.pi-azimuths) \
           or ( azimuths.size == 1 and azimuths[0] > np.pi ):
            def reflect(point): return point[0], -point[1]
        else:
            def reflect(point): return point
        new_points = array_map(reflect,new_points)

    if not track:
        return new_points
    else:
        def identity(x): return x
        def compose2(f,g): return lambda x : f(g(x))
        operations = (reflect, northernize, orient, anchor)
        operation = functools.reduce(compose2, operations, identity)
        return new_points, operation

min_points, operation = organize_points(min_points, track = True)
min_polar = np.vstack([min_points[:,1], abs(np.sin(min_points[:,0]))])
min_plot = plt.polar(min_polar[0,:], min_polar[1,:], "o")

min_plot[0].set_clip_on(False)
plt.ylim(0,1)
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])

plt.tight_layout()
plt.show()
