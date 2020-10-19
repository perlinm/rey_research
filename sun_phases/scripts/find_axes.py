#!/usr/bin/env python3

import sys, functools
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import scipy.optimize

from dicke_methods import spin_op_vec_dicke
from multilevel_methods import transition_op

np.set_printoptions(linewidth = 200)

dim = int(sys.argv[1])
try: seed = int(sys.argv[2])
except: seed = 0

####################

axis_num = 2*dim-1

drive_ops = [ np.diag(transition_op(dim,L,0)) for L in range(dim) ]

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

def distance(point_fst, point_snd):
    axis_fst = to_axis(point_fst)
    axis_snd = to_axis(point_snd)
    angle = np.arccos(np.dot(axis_fst,axis_snd)) / np.pi
    return min(angle,1-angle)
def distances(points):
    points_mat_shape = (points.size//2,2)
    points_mat = points.reshape(points_mat_shape)
    return ( distance(point_fst,point_snd)
             for point_fst, point_snd in it.combinations(points_mat, 2) )

def energy_cost(points):
    return sum( 1/dist**2 for dist in distances(points) )

####################

spin_op_vec = np.array([ op.todense() for op in spin_op_vec_dicke(dim-1) ])

def op_dot(A,B):
    return A.conj().ravel() @ B.ravel()

def axis_projectors(point):
    spin_op = np.tensordot(to_axis(point), spin_op_vec, axes = 1)
    _, vecs = np.linalg.eigh(spin_op)
    return [ np.outer(vec,vec.conj()) for vec in vecs.T ]

def axis_drive_ops(point):
    projectors = axis_projectors(point)
    return [ sum([ coef * proj for coef, proj in zip(drive,projectors) ])
             for drive in drive_ops ]

def all_drive_ops(points):
    return np.array([ axis_drive_ops(point) for point in points ])

def proj_span_norms(points):
    axis_num = points.shape[0]
    drive_ops = all_drive_ops(points)
    norms = np.zeros(dim**2)
    for LL in range(dim):
        overlap_mat = np.zeros((axis_num,)*2, dtype = complex)
        for ii, jj in it.product(range(axis_num), repeat = 2):
            overlap_mat[ii,jj] = op_dot(drive_ops[ii,LL,:,:],drive_ops[jj,LL,:,:])
        idx_min, norm_num = LL**2, 2*LL+1
        norms_LL = np.linalg.eigvalsh(overlap_mat)[-norm_num:]
        norms[ idx_min : idx_min + norms_LL.size ] = norms_LL
    return norms

def proj_span_dim(points):
    norms = proj_span_norms(points)
    return sum(np.logical_not(np.isclose(norms,0)))

def overlap_cost(points, indices = None):
    points_mat_shape = (points.size//2,2)
    points_mat = points.reshape(points_mat_shape)
    norms = proj_span_norms(points_mat)
    return sum(1/norms)

def random_points(seed = seed):
    np.random.seed(seed)
    points = np.random.rand(axis_num,2)
    points[:,0] = np.arccos(2*points[:,0]-1)
    points[:,1] *= 2*np.pi
    return points

####################

rnd_points = random_points()
rnd_norms = proj_span_norms(rnd_points)
print(energy_cost(rnd_points))
print(overlap_cost(rnd_points))
print(np.sort(rnd_norms[rnd_norms < 1]))
print()

optimum = scipy.optimize.minimize(overlap_cost, rnd_points.ravel())
min_points = optimum.x.reshape(rnd_points.shape)
min_norms = proj_span_norms(min_points)
print()
print(energy_cost(min_points))
print(overlap_cost(min_points))
print(np.sort(min_norms[min_norms < 1]))

####################

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
            rot_axis /= np.linalg.norm(rot_axis)
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
rnd_points = organize_points(rnd_points, operation)

min_polar = np.vstack([min_points[:,1], abs(np.sin(min_points[:,0]))])
rnd_polar = np.vstack([rnd_points[:,1], abs(np.sin(rnd_points[:,0]))])
for min_polar_point, rnd_polar_point in zip(min_polar.T, rnd_polar.T):
    plt.polar([min_polar_point[0], rnd_polar_point[0]],
              [min_polar_point[1], rnd_polar_point[1]],
              color = "gray", linestyle = ":", linewidth = 1)
min_plot = plt.polar(min_polar[0,:], min_polar[1,:], "o")
rnd_plot = plt.polar(rnd_polar[0,:], rnd_polar[1,:], ".")

min_plot[0].set_clip_on(False)
rnd_plot[0].set_clip_on(False)
plt.ylim(0,1)
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])

plt.tight_layout()
plt.show()
