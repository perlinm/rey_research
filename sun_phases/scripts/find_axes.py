#!/usr/bin/env python3

import sys, functools
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import scipy.optimize

from dicke_methods import spin_op_vec_dicke

np.set_printoptions(linewidth = 200)

dim = int(sys.argv[1])
try: seed = int(sys.argv[2])
except: seed = 0

####################

spirality_guess = 1
axis_num = 2*dim-1

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
    return A.conj().flatten() @ B.flatten()

def axis_projectors(point):
    spin_op = np.tensordot(to_axis(point), spin_op_vec, axes = 1)
    vecs = np.linalg.eigh(spin_op)[1].T
    return [ np.outer(vec,vec.conj()) for vec in vecs ]

def all_projectors(points):
    return list(it.chain.from_iterable(( axis_projectors(point) for point in points )))

def proj_span_norms(points):
    projectors = all_projectors(points)
    overlap_mat = np.zeros((len(projectors),)*2, dtype = complex)
    for ii, jj in it.product(range(len(projectors)), repeat = 2):
        overlap_mat[ii,jj] = op_dot(projectors[ii],projectors[jj])
    vals = np.linalg.eigvalsh(overlap_mat)
    return vals[-dim**2:]

def proj_span_dim(points):
    norms = proj_span_norms(points)
    return sum(np.logical_not(np.isclose(norms,0)))

def overlap_cost(points):
    points_mat_shape = (points.size//2,2)
    points_mat = points.reshape(points_mat_shape)
    vals = proj_span_norms(points_mat)
    return sum(1/vals)

####################

def random_points(seed = seed):
    np.random.seed(seed)
    points = np.random.rand(axis_num,2)
    points[:,0] = np.arccos(2*points[:,0]-1)
    points[:,1] *= 2*np.pi
    return points

# taken from alem2012sparse
def generate_spiral_points(spirality):
    h_vals = 1 - 2*np.arange(axis_num)/axis_num
    theta = np.arccos(h_vals)
    phi = np.zeros(h_vals.size)
    for jj in range(1,phi.size):
        phi[jj] = phi[jj-1] + spirality / np.sqrt(axis_num*(1-h_vals[jj]**2))
    return np.array([theta,phi]).T
def spirality_cost(spirality, cost_fun = overlap_cost):
    return cost_fun(generate_spiral_points(spirality))
def spiral_points(spirality_guess = spirality_guess):
    optimum = scipy.optimize.minimize(spirality_cost, spirality_guess)
    return generate_spiral_points(optimum.x)

# taken from https://bduvenhage.me/geometry/2019/07/31/generating-equidistant-vectors.html
def equidistant_points(axis_num = axis_num):
    golden_ratio = ( 1 + np.sqrt(5) ) / 2;
    golden_angle = ( 2 - golden_ratio ) * 2*np.pi
    lat = np.arccos( 1 - 2 * np.arange(1, axis_num+1) / (axis_num+1) )
    lon = golden_angle * np.arange(1, axis_num+1)
    return np.array([lon,lat]).T

####################

spr_points = equidistant_points()
spr_norms = proj_span_norms(spr_points)
print(energy_cost(spr_points))
print(overlap_cost(spr_points))
print(spr_norms[spr_norms < 1])
print()

optimum = scipy.optimize.minimize(overlap_cost, spr_points.flatten())
min_points = optimum.x.reshape(spr_points.shape)
min_norms = proj_span_norms(min_points)
print()
print(energy_cost(min_points))
print(overlap_cost(min_points))
print(min_norms[min_norms < 1])

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
spr_points = organize_points(spr_points, operation)

min_polar = np.vstack([min_points[:,1], abs(np.sin(min_points[:,0]))])
spr_polar = np.vstack([spr_points[:,1], abs(np.sin(spr_points[:,0]))])
for min_polar_point, spr_polar_point in zip(min_polar.T, spr_polar.T):
    plt.polar([min_polar_point[0], spr_polar_point[0]],
              [min_polar_point[1], spr_polar_point[1]],
              color = "gray", linestyle = ":", linewidth = 1)
min_plot = plt.polar(min_polar[0,:], min_polar[1,:], "o")
spr_plot = plt.polar(spr_polar[0,:], spr_polar[1,:], ".")

min_plot[0].set_clip_on(False)
spr_plot[0].set_clip_on(False)
plt.ylim(0,1)
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])

plt.tight_layout()
plt.show()
