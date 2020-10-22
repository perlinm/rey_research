#!/usr/bin/env python3

import sys, functools, random
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
random.seed(seed)
np.random.seed(seed)

####################

samples = 100
axis_num = 2*dim-1
# axis_num = (3*dim)//2 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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
    return abs(sum(1/norms))

def random_points():
    points = np.random.rand(axis_num,2)
    points[:,0] = np.arccos(2*points[:,0]-1)
    points[:,1] *= 2*np.pi
    return points

def unifrm_points():
    points = np.random.rand(axis_num,2)
    points[:,0] *= np.pi
    points[:,1] *= 2*np.pi
    return points

# taken from https://ieeexplore.ieee.org/document/6508014
def spiral_points(spirality = 3.6):
    base_vals = -1 + 2*np.arange(axis_num) / (axis_num-1)
    spiral_polar = np.arccos(base_vals)
    spiral_azimuth = np.zeros(axis_num-1)
    for kk in range(1,spiral_azimuth.size):
        diff = spirality / np.sqrt( axis_num * (1-base_vals[kk]**2) )
        spiral_azimuth[kk] = spiral_azimuth[kk-1] + diff
    grid_points = list(it.product(spiral_polar[1:-1], spiral_azimuth))
    grid_points += [ (0,0) ]
    return np.array(random.sample(grid_points, axis_num))

####################

samples = 10**4
cutoff = 0.95

def cutoff_val(costs):
    return sorted(costs)[int(cutoff*len(costs))]

random_costs = [ overlap_cost(random_points()) for _ in range(samples) ]
unifrm_costs = [ overlap_cost(unifrm_points()) for _ in range(samples) ]
spiral_costs = [ overlap_cost(spiral_points()) for _ in range(samples) ]

min_val = min(random_costs + unifrm_costs + spiral_costs)
max_val = min([ cutoff_val(random_costs),
                cutoff_val(unifrm_costs),
                cutoff_val(spiral_costs) ])

def range_ratio(costs):
    return len([ val for val in costs if val <= max_val ]) / len(costs)

print(range_ratio(random_costs))
print(range_ratio(unifrm_costs))
print(range_ratio(spiral_costs))

bins = np.logspace(int(np.log10(min_val)), int(np.log10(max_val))+1, 100)
plt.hist(random_costs, bins = bins, alpha = 0.5, label = "R")
plt.hist(unifrm_costs, bins = bins, alpha = 0.5, label = "U")
plt.hist(spiral_costs, bins = bins, alpha = 0.5, label = "S")
plt.gca().set_xscale("log")

plt.title(dim)
plt.legend(loc = "best")
plt.tight_layout()
plt.show()


exit()

spr_point_sets = [ spiral_points() for _ in range(samples) ]
spr_points = min(spr_point_sets, key = overlap_cost)
spr_norms = proj_span_norms(spr_points)
print(overlap_cost(spr_points))
print(np.sort(spr_norms[spr_norms < 1]))

min_optimum = scipy.optimize.minimize(overlap_cost, spr_points.ravel())
min_points = min_optimum.x.reshape(spr_points.shape)
min_norms = proj_span_norms(min_points)
print()
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
