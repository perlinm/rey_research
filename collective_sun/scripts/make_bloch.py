#!/usr/bin/env python3

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

xhat = np.array([1.,0.,0.])
yhat = np.array([0.,1.,0.])
zhat = np.array([0.,0.,1.])

width = 2

def vec_xy(angle = 0):
    return np.cos(angle) * yhat - np.sin(angle) * xhat

def plot_sphere(vectors, points = None, color = "#d62728"):
    sphere = qt.Bloch()
    sphere.add_vectors(vectors)
    if points is not None:
        zipped_points = [ np.array(pts) for pts in zip(*points) ]
        sphere.add_points(zipped_points)

    sphere.frame_alpha = 0
    sphere.xlabel, sphere.ylabel, sphere.zlabel = [["",""]]*3
    sphere.figsize = (width,width)

    sphere.vector_color = [ color ]
    sphere.point_color = [ color ]

    sphere.render()
    return sphere

sphere = plot_sphere(vec_xy(0))
plt.savefig("../figures/bloch_x.pdf")

angle = 2*np.pi * 2/3
points = 10
points = [ vec_xy(part) for part in np.linspace(0,angle,points) ]
sphere = plot_sphere(vec_xy(angle), points)
plt.savefig("../figures/bloch_xy.pdf")
