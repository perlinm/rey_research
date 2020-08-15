#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools as it

fig_dir = "./figures/"

params = { "font.family" : "sans-serif",
           "font.serif" : "Computer Modern",
           "text.usetex" : True,
           "font.size" : 10 }
plt.rcParams.update(params)

fig_dir = "../figures/"

######################################################################
# plot sketch of SWAP operation

radius = 0.4
height = 0.5

color_cycle = [ _["color"] for _ in plt.rcParams["axes.prop_cycle"] ]

def place_spins(centers, angles, colors, labels = None, height = height):
    assert(len(centers) == len(angles) == len(colors))

    x_vals, y_vals = list(zip(*centers))
    axis_pad = 2.25 * radius
    def axis_width(vals):
        return max(vals) - min(vals) + 2*axis_pad

    width = axis_width(x_vals) / axis_width(y_vals) * height
    figsize = (width,height)
    figure, axis = plt.subplots(figsize = figsize)

    if labels is None: labels = [ " " ] * len(centers)
    for center, angle, color, label in zip(centers, angles, colors, labels):
        center = np.array(center)

        circle = plt.Circle(center, radius = radius, facecolor = color)
        axis.add_patch(circle)

        arrow_tip = 4 * radius * np.array([ np.sin(angle), np.cos(angle) ])
        arrow_base = np.array(center) - arrow_tip*43/100
        arrow = plt.Arrow(*arrow_base, *arrow_tip, width = 0.3, color = color)
        axis.add_patch(arrow)

        axis.text(*center, f"${label}$",
                  horizontalalignment = "center",
                  verticalalignment = "center")

    axis.set_xlim(min(x_vals) - axis_pad, max(x_vals) + axis_pad)
    axis.set_ylim(min(y_vals) - axis_pad, max(y_vals) + axis_pad)
    axis.set_aspect("equal")
    axis.set_axis_off()

    figure.tight_layout(pad = 0)
    return figure, axis

def make_wiggles(centers, points = 100, waves = 4, axis = None):
    if axis is None: axis = plt.gca()
    for center_lft, center_rht in it.combinations(centers,2):
        x_lft, x_rht = center_lft[0], center_rht[0]
        y_lft, y_rht = center_lft[1], center_rht[1]
        angle = np.arctan2(y_rht - y_lft, x_rht - x_lft)

        t_vals = np.linspace(0, 1, points)
        h_vals = radius/5 * np.sin(t_vals * 2*np.pi * waves)

        x_vals = np.linspace(x_lft, x_rht, points) - np.sin(angle) * h_vals
        y_vals = np.linspace(y_lft, y_rht, points) + np.cos(angle) * h_vals
        axis.plot(x_vals, y_vals, "r", zorder = -1)

### spin pair

centers = [ (-1,0), (+1,0) ]
angles = [ -np.pi/3, np.pi/6 ]
colors = color_cycle[:2]
labels = [ "i", "j" ]

place_spins(centers, angles, colors, labels)
make_wiggles(centers)
plt.savefig(fig_dir + "sketch_spins.pdf")

place_spins(centers, angles[::-1], colors[::-1], labels)
make_wiggles(centers)
plt.savefig(fig_dir + "sketch_spins_swap.pdf")

### spin triplet

centers = [ (0,0), (+1,0), (+2,0) ]
angles = [ np.pi/6 ] * 3
colors = [ color_cycle[0] ] * 3
place_spins(centers, angles, colors = colors)
plt.savefig(fig_dir + "sketch_uniform.pdf")

centers = [ (0,0), (+1,0), (+2,0) ]
angles = [ np.pi/6, 0, np.pi/3 ]
colors = color_cycle[:3]
place_spins(centers, angles, colors = colors)
plt.savefig(fig_dir + "sketch_different.pdf")

plt.show()
