#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
centers = [ (-1,0), (+1,0) ]
angles = [ -np.pi/3, np.pi/6 ]
colors = [ "#1f77b4", "#ff7f0e" ]
labels = [ "i", "j" ]

def make_spin_sketch(centers, angles, colors, labels):
    plt.figure(figsize = (1.5,1))

    points, waves = 100, 4
    x_lft, x_rht = centers[0][0], centers[1][0]
    disp = x_rht - x_lft
    x_vals = np.linspace(x_lft, x_rht, points)
    y_vals = radius/5 * np.sin(x_vals/disp * 2*np.pi * waves)
    plt.plot(x_vals, y_vals, "r", zorder = -1)

    for center, angle, color, label in zip(centers, angles, colors, labels):
        center = np.array(center)

        circle = plt.Circle(center, radius = radius, facecolor = color)
        plt.gca().add_patch(circle)

        arrow_tip = 4 * radius * np.array([ np.sin(angle), np.cos(angle) ])
        arrow_base = np.array(center) - arrow_tip*43/100
        arrow = plt.Arrow(*arrow_base, *arrow_tip, width = 0.3, color = color)
        plt.gca().add_patch(arrow)

        plt.text(*center, f"${label}$",
                 horizontalalignment = "center",
                 verticalalignment = "center")

    plt.gca().set_xlim(-5*radius,5*radius)
    plt.gca().set_ylim(-3*radius,3*radius)
    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.tight_layout(pad = 0)

make_spin_sketch(centers, angles, colors, labels)
plt.savefig(fig_dir + "sketch_spins.pdf")

make_spin_sketch(centers, angles[::-1], colors[::-1], labels)
plt.savefig(fig_dir + "sketch_spins_swap.pdf")

plt.show()
