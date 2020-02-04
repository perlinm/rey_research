#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

figsize = 2 * (0.7,)

radius = 1
dist = 0.6

font_size = 11

fig_dir = "../figures/diagrams/"

params = { "font.family" : "serif",
           "font.serif" : "Computer Modern",
           "text.usetex" : True,
           "font.size" : font_size }
plt.rcParams.update(params)

##################################################
# define circles

circles = "abc"

angles = [ 0, +2*np.pi/3, -2*np.pi/3 ]
def rotate(angle):
    if type(angle) is int:
        angle = angles[angle]
    if type(angle) is str:
        angle = angles[circles.index(angle)]
    return np.array([ [  np.cos(angle), -np.sin(angle) ],
                      [ +np.sin(angle),  np.cos(angle) ] ])
def vec(circle): return rotate(circle)[:,1]

circle_centers = { circle : vec(circle) * dist
                   for circle in circles }

##################################################
# define centers of regions

region_centers = { "abc" : np.zeros(2) }

# inner / outer intersections along the axis of each circle
int_term = np.sqrt(radius**2 - 3/4 * dist**2)
inner_ints = { circle : vec(circle) * ( +int_term - dist/2 )
               for circle in circles }
outer_ints = { circle : vec(circle) * ( -int_term - dist/2 )
               for circle in circles }

# innermost / outermost symmetric points
inner_mids = { circle : circle_centers[circle] - vec(circle) * radius
               for circle in circles }
outer_mids = { circle : circle_centers[circle] + vec(circle) * radius
               for circle in circles }

# centers of overlapping regions
region_mids = { "abc" : np.zeros(2) }
region_mids["a"] = ( outer_mids["a"] + inner_ints["a"] ) / 2
region_mids["b"] = ( outer_mids["b"] + inner_ints["b"] ) / 2
region_mids["c"] = ( outer_mids["c"] + inner_ints["c"] ) / 2

mid_bc_fst = np.mean([ outer_ints["a"], inner_ints["b"], inner_ints["c"] ], axis = 0)
mid_bc_snd = ( outer_ints["a"] + inner_mids["a"]  ) / 2
region_mids["bc"] = ( mid_bc_fst + mid_bc_snd ) / 2
region_mids["ab"] = rotate("c") @ region_mids["bc"]
region_mids["ac"] = rotate("b") @ region_mids["bc"]

##################################################
# define constructor for diagram

def place_dots(region, num, markers = None, region_mids = region_mids,
               sep = radius/3, markersize = 3):
    if markers is None:
        markers = "ko" * num

    vec_list = []

    if num == 1:
        vec_list = [ np.zeros(2) ]

    elif num == 2:
        sep_vec = np.array([sep,0])
        vec_list = [ -sep_vec/2, +sep_vec/2 ]

    elif num == 3:
        sep_vec = np.array([0,sep])
        vec_list = [ rotate(jj) @ sep_vec / np.sqrt(3) for jj in range(3) ]

    elif num > 3:
        print("capability to place more than two dots not implemented")
        exit()

    point = region_mids[region]
    marker_list = [ markers[2*jj:2*jj+2] for jj in range(num) ]
    for vec, marker in zip(vec_list, marker_list):
        pos = point + vec
        size = markersize + ( 1 if marker[1] == "+" else 0 )
        plt.plot(*pos, marker, markersize = size)

def make_triple_diagram(region_dots, markers = {}, figsize = figsize, pad = radius/20):
    fig = plt.figure(figsize = figsize)
    plt.axes().set_aspect("equal")

    # plot circles
    for center in circle_centers.values():
        circle = patches.Circle(center, radius, color = "k", fill = False)
        plt.gca().add_patch(circle)

    # set axis limits
    ymax = dist + radius
    ymin = -dist/2 - radius
    xmax = dist*np.sqrt(3)/2 + radius
    xmin = -xmax

    y_spread = ymax - ymin
    x_spread = xmax - xmin

    if y_spread > x_spread:
        xmin -= ( y_spread - x_spread ) / 2
        xmax += ( y_spread - x_spread ) / 2
    if x_spread > y_spread:
        ymin -= ( x_spread - y_spread ) / 2
        ymax += ( x_spread - y_spread ) / 2

    plt.ylim(ymin-pad, ymax+pad)
    plt.xlim(xmin-pad, xmax+pad)

    # plot dots
    for region, num in region_dots.items():
        place_dots(region, num, markers.get(region))

    # trim figure and return handle
    plt.axis("off")
    plt.tight_layout(pad = 0)
    return fig

def make_double_diagram(region_dots, markers = {}, size_x = figsize[1], pad = radius/20):
    dist = 0.5
    size_y = (radius+pad) / ( radius+3/4*dist*3/4+pad ) * size_x
    _figsize = ( size_x, size_y )
    fig = plt.figure(figsize = _figsize)
    plt.axes().set_aspect("equal")

    # plot circles
    centers = [ -dist, dist ]
    for center in centers:
        circle = patches.Circle([center,0], radius, color = "k", fill = False)
        plt.gca().add_patch(circle)

    # set axis limits
    xmax = dist + radius
    ymax = radius
    plt.xlim(-xmax-pad, xmax+pad)
    plt.ylim(-ymax-pad, ymax+pad)

    # plot dots
    region_mids = { "a" : ( -radius, 0),
                    "b" : ( +radius, 0),
                    "ab" : ( 0, 0) }
    for region, num in region_dots.items():
        place_dots(region, num, markers.get(region), region_mids)

    # trim figure and return handle
    plt.axis("off")
    plt.tight_layout(pad = 0)
    return fig

def add_triple_labels(labels, pads = [ 0.3, 0.4, 0.4 ]):
    for circle, label, pad in zip(circles, labels, pads):
        pos = vec(circle) * ( dist + radius * ( 1 + pad ) )
        plt.text(*pos, f"${label}$",
                 horizontalalignment = "center", verticalalignment = "center")

def add_double_labels(labels, pads = [ 0.3, 0.3 ]):
    for label, pad, direction in zip(labels, pads, [-1,+1]):
        plt.text(direction * dist, radius + pad, f"${label}$",
                 horizontalalignment = "center", verticalalignment = "center")

##################################################
# make diagrams

### example diagrams
dots = { "a" : 1, "abc" : 1, "ac" : 1, "bc" : 2 }
make_triple_diagram(dots, figsize = (1,1))
add_triple_labels(["w_1","w_2","w_3"], pads = [ 0.25, 0.5, 0.55 ])
plt.tight_layout(pad = 0)
plt.savefig(fig_dir + "example_triple.pdf")

make_triple_diagram(dots, {"a" : "rx"}, figsize = (1,1))
add_triple_labels(["w_1","w_2","w_3"], pads = [ 0.25, 0.5, 0.55 ])
plt.tight_layout(pad = 0)
plt.savefig(fig_dir + "example_triple_x_top.pdf")

make_triple_diagram(dots, {"a" : "r+"}, figsize = (1,1))
add_triple_labels(["w_1","w_2","w_3"], pads = [ 0.25, 0.5, 0.55 ])
plt.tight_layout(pad = 0)
plt.savefig(fig_dir + "example_triple_p_top.pdf")

make_triple_diagram(dots, {"a" : "rx", "ac" : "bx"}, figsize = (1,1))
add_triple_labels(["w_1","w_2","w_3"], pads = [ 0.25, 0.5, 0.55 ])
plt.tight_layout(pad = 0)
plt.savefig(fig_dir + "example_triple_xx_2.pdf")

make_triple_diagram(dots, {"ac" : "rx", "bc" : "korx"}, figsize = (1,1))
add_triple_labels(["w_1","w_2","w_3"], pads = [ 0.25, 0.5, 0.55 ])
plt.tight_layout(pad = 0)
plt.savefig(fig_dir + "example_triple_xx_1.pdf")

make_triple_diagram(dots, {"abc" : "r+", "ac" : "r+"}, figsize = (1,1))
add_triple_labels(["w_1","w_2","w_3"], pads = [ 0.25, 0.5, 0.55 ])
plt.tight_layout(pad = 0)
plt.savefig(fig_dir + "example_triple_pp_1.pdf")
plt.close()

dots = { "a" : 3, "b" : 2, "ab" : 2 }
make_double_diagram(dots, {"ab" : "korx"})
add_double_labels(["w_1","w_2"])
plt.tight_layout(pad = 0.15)
plt.savefig(fig_dir + "example_double.pdf")

make_double_diagram(dots, {"ab" : "korx", "a" : "bxkoko"})
add_double_labels(["w_1","w_2"])
plt.tight_layout(pad = 0.15)
plt.savefig(fig_dir + "example_double_x.pdf")

make_double_diagram(dots, {"ab" : "korx", "a" : "b+koko"})
add_double_labels(["w_1","w_2"])
plt.tight_layout(pad = 0.15)
plt.savefig(fig_dir + "example_double_p.pdf")
plt.close()

### single-body product
for assignment in [ { "a" : 1, "b" : 1, "ab" : 0 },
                    { "ab" : 1 } ]:
    make_double_diagram(assignment)
    add_double_labels(["v","w"])
    plt.tight_layout(pad = 0.05)
    tag = assignment["ab"]
    plt.savefig(fig_dir + f"single_body_{tag}.pdf")

plt.close()

### two-body product
for assignment in [ { "a" : 2, "b" : 2, "ab" : 0 },
                    { "a" : 1, "b" : 1, "ab" : 1 },
                    { "ab" : 2 } ]:
    make_double_diagram(assignment)
    add_double_labels(["v","w"])
    plt.tight_layout(pad = 0.05)
    tag = assignment["ab"]
    plt.savefig(fig_dir + f"two_body_{tag}.pdf")

plt.close()

### twiple two-body product

# 6-point diagram
make_triple_diagram({ "a" : 2, "b" : 2, "c" : 2 })
plt.savefig(fig_dir + "triple_6.pdf")

make_triple_diagram({ "a" : 2, "b" : 2, "c" : 2 }, figsize = (0.9,0.7))
add_triple_labels(["u","v","w"])
plt.tight_layout(pad = 0.05)
plt.savefig(fig_dir + "triple_6_uvw.pdf")

plt.close()

# 4-point diagrams
make_triple_diagram({ "a" : 1, "b" : 1, "c" : 1, "abc" : 1 })
plt.savefig(fig_dir + "triple_4_1.pdf")


make_triple_diagram({ "a" : 2, "b" : 1, "c" : 1, "bc" : 1 })
plt.savefig(fig_dir + "triple_4_0.pdf")

plt.close()

# 2-point diagrams
make_triple_diagram({ "abc" : 2 })
plt.savefig(fig_dir + "triple_2_0.pdf")

make_triple_diagram({ "a" : 1, "bc" : 1, "abc" : 1 })
plt.savefig(fig_dir + "triple_2_1.pdf")

make_triple_diagram({ "a" : 2, "bc" : 2 })
plt.savefig(fig_dir + "triple_2_2.pdf")

make_triple_diagram({ "ab" : 1, "ac" : 1, "b" : 1, "c" : 1 })
plt.savefig(fig_dir + "triple_2_3.pdf")

plt.close()

for labels in [ ["u","v","w"], ["v","w","u"], ["w","u","v"] ]:
    make_triple_diagram({ "ab" : 1, "ac" : 1, "b" : 1, "c" : 1 }, figsize = (0.9,0.7))
    add_triple_labels(labels)
    plt.tight_layout(pad = 0)
    tag = "".join(labels)
    plt.savefig(fig_dir + f"triple_2_3_{tag}.pdf")

plt.close()
