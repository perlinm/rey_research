#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import cmocean

params = { "font.size" : 10,
           "text.usetex" : True }
plt.rcParams.update(params)
fig_dir = "../figures/"

max_val = 5
points = 400
sx, sy = 1, 2

figsize = (1,1)
levels = 10
cmap = cmocean.cm.amp

xs = np.linspace(-max_val,max_val,points)
ys = np.linspace(-max_val,max_val,points)
XS, YS = np.meshgrid(xs, ys)
ZS = np.exp(-((XS/sx)**2 + (YS/sy)**2)/2)

plt.figure(figsize = figsize)
plt.contourf(XS, YS, ZS, levels = levels, cmap = cmap)
plt.xticks([])
plt.yticks([])
plt.xlabel("$X$")
plt.ylabel("$P$")

plt.arrow(+max_val*.8, 0, -max_val/4, 0, color = "c", head_width = .2)
plt.arrow(-max_val*.8, 0, +max_val/4, 0, color = "c", head_width = .2)
plt.arrow(0, +max_val*.45, 0, +max_val/4, color = "c", head_width = .2)
plt.arrow(0, -max_val*.45, 0, -max_val/4, color = "c", head_width = .2)

plt.tight_layout(pad = 0.1)
plt.savefig(fig_dir + "squeezed_gaussian.pdf")
