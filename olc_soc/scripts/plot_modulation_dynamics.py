#!/usr/bin/env python3

import glob, sys, os
from pylab import *

if len(sys.argv) not in [ 3, 4, 5 ]:
    print("usage: {} mod_type depth [view] [save]".format(sys.argv[0]))
    exit()

mod_type = sys.argv[1]
depth = sys.argv[2]

save = "save" in sys.argv
view = "view" in sys.argv

data_dir = "../data/"
fig_dir = "../figures/"
if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

# some plot options
fig_dpi = 600
fig_size = (4,3)

params = { "text.usetex" : True,
           "text.latex.preamble" : [ r"\usepackage{physics}",
                                     r"\usepackage{dsfont}" ] }
rcParams.update(params)

# import data from files
band_file = "{}_mod_band_V{}.txt".format(mod_type,depth)
spin_file = "{}_mod_spin_V{}.txt".format(mod_type,depth)
assert(os.path.isfile(data_dir+band_file))
assert(os.path.isfile(data_dir+spin_file))

band_data = loadtxt(data_dir+band_file, ndmin = 2)
spin_data = loadtxt(data_dir+spin_file, ndmin = 2)

# read in metadata
with open(data_dir+band_file) as f:
    for line in f:
        if "time" not in line: continue
        else:
            if "maximum" in line:
                max_time = double(line.split()[-1])
            if "label" in line:
                time_text = line.split()[-1]
                break

times = linspace(0, max_time, len(band_data[0,1:]))

def borders(array):
    borders = zeros(len(array)+1)
    borders[1:-1] = (array[1:] + array[:-1]) / 2
    borders[0] = array[0] - (array[1] - array[0]) / 2
    borders[-1] = array[-1] + (array[-1] - array[-2]) / 2
    return borders

time_borders = borders(times)
q_borders = borders(band_data[:,0])

figure(figsize = fig_size)
gca().set_rasterization_zorder(1)

pcolormesh(time_borders, q_borders, band_data[:,1:], zorder = 0)
cb = colorbar()
clim(0,1)
cb.formatter.set_powerlimits((-2, 3))
cb.update_ticks()

title(r"$\left<\op{q}\otimes\op{1}\otimes\mathds{1}\right>$", zorder = 1)
xlabel(time_text, zorder = 1)
ylabel("$q/k_L$", zorder = 1)
tight_layout()
if save:
    savefig(fig_dir+band_file.replace(".txt",".pdf"), rasterized = True, dpi = fig_dpi)

figure(figsize=fig_size)
gca().set_rasterization_zorder(1)

pcolormesh(time_borders, q_borders, spin_data[:,1:], zorder = 0)
cb = colorbar()
clim(0,1)
cb.formatter.set_powerlimits((-2, 3))
cb.update_ticks()

title(r"$\left<\op{q}\otimes\mathds{1}\otimes\op{1}\right>$", zorder = 1)
xlabel(time_text, zorder = 1)
ylabel("$q/k_L$", zorder = 1)
tight_layout()
if save:
    savefig(fig_dir+spin_file.replace(".txt",".pdf"), rasterized = True, dpi = fig_dpi)

if view: show()
