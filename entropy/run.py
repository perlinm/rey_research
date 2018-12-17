#!/usr/bin/env python2

import sys, sympy, sage

from entropy_methods import *

if not len(sys.argv) == 2 or not sys.argv[1].isdigit():
    print("usage: {} party_number".format(sys.argv[0]))
    exit()

system_num = int(sys.argv[1])
if system_num > 10:
    print("no.")
    exit()

# first system_num capital letters
systems = [ chr(jj) for jj in range(ord("A"), ord("Z")) ][:system_num]
systems_ext = systems + [ "Z" ]

# subadditivity, strong subadditivity, and monogamy of mutual information
SA = e_vec([ ("A",1), ("B",1), ("AB",-1) ])
SSA = e_vec([ ("AB",1), ("BC",1), ("B",-1), ("ABC",-1) ])
MMI = e_vec([ ("AB",1), ("AC",1), ("BC",1), ("A",-1), ("B",-1), ("C",-1), ("ABC",-1) ])

rest_vecs = [ SSA ]


##########################################################################################

def cone_contains(outer, inner):
    return all([ outer.contains(ray) for ray in inner.rays() ])

def cones_equal(first, second):
    return ( len(first.rays()) == len(second.rays()) and
             all([ ray in first.rays() for ray in second.rays() ]) )

##########################################################################################

# subsystems of the entire system
subsystems = [ "".join(sys) for sys in list(power_set(systems))[1:] ]
symbols = [ sympy.symbols(subsystem.lower(), positive = True)
            for subsystem in subsystems ]

# declare a dual vector describing a quantity that is monotone under sub-system processing
processing_monotone = e_vec(subsystems, symbols)

# identify all positive vectors derived from the processing monotone
positive_vecs = set( processing_monotone.relabeled(sys,sys+systems_ext[-1])
                     - processing_monotone for sys in systems )

# identify cones restricting the positive vectors
positive_cones = {}
for vec in positive_vecs:
    extreme_rays = []
    restricted_vals = set([ val if val > 0 else -val for val in vec.vals ])
    for val in restricted_vals:
        key = { symbol : 0 for symbol in symbols }
        for sign in [ 1, -1 ]:
            key[val] = sign
            extreme_rays.append(vec.evaluated(key).standard_form(systems_ext))
    positive_cones[vec] = sage.geometry.cone.Cone(extreme_rays)

# identify cone restricting positive dual vectors
rest_rays = [ get_super_vectors(systems_ext, vec) for vec in rest_vecs ]
rest_cones = [ sage.geometry.cone.Cone([ ray.standard_form(systems_ext) for ray in rays ])
               for rays in rest_rays ]
rest_cone = rest_cones[0]
for jj in range(1,len(rest_cones)):
    rest_cone = rest_cone.intersection(rest_cones[jj])

for vec, cone in positive_cones.items():
    print(vec)
    print
    print(cone.intersection(rest_cone).rays())
    print
    print
