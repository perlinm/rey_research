#!/usr/bin/env python2

import sys, sympy, sage

from entropy_methods import *

if not len(sys.argv) == 2 or not sys.argv[1].isdigit():
    print("usage: {} party_number".format(sys.argv[0]))
    exit()

system_num = int(sys.argv[1])
if system_num > 4:
    print("no.")
    exit()

sage_cone = sage.geometry.cone.Cone
def cone_contains(outer, inner):
    return all([ outer.contains(ray) for ray in inner.rays() ])

# identify primary systems by the first system_num capital letters
systems = [ chr(jj) for jj in range(ord("A"), ord("Z")) ][:system_num]
systems_ext = systems + [ "Z" ] # "extended" systems

# subadditivity, strong subadditivity, and monogamy of mutual information
SA = e_vec([ ("A",1), ("B",1), ("AB",-1) ])
SSA = e_vec([ ("AB",1), ("BC",1), ("B",-1), ("ABC",-1) ])
MMI = e_vec([ ("AB",1), ("AC",1), ("BC",1), ("A",-1), ("B",-1), ("C",-1), ("ABC",-1) ])

inequality_vecs = [ SSA ]

# subsystems of the entire system
subsystems = [ "".join(sys) for sys in list(power_set(systems))[1:] ]
symbols = [ sympy.symbols(subsystem.lower(), positive = True)
            for subsystem in subsystems ]

# declare a dual vector describing a quantity that is monotone under sub-system processing
processing_monotone = e_vec(subsystems, symbols)

# identify all positive vectors derived from the processing monotone
positive_vecs = { sys : processing_monotone.relabeled(sys,sys+systems_ext[-1])
                  - processing_monotone for sys in systems }
positive_vecs_full = { sys : vec.standard_form(systems_ext)
                       for sys, vec in positive_vecs.items() }

# find cones containing (i.e. restricting allowable) positive dual vectors
# defined by monotinicity under sub-system processing
print("constructing positivity cones...")
positive_cones = {}
for sys, vec in positive_vecs.items():
    extreme_rays = []
    restricted_vals = set([ val if val > 0 else -val for val in vec.vals ])
    for val in restricted_vals:
        key = { symbol : 0 for symbol in symbols }
        for sign in [ 1, -1 ]:
            key[val] = sign
            extreme_rays.append(vec.evaluated(key).standard_form(systems_ext))
    positive_cones[sys] = sage_cone(extreme_rays)

# find cone containing (i.e. restricting allowable) positive dual vectors
# defined by choices of entropic inequalities that states must satisfy
print("constructing inequality restriction cone...")
inequality_rays = [ get_positive_vectors(systems_ext, vec) for vec in inequality_vecs ]
inequality_cones = [ sage_cone([ ray.standard_form(systems_ext) for ray in rays ])
                     for rays in inequality_rays ]
inequality_cone = reduce(lambda x, y : x.intersection(y), inequality_cones)

# intersect cones restricting positive dual vectors
print("intersecting positivity and inequality restriction cones...")
for sys in positive_cones.keys():
    positive_cones[sys] = positive_cones[sys].intersection(inequality_cone)

# pull back cone restricting positive dual vectors
print("constructing pullback cones...")
pullback_cone = {}
for sys, cone in positive_cones.items():
    rays = [ list(ray) for ray in cone.rays() ]
    pullback_rays = []
    for ray in rays:
        pullback_ray = [ 0 ] * len(symbols)
        for idx, symbol in enumerate(symbols):
            if symbol not in positive_vecs_full[sys]: continue
            pullback_ray[idx] = ray[positive_vecs_full[sys].index(symbol)]
        pullback_rays.append(pullback_ray)
    for idx, symbol in enumerate(symbols):
        if symbol in positive_vecs_full[sys]: continue
        pullback_rays.append([ +1 if nn == idx else 0 for nn in range(len(symbols)) ])
        pullback_rays.append([ -1 if nn == idx else 0 for nn in range(len(symbols)) ])
    pullback_cone[sys] = sage_cone(pullback_rays)

print("intersecting pullback cones...")
final_cone = reduce(lambda x, y : x.intersection(y), pullback_cone.values())

for ray in final_cone.rays():
    print(e_vec(ray,systems))
