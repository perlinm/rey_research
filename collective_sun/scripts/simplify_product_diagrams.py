#!/usr/bin/env python3

import numpy as np
import itertools as it
import copy

from itertools_extension import set_diagrams
from operator_product_methods import diagram_vec

dimensions = [ 2, 2, 2 ]

def num_assignments(diagram):
    factors = set.union(*[ set(subset) for subset in diagram.keys() ])
    dimensions = { factor : sum( points for subset, points in diagram.items()
                                 if factor in subset )
                   for factor in factors }
    num = np.prod([ np.math.factorial(dimension) for dimension in dimensions.values() ])
    den = np.prod([ np.math.factorial(points) for points in diagram.values() ])
    return num / den

def diagram_weight(diagram):
    return sum( num_indices
                for ops, num_indices in diagram.items()
                if len(ops) % 2 == 1 )

# collect diagrams by weight of the multi-body operator they are associated with
weight_vec = {}
for diagram in set_diagrams(dimensions):
    weight = diagram_weight(diagram)
    vec = num_assignments(diagram) * diagram_vec(diagram)
    if weight not in weight_vec:
        weight_vec[weight] = 0
    weight_vec[weight] += vec

for weight, vec in weight_vec.items():
    print(weight)
    print("-"*10)

    def _name(diag):
        diag_name = f"{diagram_weight(diag)}_"
        if (0,1,2) in diag.keys():
            diag_name += str(diag[0,1,2])
        else:
            diag_name += "0"
        double_occupancies = [ str(markers) for region, markers in diag.items()
                               if len(region) == 2 ]
        diag_name += "".join(sorted(double_occupancies)[::-1])

        return diag_name

    reduced_vec = vec.reduce().join_permutations()
    names_diags_coefs = sorted(zip(map(_name, reduced_vec.diags),
                                   reduced_vec.diags,
                                   reduced_vec.coefs))

    for name, diag, coef in names_diags_coefs:
        if coef == int(coef): coef = int(coef)
        print("name:", name)
        print("coefficient:", coef)
        print(diag)
        print()
    print()

