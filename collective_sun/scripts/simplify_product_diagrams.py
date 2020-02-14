#!/usr/bin/env python3

import numpy as np
import itertools as it
import copy

from itertools_extension import set_diagrams
from operator_product_methods import diagram_vec

dimensions = [ 2, 2, 2 ]

# construct a random symmetric tensor with zeros on all diagonal blocks
def random_tensor(dimension):
    tensor = np.zeros((spins,)*dimension)
    for comb in it.combinations(range(spins), dimension):
        num = np.random.rand()
        for perm in it.permutations(comb):
            tensor[perm] = num
    return tensor

# collect diagrams by weight of the multi-body operator they are associated with
weight_vec = {}
for diagram in set_diagrams(dimensions):
    weight = sum( num_indices
                  for ops, num_indices in diagram.items()
                  if len(ops) % 2 == 1 )
    if weight not in weight_vec.keys():
        weight_vec[weight] = diagram_vec(diagram)
    else:
        weight_vec[weight] += diagram_vec(diagram)

for weight, vec in weight_vec.items():
    print(weight)
    print("-"*10)

    def _name(diag):
        diag_name = ""
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
                                   reduced_vec.coefs))[::-1]

    for name, diag, coef in names_diags_coefs:
        if coef == int(coef): coef = int(coef)
        print("name:", name)
        print("coefficient:", coef)
        print(diag)
        print()
    print()

