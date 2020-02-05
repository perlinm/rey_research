#!/usr/bin/env python3

import numpy as np
import itertools as it
import functools, operator, copy

from itertools_extension import set_diagrams

ranks = [ 2, 2, 2 ]

class diagram_vec:
    def __init__(self, diagrams = None, coefficients = None):
        if diagrams is None:
            self.diags = []
        elif type(diagrams) is list:
            self.diags = diagrams
        elif type(diagrams) is dict:
            self.diags = [ diagrams ]
        else:
            print("diagram cannot be initialized with type:", type(diagrams))

        if coefficients is None:
            self.coefs = np.ones(len(self.diags)).astype(int)
        elif hasattr(coefficients, "__getitem__"):
            assert(len(coefficients) == len(self.diags))
            self.coefs = np.array(coefficients)
        else:
            self.coefs = np.array([ coefficients ])

    def __repr__(self):
        return "\n".join([ f"{coef} : {diag}"
                           for coef, diag in zip(self.coefs, self.diags) ])

    def __str__(self): return self.__repr__()

    def __add__(self, other):
        if other == 0: return self
        new_diags = self.diags
        new_coefs = self.coefs
        for other_diag, other_coef in zip(other.diags, other.coefs):
            try:
                diag_idx = new_diags.index(other_diag)
                new_coefs[diag_idx] += other_coef
            except:
                new_diags += [ other_diag ]
                new_coefs = np.append(new_coefs, other_coef)
        new_diags = [ diag for dd, diag in enumerate(new_diags) if new_coefs[dd] != 0 ]
        return diagram_vec(new_diags, new_coefs[new_coefs != 0])
    def __radd__(self, other):
        return self + other
    def __sub__(self, other):
        if other == 0: return self
        return self + diagram_vec(other.diags, -other.coefs)

    def __rmul__(self, scalar):
        return diagram_vec(self.diags, scalar * self.coefs)
    def __mul__(self, scalar):
        return scalar * self
    def __truediv__(self, scalar):
        return 1/scalar * self

    def __pos__(self): return self
    def __neg__(self): return -1 * self

    def reduce(self):
        return sum( coef * reduce_diagram(diag)
                    for coef, diag in zip(self.coefs, self.diags) )

    def equate_permutations(self):
        new_diags = []
        new_coefs = []
        for diag, coef in zip(self.diags, self.coefs):
            labels = functools.reduce(set.union, ( set(region) for region in diag.keys() ))
            eliminated_diag = False
            for perm in it.permutations(labels):
                perm_diag = permute_diagram(diag, perm)
                for dd, new_diag in enumerate(new_diags):
                    if new_diag == perm_diag:
                        new_coefs[dd] += coef
                        eliminated_diag = True
                        break
                if eliminated_diag: break
            if not eliminated_diag:
                new_diags += [ diag ]
                new_coefs += [ coef ]
        new_diags = [ diag for dd, diag in enumerate(new_diags) if new_coefs[dd] != 0 ]
        new_coefs = [ coef for coef in new_coefs if coef != 0 ]
        return diagram_vec(new_diags, new_coefs)

# permute the labels on a diagram
def permute_diagram(diagram, permutation):
    return { tuple(sorted( permutation[idx] for idx in region )) : markers
             for region, markers in diagram.items() }

# reduce a diagram by eliminating filled dots and crosses
def reduce_diagram(diagram):
    diagram = { region : markers if type(markers) is tuple else (markers,0,0)
                for region, markers in diagram.items()
                if markers != 0 and markers != (0,0,0) }

    def _net_markers(mm):
        return sum([ markers[mm] for region, markers in diagram.items() ])

    # first, cover the simpler case of a diagram without any crosses,
    #        in which we eliminate any filled dots that we find
    if _net_markers(2) == 0:
        # if there are no crosses and no filled dots, then we are done simplifying
        if _net_markers(0) == 0:
            empty_dot_diagram = { region : markers[1]
                                  for region, markers in diagram.items() }
            return diagram_vec(empty_dot_diagram)

        for region, markers in diagram.items():
            if markers[0] == 0: continue
            empty_copy = copy.deepcopy(diagram)
            cross_copy = copy.deepcopy(diagram)
            empty_copy[region] = ( markers[0]-1, markers[1]+1, markers[2] )
            cross_copy[region] = ( markers[0]-1, markers[1], markers[2]+1 )
            return reduce_diagram(empty_copy) - reduce_diagram(cross_copy)

    # otherwise, eliminate any crosses in a diagram
    else:
        for region, markers in diagram.items():
            if markers[2] == 0: continue

            def _take_from_region(other_region):
                other_markers = diagram[other_region]
                mutiplicity = other_markers[0]
                if mutiplicity == 0: return

                new_diagram = copy.deepcopy(diagram)
                new_diagram[region] = markers[:2] + ( markers[2]-1, )
                new_diagram[other_region] = ( other_markers[0]-1, ) + other_markers[-2:]

                joint_region = tuple(sorted( region + other_region ))
                joint_markers = new_diagram.get(joint_region)
                if joint_markers == None: joint_markers = (0,0,0)
                new_diagram[joint_region] = ( joint_markers[0]+1, ) + joint_markers[-2:]

                return mutiplicity * reduce_diagram(new_diagram)

            return sum([ _take_from_region(other_region)
                         for other_region, other_markers in diagram.items()
                         if all( primary_set not in other_region
                                 for primary_set in region )
                         and other_markers[0] > 0 ])

# construct a random symmetric tensor with zeros on all diagonal blocks
def random_tensor(rank):
    tensor = np.zeros((spins,)*rank)
    for comb in it.combinations(range(spins), rank):
        num = np.random.rand()
        for perm in it.permutations(comb):
            tensor[perm] = num
    return tensor

def symmetry(diagram):
    return np.product([ np.math.factorial(val[0] if type(val) is tuple else val)
                        for val in diagram.values() ])

# collect diagrams by weight of the multi-body operator they are associated with
weight_vec = {}
for diagram in set_diagrams(ranks):
    weight = sum( num_indices
                  for ops, num_indices in diagram.items()
                  if len(ops) % 2 == 1 )
    if weight == 0: continue
    if weight not in weight_vec.keys():
        weight_vec[weight] = diagram_vec(diagram) / symmetry(diagram)
    else:
        weight_vec[weight] += diagram_vec(diagram) / symmetry(diagram)

for weight, vec in list(weight_vec.items())[::-1]:
    print(weight)
    print("-"*10)

    reduced_vec = vec.equate_permutations().reduce().equate_permutations()
    diags_coefs = list(zip(reduced_vec.diags, reduced_vec.coefs))

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

    diags_coefs = sorted( diags_coefs, key = lambda d_c : ( _name(d_c[0]) ) )

    for diag, coef in diags_coefs:
        diag_name = _name(diag)
        if coef == int(coef): coef = int(coef)
        print("name:", diag_name)
        print("coefficient:", coef)
        print(diag)
        print()
    print()

