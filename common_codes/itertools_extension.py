#!/usr/bin/env python3

import itertools as it

# standard "itertools" method
# https://docs.python.org/3/library/itertools.html
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return it.chain.from_iterable( it.combinations(s, r) for r in range(len(s)+1) )

# generate all distinct ways to put `num` identical objects into `bins` distinct bins
# method: use a "stars and bars" construction, where the bars define bins for the stars
# returns a generator that loops over all populations of the bins
def assignments(num, bins):
    for bars in it.combinations(range(num+bins-1), bins-1):
        bars = (-1,) + bars + (num+bins-1,)
        yield tuple( bars[bin+1] - bars[bin] - 1 for bin in range(bins) )

# add two dictionaries
def _add_dicts(dict_1, dict_2):
    combined_dict = dict(dict_1)
    for key, val in dict_2.items():
        if key in combined_dict.keys():
            combined_dict[key] += val
        else:
            combined_dict[key] = val
    return combined_dict

# generate all possible set (Venn) diagrams of primary sets with (given) fixed sizes
# primary sets are labeled by integers in `range(len(set_sizes))`,
#   such that `set_sizes[jj]` is the number of elements in primary set `jj`
# returns a generator that loops over all set diagrams, where each set diagram
#   is a dictionary mapping <subset of primary sets> to <number of shared elements>
def set_diagrams(set_sizes, _fst = 0):

    if len(set_sizes) > 1:

        fst_size, other_sizes = set_sizes[0], set_sizes[1:]
        others_powerset = list(powerset(range(1+_fst,1+_fst+len(other_sizes))))

        for fst_assignment in assignments(fst_size, len(others_powerset)):
            remaining_sizes = list(other_sizes)

            for fst_in_bin, others_subset in zip(fst_assignment, others_powerset):
                for other in others_subset:
                    remaining_sizes[other-_fst-1] -= fst_in_bin

            if True in map( lambda ss : ss < 0, remaining_sizes ):
                continue

            fst_assignment = { (_fst,) + others_bin : fst_in_bin
                               for others_bin, fst_in_bin
                               in zip(others_powerset, fst_assignment)
                               if fst_in_bin != 0 }

            for remainder_diagram in set_diagrams(remaining_sizes,_fst+1):
                yield _add_dicts(fst_assignment, remainder_diagram)

    if len(set_sizes) == 1:
        if set_sizes[0] == 0:
            yield {}
        else:
            yield { (_fst,) : set_sizes[0] }

# "accel_asc" algorithm written by Jerome Kelleher:
# http://jeromekelleher.net/generating-integer-partitions.html
def partitions(n):
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]
