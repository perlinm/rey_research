#!/usr/bin/env python3

import itertools, sympy

def is_numeric(val): return isinstance(val, (int, float))

# split partition into subsystems
def primary_subsystems(partition):
    subsystems = []
    start_idx = 0
    while start_idx < len(partition):
        if partition[start_idx] == " ":
            start_idx += 1
            continue
        end_idx = start_idx + 1
        while end_idx < len(partition) and not partition[end_idx].isalpha():
            end_idx += 1
        subsystems.append(partition[start_idx:end_idx])
        start_idx = end_idx
    return subsystems

# return power set of a list of objects
def power_set(collection):
    if type(collection) == str:
        collection = primary_subsystems(collection)
    iterable = itertools.chain.from_iterable(itertools.combinations(collection, nn)
                                             for nn in range(len(collection)+1))
    return list( list(part) for part in iterable )

# entropy vector object class
class e_vec:
    # initialize an entropy vector
    def __init__(self, *args):
        if len(args) == 1: # args is a list of pairs: (partition, coefficient)
            self.parts, self.vals = zip(*args[0])
        elif len(args) == 2: # args is two ordered lists: partitions, coefficients
            self.parts, self.vals = args[0], args[1]
        elif len(args) == 3:
            # args[0] is an ordered list of subsystems
            # args[1] is an ordered list of partitions (by index of subsystems)
            # args[2] is an ordered list of coefficients
            self.subs, self.parts, self.vals = args
        elif len(args) == 0: # empty vector
            self.subs, self.parts, self.vals = [], [], []
        else:
            print("invalid initialization of an entropy vector!")
            exit()

        if len(self.parts) == 0:
            self.subs = []
            return

        # if necessary, properly identify subsystems and partitions (by index of subsystem)
        if len(args) in [ 1, 2 ]:
            if type(self.parts) is str:
                self.parts = [ primary_subsystems(part) for part in self.parts ]
            else:
                self.parts = list(self.parts)
            self.subs = sorted(list(set.union(*[ set(part) for part in self.parts ])))

        self.sort()
        self.simplify()

    # sort vector partitions and coefficients
    def sort(self):
        # sort the subsystems within each partition
        self.parts = [ sorted(part, key = lambda sys : self.subs.index(sys))
                       for part in self.parts ]
        # sort the list of partitions and corresponding values
        def partition_sort_key(part):
            return tuple([len(part)] + [self.subs.index(sys) for sys in part])
        zipped_vec = sorted(zip(self.parts, self.vals),
                            key = lambda x : partition_sort_key(x[0]))
        self.parts, self.vals = map(list,zip(*zipped_vec))

    # simplify entropy vector
    def simplify(self):
        # combine identical partitions
        text_parts = [ "".join(part) for part in self.parts ]
        for part in set(text_parts):
            indices = [ jj for jj in range(len(text_parts)) if text_parts[jj] == part ]
            if len(indices) > 1:
                self.vals[indices[0]] = sum([ self.vals[jj] for jj in indices ])
                for jj in indices[1:][::-1]:
                    del self.vals[jj]
                    del self.parts[jj]
                    del text_parts[jj]

        # remove zeros
        zero_indices = [ jj for jj in range(len(self.vals)) if self.vals[jj] == 0 ]
        for idx in zero_indices[::-1]:
            del self.parts[idx]
            del self.vals[idx]

    # return new entropy vector with relabeled subsystems
    def relabeled(self, *args, split = True):
        args = list(args)
        if len(args) == 1:
            # args is a single list of new subsystem names
            old_subsystems, new_subsystems = self.subs, args[0]
        elif len(args) == 2 and type(args[0]) == list:
            # args is two lists: old subsystems and new ones
            old_subsystems, new_subsystems = args
        else:
            # args is two strings: an old subsystem and its new name
            old_subsystems, new_subsystems = [args[0]], [args[1]]

        def relabel(system):
            if system in old_subsystems:
                return new_subsystems[old_subsystems.index(system)]
            else: return system

        subsystems = [ relabel(sys) for sys in self.subs ]
        partitions = [ [ relabel(sys) for sys in part ]
                       for part in self.parts ]
        new_vec = e_vec(subsystems, partitions, self.vals)
        if split: return new_vec.split()
        else: return new_vec

    # return new entropy vector with split subsystems
    def split(self):
        return e_vec([ "".join(part) for part in self.parts ], self.vals)

    # returns a full vector in an ordered basis
    def standard_form(self, system = None):
        if system == None: subs = self.subs
        else: subs = system
        sub_power_set = list(power_set(subs))[1:]
        if any(part not in sub_power_set for part in self.parts):
            return self.split().standard_form(system)
        vec = [ 0 ] * len(sub_power_set)
        for part, val in zip(self.parts, self.vals):
            vec[sub_power_set.index(part)] = val
        return vec

    # return "purification dual" with respect to a given subsystem
    def purification_dual(self, subsystem):
        subs_set = set(range(len(self.subs)))
        idx = self.subs.index(subsystem)
        def trim(part): return set(part).difference(set([idx]))
        def dual_part(part): return sorted(list(subs_set.difference(trim(part))))
        new_parts = [ part if idx not in part else dual_part(part)
                      for part in self.parts ]
        new_vec = e_vec(self.subs, new_parts, self.vals)
        return new_vec

    # evaluate symbolic entries according to dictionary
    def evaluated(self, dict):
        vals = [ val if is_numeric(val) else val.subs(dict) for val in self.vals ]
        return e_vec(self.subs, self.parts, vals)

    # reduced "reduced" vector obtained by making some subsystems trivial
    def reduce(self, trivial_subsystems):
        if type(trivial_subsystems) == str:
            trivial_subsystems = primary_subsystems(trivial_subsystems)
        subs = [ sys for sys in self.subs if sys not in trivial_subsystems ]
        parts = [ [ sys for sys in part if sys not in trivial_subsystems ]
                  for part in self.parts ]
        if all( len(part) == 0 for part in parts ): return e_vec()
        parts, vals = zip(*[ (part, val) for part, val in zip(parts, self.vals)
                             if len(part) > 0])
        return e_vec(subs, parts, vals)

    # return human-readable text identifying this entropy vector
    def __repr__(self):
        signs = [ "+" if val > 0 else "-" for val in self.vals ]
        abs_vals = [ "" if val in [1,-1]
                     else ( f" {val}" if val > 0 else f" {-val}" )
                     for val in self.vals]
        text_parts = [ "".join(part) for part in self.parts ]
        return " ".join([ "{}{} ({})".format(sign,abs_val,part)
                          for sign,abs_val,part in zip(signs,abs_vals,text_parts) ])

    def __hash__(self):
        return hash(self.split().__repr__())

    # relations and operations
    def __eq__(self, other): # comparison x == y
        return self.__hash__() == other.__hash__()
    def __pos__(self): return self # unary "+"
    def __neg__(self): # unary "-", i.e. negation
        return e_vec(self.subs, self.parts, [ -val for val in self.vals ])
    def __add__(self, other): # addition of vectors: x + y
        left, right = self.split(), other.split()
        subs = sorted(list(set(left.subs + right.subs)))
        parts = left.parts + right.parts
        vals = left.vals + right.vals
        new_vec = e_vec(subs, parts, vals)
        new_vec.sort()
        new_vec.simplify()
        return new_vec
    def __sub__(self, other): return self + (-other) # subtraction of vectors: x - y

# return iterator over all partitions of a collection of items
def get_all_partitions(collection):
    if len(collection) == 1:
        yield [ collection ]
        return
    first = collection[0]
    for smaller in get_all_partitions(collection[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        yield [ [ first ] ] + smaller

# get all vectors acquired by subsystem permutation, purification duals
def get_seeded_vecs(seed_vec):
    # get all unique permutations of the input vector
    permutation_vecs = set( seed_vec.relabeled(permutation).split()
                            for permutation in itertools.permutations(seed_vec.subs) )

    # get all purification duals of all permutation vectors
    purification_vecs = set( vec.purification_dual(sub)
                             for vec in permutation_vecs
                             for sub in vec.subs )

    # collect all vectors obtained by additionally making subsystems trivial
    seeded_vecs = set( vec.reduce(subset)
                       for vec in set.union(permutation_vecs, purification_vecs)
                       for subset in power_set(vec.subs) )
    return seeded_vecs

# given a seed vector, compute all corresponding "mirror vectors",
# then collect all vectors of the same form as the mirror vectors for a larger system
def get_super_vectors(subsystem_text, seed_vector):
    subsystems = primary_subsystems(subsystem_text)
    seeded_vectors = get_seeded_vecs(seed_vector)

    full_vecs = set()
    # each seeded vector addresses a k-partite system, while our system is n-partite,
    # so for each size p from k to n
    for size in range(len(seed_vector.subs), len(subsystems)+1):
        # consider all p-partite combinations of the n subsystems
        for combination in itertools.combinations(subsystems, size):
            # for each of these p-partite combinations, consider all k-partitions
            for partition in get_all_partitions(list(combination)):
                # generate all mirror vectors on this k-partition
                new_subsystems = [ "".join(sorted(parts)) for parts in partition ]
                full_vecs.update([ vec.relabeled(new_subsystems)
                                   for vec in seeded_vectors
                                   if len(new_subsystems) == len(vec.subs) ])

    # collect all vectors obtained by making subsystems trivial
    system_vecs = set( vec.reduce(subset)
                       for vec in full_vecs
                       for subset in power_set(vec.subs) )
    return set([ vec for vec in system_vecs if vec != e_vec() ])

# subadditivity, strong subadditivity, and negativity of tripartite information
SA = e_vec([ ("A",1), ("B",1), ("AB",-1) ])
SSA = e_vec([ ("AB",1), ("BC",1), ("B",-1), ("ABC",-1) ])
MMI = e_vec([ ("AB",1), ("AC",1), ("BC",1), ("A",-1), ("B",-1), ("C",-1), ("ABC",-1) ])

systems = "AB"
partitions = [ "".join(subsystem)
               for subsystem in list(power_set(systems))[1:] ]
symbols = [ sympy.symbols(part.lower(), positive = True)
            for part in partitions ]

vecs_SA = get_super_vectors(systems, SA)
vecs_SSA = get_super_vectors(systems, SSA)
vecs_MMI = get_super_vectors(systems, MMI)

# declare a dual vector describing a quantity that is monotone under sub-system processing
processing_monotone = e_vec(partitions, symbols)

# identify all positive vectors derived from the processing monotone
positive_vecs = set( processing_monotone.relabeled(sub,sub+"Q").split()
                     - processing_monotone
                     for sub in primary_subsystems(systems) )

# identify cones restricting the positive vectors
positive_cones = {}
for vec in positive_vecs:
    positive_cones[vec] = []
    restricted_vals = set([ val if val > 0 else -val for val in vec.vals ])
    for val in restricted_vals:
        key = { symbol : 0 for symbol in symbols }
        for sign in [ 1, -1 ]:
            key[val] = sign
            positive_cones[vec].append(vec.evaluated(key))

print(processing_monotone)
print("------------------------------")
print()
for vec in positive_vecs:
    print(vec)
    for ray in positive_cones[vec]:
        print(" ",ray)
    print()
