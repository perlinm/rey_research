#!/usr/bin/env python3

import itertools, sympy

def is_numeric(val): return isinstance(val, (int, float))

# split partition into subsystems
def primary_subsystems(input_partition):
    partition = input_partition.replace(" ","")
    subsystems = []
    start_idx = 0
    while start_idx < len(partition):
        end_idx = start_idx + 1
        while end_idx < len(partition) and not partition[end_idx].isalpha():
            end_idx += 1
        subsystems.append(partition[start_idx:end_idx])
        start_idx = end_idx
    return subsystems

# return power set of a list of objects
def power_set(input_list):
    iterable = itertools.chain.from_iterable(itertools.combinations(input_list, nn)
                                             for nn in range(len(input_list)+1))
    return list( list(part) for part in iterable )

# entropy vector object class
class e_vec:
    # initialize an entropy vector
    def __init__(self, *args):
        if len(args) == 1: # args is a list of pairs: (partition, coefficient)
            input_parts, self.vals = zip(*args[0])
        elif len(args) == 2: # args is two ordered lists: partitions, coefficients
            input_parts, self.vals = args[0], args[1]
        elif len(args) == 3:
            # args[0] is an ordered list of subsystems
            # args[1] is an ordered list of partitions (by index of subsystems)
            # args[2] is an ordered list of coefficients
            self.subs, self.parts, self.vals = args
        else:
            print("invalid initialization of an entropy vector!")
            exit()

        # if necessary, properly identify subsystems and partitions (by index of subsystem)
        if len(args) < 3:
            text_parts = [ primary_subsystems(part) for part in input_parts ]
            subs_set = set.union(*[ set(part) for part in text_parts ])

            self.subs = sorted(list(subs_set))
            self.parts = [ sorted([ self.subs.index(sys) for sys in part ])
                           for part in text_parts ]

        self.sort()
        self.simplify()

    # sort vector partitions and coefficients
    def sort(self):
        zipped_vec = sorted(zip(self.parts, self.vals), key = lambda x : (len(x[0]),x[0]))
        self.parts, self.vals = map(list,zip(*zipped_vec))

    # simplify entropy vector
    def simplify(self):
        # combine identical partitions
        for part in self.parts:
            indices = [ jj for jj in range(len(self.parts)) if self.parts[jj] == part ]
            if len(indices) > 1:
                self.vals[indices[0]] = sum([ self.vals[jj] for jj in indices ])
                for jj in indices[1:]:
                    del self.vals[jj]
                    del self.parts[jj]

        # remove zeros
        zero_indices = [ jj for jj in range(len(self.vals)) if self.vals[jj] == 0 ]
        for idx in zero_indices[::-1]:
            del self.parts[idx]
            del self.vals[idx]

    # return new entropy vector with relabeled subsystems
    def relabeled(self, *args):
        if len(args) == 1:
            old_subsystems, new_subsystems = self.subs, args[0]
        else:
            old_subsystems, new_subsystems = args
        subsystems = self.subs.copy()
        for jj in range(len(old_subsystems)):
            old_subsystem, new_subsystem = old_subsystems[jj], new_subsystems[jj]
            if not old_subsystem in self.subs:
                print("subsystem not in this entropy vector: {}".format(old_subsystem))
                exit()
            subsystems[self.subs.index(old_subsystem)] = new_subsystem
        return e_vec(subsystems, self.parts, self.vals)

    # return partitions in text (rather than index) format
    def text_parts(self):
        return [ "".join([ self.subs[jj] for jj in part ]) for part in self.parts ]

    # return new entropy vector with split subsystems
    def split(self):
        return e_vec(self.text_parts(), self.vals)

    # returns a full vector in an ordered basis
    def standard_form(self):
        text_parts = self.text_parts()
        sub_power_set = [ "".join(part) for part in power_set(self.subs)[1:] ]
        vec = [ 0 ] * len(sub_power_set)
        for jj in range(len(self.vals)):
            vec[sub_power_set.index(text_parts[jj])] = self.vals[jj]
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
        new_vec.sort()
        return new_vec

    # evaluate symbolic entries according to dictionary
    def evaluated(self, dict):
        vals = [ val if is_numeric(val) else val.subs(dict) for val in self.vals ]
        return e_vec(self.subs, self.parts, vals)

    # return human-readable text identifying this entropy vector
    def __repr__(self):
        signs = [ "+" if val > 0 else "-" for val in self.vals ]
        abs_vals = [ "" if val in [1,-1]
                     else ( f" {val}" if val > 0 else f" {-val}" )
                     for val in self.vals]
        parts = self.text_parts()
        return " ".join([ "{}{} ({})".format(signs[jj],abs_vals[jj],parts[jj])
                          for jj in range(len(self.vals)) ])

    def __hash__(self):
        return hash(self.split().__repr__())

    # relations and operations
    def __eq__(self, other): # comparison x == y
        return self.split().__repr__() == other.split().__repr__()
    def __pos__(self): return self # unary "+"
    def __neg__(self): # unary "-", i.e. negation
        return e_vec(self.subs, self.parts, [ -val for val in self.vals ])
    def __add__(self, other): # addition of vectors: x + y
        left, right = self.split(), other.split()
        subs = sorted(list(set(left.subs + right.subs)))
        parts = [ sorted([ subs.index(sys) for sys in primary_subsystems(text_part) ])
                  for text_part in left.text_parts() + right.text_parts() ]
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

# get all vectors acquired by subsystem permutation and purification duals
def get_mirror_vecs(seed_vec):
    # get all unique permutations of the input vector
    permutation_vecs = set( seed_vec.relabeled(permutation).split()
                            for permutation in itertools.permutations(seed_vec.subs) )

    # get all purification duals of all permutation vectors
    purification_vecs = set( vec.purification_dual(sub)
                             for vec in permutation_vecs
                             for sub in vec.subs )

    return set.union(permutation_vecs, purification_vecs)

# given a seed vector, compute all corresponding "mirror vectors",
# then collect all vectors of the same form as the mirror vectors for a larger system
def get_super_vectors(subsystem_text, seed_vector):
    subsystems = primary_subsystems(subsystem_text)
    mirror_vectors = get_mirror_vecs(seed_vector)
    supersystem_vecs = []
    for partition in get_all_partitions(subsystems):
        new_subsystems = [ "".join(sorted(parts)) for parts in partition ]
        supersystem_vecs += [ vec.relabeled(new_subsystems) for vec in mirror_vectors
                              if len(partition) == len(vec.subs) ]
    return set(supersystem_vecs)

SA = e_vec([ ("A",1), ("B",1), ("AB",-1) ])
SSA = e_vec([ ("AB",1), ("BC",1), ("B",-1), ("ABC",-1) ])
NTI = e_vec([ ("AB",1), ("AC",1), ("BC",1), ("A",-1), ("B",-1), ("C",-1), ("ABC",-1) ])


systems = "AB"
subsystems = [ "".join(subsystem)
               for subsystem in list(power_set(systems))[1:] ]
symbols = [ sympy.symbols(subsystem.lower(), positive = True)
            for subsystem in subsystems ]

processing_monotone = e_vec(subsystems, symbols)
positive_vecs = set( processing_monotone.relabeled([sub],[sub+" Q"]).split()
                     - processing_monotone
                     for sub in primary_subsystems(systems) )

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
print()
for vec in positive_vecs:
    print(vec)
    print(positive_cones[vec])
    print()


vecs_SA = get_super_vectors(systems, SA)
vecs_SSA = get_super_vectors(systems, SSA)
vecs_NTI = get_super_vectors(systems, NTI)
