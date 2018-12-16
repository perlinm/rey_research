#!/usr/bin/env python3

import itertools

# split partition into subsystems
def get_subsystems(input_partition):
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
            # args[1] is an ordered list of partitions (defined by indices)
            # args[2] is an ordered list of coefficients
            self.subs, self.parts, self.vals = args
        else:
            print("invalid initialization of an entropy vector!")
            exit()

        self.vals = list(self.vals)
        if len(args) == 3: return None

        # remove zeros
        zero_indices = [ jj for jj in range(len(self.vals)) if self.vals[jj] == 0 ]
        for idx in zero_indices[::-1]:
            del input_parts[idx]
            del self.vals[idx]

        # identify subsystems and partitions
        all_subs = set()
        parts_text = []
        for jj in range(len(input_parts)):
            parts_text.append([])
            for subsystem in get_subsystems(input_parts[jj]):
                all_subs.add(subsystem)
                parts_text[jj].append(subsystem)
        self.subs = sorted(list(all_subs))
        self.parts = [ sorted([ self.subs.index(sys) for sys in part ])
                       for part in parts_text ]

        # sort partitions and values
        self.parts, self.vals = zip(*sorted(zip(self.parts, self.vals),
                                            key = lambda x : (len(x[0]),x[0])))

        # if any partitions are identical, combine them
        for part in self.parts:
            indices = [ jj for jj in range(len(self.parts)) if self.parts[jj] == part ]
            if len(indices) > 1:
                self.vals[indices[0]] = sum([ self.vals[jj] for jj in indices ])
                for jj in indices[1:]:
                    del self.vals[jj]
                    del self.parts[jj]

    # return new entropy vector with relabeled subsystems
    def relabeled(self, old_subsystems, new_subsystems):
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
    def full_vec(self):
        text_parts = self.text_parts()
        sub_power_set = [ "".join(part) for part in power_set(self.subs)[1:] ]
        vec = [ 0 ] * len(sub_power_set)
        for jj in range(len(self.vals)):
            vec[sub_power_set.index(text_parts[jj])] = self.vals[jj]
        return vec

    # return human-readable text identifying this entropy vector
    def __repr__(self):
        signs = [ "+" if val > 0 else "-" for val in self.vals ]
        abs_vals = [ "" if abs(val) == 1 else f" {str(abs(val))}"
                      for val in self.vals ]
        parts = self.text_parts()
        return " ".join([ "{}{} {}".format(signs[jj],abs_vals[jj],parts[jj])
                          for jj in range(len(self.vals)) ])

    # compare two entropy vectors
    def __eq__(self, other):
        return self.split().__repr__() == other.split().__repr__()

# return iterator over all partitions of a collection of items
def get_partitions(collection):
    if len(collection) == 1:
        yield [ collection ]
        return

    first = collection[0]
    for smaller in get_partitions(collection[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        yield [ [ first ] ] + smaller

# def get_partitions(collection, parts):
#     if parts == 1: return [ collection ]
#     else:
#         return [ [ collection[:jj] ] + get_partitions(collection[jj:], parts-1)
#                  for jj in range(1,len(collection)) ]

# use a single k-partite entropy vector to generate entropy vectors of
#   k-partitions of an n-partite system
def get_vecs(subsystem_text, vec, known_permutations = None):
    vec_subs = len(vec.subs) # number of subsystems in entropy vector
    subsystems = get_subsystems(subsystem_text) # identify subsystems from given text
    if len(subsystems) < vec_subs: return []

    if known_permutations == None:
        # if no permutations were given, collect all permutations of vec_subs elements
        permutations = list(itertools.permutations(range(vec_subs)))
    else:
        # otherwise, use only the given permutations
        permutations = known_permutations.copy()

    partition_vecs = [] # collect entropy vectors for all partitions
    for partition in get_partitions(subsystems):
        if len(partition) != vec_subs: continue
        new_subsystems = [ "".join(sorted(parts)) for parts in partition ]
        for permutation in permutations:
            new_subsystems_permuted = [ new_subsystems[permutation[jj]]
                                        for jj in range(vec_subs) ]
            partition_vecs.append(vec.relabeled(vec.subs, new_subsystems_permuted))
    output_vecs = []
    for vec in partition_vecs:
        if vec not in output_vecs:
            output_vecs.append(vec)
    return output_vecs

subadditivity = e_vec([ ("A",1), ("B",1), ("AB",-1) ])
subadditivity_perms = [ [0,1] ]

SSA_1 = e_vec([ ("AB",1), ("BC",1), ("B",-1), ("ABC",-1) ])
SSA_1_perms = [ [0,1,2], [1,2,0], [2,0,1] ]

SSA_2 = e_vec([ ("AB",1), ("BC",1), ("A",-1), ("C",-1) ])
SSA_2_perms = [ [0,1,2], [1,2,0], [2,0,1] ]

MMI = e_vec([ ("AB",1), ("AC",1), ("BC",1), ("A",-1), ("B",-1), ("C",-1), ("ABC",-1) ])
MMI_perms = [ [0,1,2] ]

vecs = get_vecs("ABC", subadditivity)
for vec in vecs:
    print(vec.split())

