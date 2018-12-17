#!/usr/bin/env python2

import itertools

# split a system into its primary subsystems
def primary_subsystems(system):
    systems = []
    start_idx = 0
    while start_idx < len(system):
        if system[start_idx] == " ":
            start_idx += 1
            continue
        end_idx = start_idx + 1
        while end_idx < len(system) and not system[end_idx].isalpha():
            end_idx += 1
        systems.append(system[start_idx:end_idx])
        start_idx = end_idx
    return systems

# return power set of a collection of objects
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
        if len(args) == 1: # args is a list of pairs: (subsystem, coefficient)
            self.subs, self.vals = zip(*args[0])
        elif len(args) == 2:
            if len(args[0]) == len(args[1]):
                # args is two ordered lists: subsystems, coefficients
                self.subs, self.vals = args
            else:
                # args[0] is an entropy vector in standard form
                # args[1] defines the primary systems
                self.vals, self.systems = args
                if type(self.systems) is str:
                    self.systems = primary_subsystems(self.systems)
                self.subs = list(power_set(self.systems))[1:]
        elif len(args) == 3:
            # args[0] is an ordered list of primary systems
            # args[1] is an ordered list of subsystems
            # args[2] is an ordered list of coefficients
            self.systems, self.subs, self.vals = args
        elif len(args) == 0: # empty vector
            self.systems, self.subs, self.vals = [], [], []
            return
        else:
            print("invalid initialization of an entropy vector!")
            exit()

        if len(self.subs) == 0:
            self.systems = []
            return

        # if necessary, properly identify primary systems and subsystems
        if len(args) in [ 1, 2 ]:
            self.subs = list(self.subs)
            self.systems = sorted(list(set.union(*[ set(sub) for sub in self.subs ])))

        self.sort()
        self.simplify()

    # sort vector subsystems and coefficients
    def sort(self):
        # sort the systems within each subsystem
        self.subs = [ sorted(sub, key = lambda sys : self.systems.index(sys))
                       for sub in self.subs ]
        # sort the list of subsystems and corresponding values
        def subsystem_sort_key(sub):
            return tuple([len(sub)] + [self.systems.index(sys) for sys in sub])
        zipped_vec = sorted(zip(self.subs, self.vals),
                            key = lambda x : subsystem_sort_key(x[0]))
        self.subs, self.vals = map(list,zip(*zipped_vec))

    # simplify entropy vector
    def simplify(self):
        # combine identical subsystems
        text_subs = [ "".join(sub) for sub in self.subs ]
        for sub in set(text_subs):
            indices = [ jj for jj in range(len(text_subs)) if text_subs[jj] == sub ]
            if len(indices) > 1:
                self.vals[indices[0]] = sum([ self.vals[jj] for jj in indices ])
                for jj in indices[1:][::-1]:
                    del self.vals[jj]
                    del self.subs[jj]
                    del text_subs[jj]

        # remove zero entries from this vector
        zero_indices = [ jj for jj in range(len(self.vals)) if self.vals[jj] == 0 ]
        for idx in zero_indices[::-1]:
            del self.subs[idx]
            del self.vals[idx]

    # return new entropy vector with relabeled systems
    # note: always splits resulting vector
    def relabeled(self, *args):
        args = list(args)
        if len(args) == 1:
            # args is a single list of new subsystem names
            old_systems, new_systems = self.systems, args[0]
        elif len(args) == 2 and type(args[0]) == list:
            # args is two lists: old subsystems and new ones
            old_systems, new_systems = args
        else:
            # args is two strings: an old subsystem and its new name
            old_systems, new_systems = [args[0]], [args[1]]

        def relabel(system):
            if system in old_systems:
                return new_systems[old_systems.index(system)]
            else: return system

        systems = [ relabel(sys) for sys in self.systems ]
        subsystems = [ [ relabel(sys) for sys in sub ]
                       for sub in self.subs ]
        return e_vec(systems, subsystems, self.vals).split()

    # return new entropy vector with split subsystems
    def split(self):
        return e_vec([ "".join(sub) for sub in self.subs ], self.vals)

    # returns a full vector in an ordered basis
    def standard_form(self, systems = None):
        if systems == None: systems = self.systems
        subsystems = list(power_set(systems))[1:]
        if any(sub not in subsystems for sub in self.subs):
            return self.split().standard_form(systems)
        vec = [ 0 ] * len(subsystems)
        for sub, val in zip(self.subs, self.vals):
            vec[subsystems.index(sub)] = val
        return vec

    # return "purification dual" with respect to a given primary system
    def purification_dual(self, system):
        systems = set(self.systems)
        def trim(subsystem):
            return set(subsystem).difference(set([system]))
        def dual_subsystem(subsystem):
            return sorted(list(systems.difference(trim(subsystem))))
        new_subsystems = [ sub if system not in sub else dual_subsystem(sub)
                           for sub in self.subs ]
        return e_vec(self.systems, new_subsystems, self.vals)

    # evaluate symbolic entries according to dictionary
    def evaluated(self, dict):
        vals = [ val if isinstance(val, (int)) else int(val.subs(dict))
                 for val in self.vals ]
        return e_vec(self.systems, self.subs, vals)

    # reduced "reduced" vector obtained by making some subsystem trivial
    def reduce(self, trivial_subsystem):
        if type(trivial_subsystem) == str:
            trivial_subsystem = primary_subsystems(trivial_subsystem)
        systems = [ sys for sys in self.systems if sys not in trivial_subsystem ]
        subsystems = [ [ sys for sys in part if sys not in trivial_subsystem ]
                       for part in self.subs ]
        if all( len(sub) == 0 for sub in subsystems ): return e_vec()
        subsystems, vals = zip(*[ (sub, val) for sub, val in zip(subsystems, self.vals)
                                  if len(sub) > 0 ])
        return e_vec(systems, subsystems, vals)

    # return human-readable text identifying this entropy vector
    def __repr__(self):
        signs = [ "+" if val > 0 else "-" for val in self.vals ]
        abs_vals = [ "" if val in [1,-1]
                     else ( " {}".format(val) if val > 0 else " {}".format(-val) )
                     for val in self.vals]
        text_subs = [ "".join(sub) for sub in self.subs ]
        return " ".join([ "{}{} ({})".format(sign,abs_val,sub)
                          for sign,abs_val,sub in zip(signs,abs_vals,text_subs) ])

    def __hash__(self):
        return hash(self.split().__repr__())

    # relations and operations
    def __eq__(self, other): # comparison x == y
        return self.__hash__() == other.__hash__()
    def __ne__(self, other): # comparison x != y
        return not (self == other)
    def __pos__(self): return self # unary "+"
    def __neg__(self): # unary "-", i.e. negation
        return e_vec(self.systems, self.subs, [ -val for val in self.vals ])
    def __add__(self, other): # addition of vectors: x + y
        left, right = self.split(), other.split()
        systems = sorted(list(set(left.systems + right.systems)))
        subs = left.subs + right.subs
        vals = left.vals + right.vals
        new_vec = e_vec(systems, subs, vals)
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

# get all vectors acquired by subsystem permutation, purification duals, and reductions
def get_seeded_vecs(seed_vec):
    # get all unique permutations of the seed vector
    permutation_vecs = set( seed_vec.relabeled(permutation)
                            for permutation in itertools.permutations(seed_vec.systems) )

    # get all purification duals of all permuted vectors
    purification_vecs = set( vec.purification_dual(sys)
                             for vec in permutation_vecs
                             for sys in vec.systems )

    # collect all vectors obtained by additionally making subsystems trivial
    seeded_vecs = set( vec.reduce(subsystem)
                       for vec in set.union(permutation_vecs, purification_vecs)
                       for subsystem in power_set(vec.systems) )
    return set([ vec for vec in seeded_vecs if len(vec.vals) > 1 ])

# given a single positive entropy vector, generate all implied positive entropy vectors
# for a given n-partite system
def get_super_vectors(systems, seed_vector):
    if type(systems) is str:
        systems = primary_subsystems(systems)
    seeded_vectors = get_seeded_vecs(seed_vector)

    system_vecs = set() # keep track of entropy vectors for this system

    # the seeded vectors address k-partite systems for k >= 2; our system is n-partite,
    # so for each size p from 2 to n
    for size in range(2, len(systems)+1):
        # consider all subsystems of size p (there are n choose p such subsystems)
        for subsystem in itertools.combinations(systems, size):
            # for each of these subsystems, consider all k-partitions
            for partition in get_all_partitions(list(subsystem)):
                # generate all seeded k-partite entropy vectors on this partition
                new_systems = [ "".join(sorted(parts)) for parts in partition ]
                system_vecs.update([ vec.relabeled(new_systems)
                                     for vec in seeded_vectors
                                     if len(vec.systems) == len(new_systems) ])

    return set([ vec for vec in system_vecs if len(vec.vals) > 1 ])
