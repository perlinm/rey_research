#!/usr/bin/env python2

import itertools, sage

sage_cone = sage.geometry.cone.Cone
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# return power set of a collection of objects
def power_set(collection):
    if type(collection) == str:
        collection = primary_subsystems(collection)
    power_set = itertools.chain.from_iterable(itertools.combinations(collection, nn)
                                              for nn in range(len(collection)+1))
    return ( set(subset) for subset in power_set )

# return iterator over all partitions of a collection of items
def get_all_partitions(collection):
    collection = collection
    if len(collection) == 1:
        yield [ set(collection) ]
        return
    first = collection[0]
    for smaller in get_all_partitions(collection[1:]):
        smaller = sorted(list(smaller))
        for n, subset in enumerate(smaller):
            subset = list(subset)
            yield [ set(x) for x in smaller[:n] ] \
                + [set([ first ] + subset)]  \
                + [ set(x) for x in smaller[n+1:] ]
        yield [ set([ first ]) ] + [ set(x) for x in smaller ]

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
    return set(systems)

# return subsystem in text form
def subsystem_text(subsystem):
    return ",".join([ str(sys) for sys in sorted(subsystem) ])

# return index of subsystem
def subsystem_index(subsystem, systems):
    power_set_iterator = power_set(sorted(systems))
    power_set_iterator.next() # skip the empty set
    index = 0
    for item in power_set_iterator:
        if item == subsystem:
            return index
        index += 1
    return None

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
            self.systems = set(self.systems)
        elif len(args) == 0: # empty vector
            self.systems, self.subs, self.vals = set(), [], []
            return
        else:
            print("invalid initialization of an entropy vector!")
            exit()

        if len(self.subs) == 0:
            self.systems = set()
            return

        self.subs = [ set([ str(sys) for sys in sub ]) for sub in self.subs ]

        # if necessary, properly identify primary systems and subsystems
        if len(args) in [ 1, 2 ]:
            self.systems = set.union(*self.subs)

        self.systems = set( str(sys) for sys in self.systems )
        self.sort_subsystems()
        self.simplify()

    # sort vector subsystems and coefficients
    def sort_subsystems(self):
        if self.subs == []: return
        # sort the list of subsystems and corresponding values
        zipped_vec = sorted(zip(self.subs, self.vals),
                            key = lambda x : subsystem_index(x[0], self.systems))
        self.subs, self.vals = map(list,zip(*zipped_vec))

    # simplify entropy vector
    def simplify(self):
        # combine identical subsystems
        for sub_tuple in set( tuple(sorted(sub)) for sub in self.subs ):
            sub = set(sub_tuple)
            indices = [ jj for jj in range(len(self.subs)) if self.subs[jj] == sub ]
            if len(indices) > 1:
                self.vals[indices[0]] = sum([ self.vals[jj] for jj in indices ])
                for jj in indices[1:][::-1]:
                    del self.vals[jj]
                    del self.subs[jj]

        # remove zero entries from this vector
        zero_indices = [ jj for jj in range(len(self.vals)) if self.vals[jj] == 0 ]
        for idx in zero_indices[::-1]:
            del self.subs[idx]
            del self.vals[idx]

    # return new entropy vector with subsystems split into primary components
    def split(self):
        if self.systems == set(): return self
        def with_primary(subsystem):
            return set.union(*[ primary_subsystems(sys) for sys in subsystem ])
        systems = with_primary(self.systems)
        subs = [ with_primary(sub) for sub in self.subs ]
        return e_vec(systems, subs, self.vals)

    # return new entropy vector with relabeled systems
    def relabeled(self, *args, **kwargs):
        args = list(args)
        if len(args) == 1:
            # args is a single list of new subsystem names
            old_systems, new_systems = sorted(self.systems), args[0]
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

        systems = [ str(relabel(sys)) for sys in self.systems ]
        subsystems = [ [ str(relabel(sys)) for sys in sub ]
                       for sub in self.subs ]
        if kwargs.get("split") and kwargs["split"] == True:
            return e_vec(systems, subsystems, self.vals).split()
        else:
            return e_vec(systems, subsystems, self.vals)

    # return "purification dual" with respect to a given primary system
    def purification_dual(self, system):
        def trim(subsystem):
            return subsystem.difference(set([system]))
        def dual_subsystem(subsystem):
            return self.systems.difference(trim(subsystem))
        new_subsystems = [ sub if system not in sub else dual_subsystem(sub)
                           for sub in self.subs ]
        return e_vec(self.systems, new_subsystems, self.vals)

    # return "reduced" vector obtained by making some subsystem trivial
    def reduced(self, trivial_subsystem):
        if type(trivial_subsystem) == str:
            trivial_subsystem = primary_subsystems(trivial_subsystem)
        systems = [ sys for sys in self.systems if sys not in trivial_subsystem ]
        subsystems = [ [ sys for sys in part if sys not in trivial_subsystem ]
                       for part in self.subs ]
        if all( len(sub) == 0 for sub in subsystems ): return e_vec()
        subsystems, vals = zip(*[ (sub, val) for sub, val in zip(subsystems, self.vals)
                                  if len(sub) > 0 ])
        return e_vec(systems, subsystems, vals)

    # returns a full vector in an ordered basis
    def standard_form(self, systems = None):
        if systems == None: systems = self.systems
        systems = set( str(sys) for sys in systems )
        subsystems = list(power_set(systems))[1:]
        if any(sub not in subsystems for sub in self.subs):
            print("some subsystems are not in the power set of the primary systems:")
            print(self.subs)
            print(subsystems)
            exit()
        vec = [ 0 ] * len(subsystems)
        for sub, val in zip(self.subs, self.vals):
            vec[subsystems.index(sub)] = val
        return vec

    # evaluate symbolic entries according to dictionary
    def evaluated(self, dict):
        vals = [ val if isinstance(val, (int)) else int(val.subs(dict))
                 for val in self.vals ]
        return e_vec(self.systems, self.subs, vals)

    # return new entropy vector which forgets about all subsystems that are not addressed
    # and enforces standardized labels for remaining primary systems
    def stripped(self):
        used_systems = set( sys for sub in self.subs for sys in sub )
        vec = self
        for sys in vec.systems:
            if sys not in used_systems:
                vec = vec.reduced(sys)
        return vec.relabeled(alphabet[:len(used_systems)])

    # determine whether two entropy vectors are "equivalent",
    # i.e. equal up to (i) forgetting about subsystems that are not addressed
    #             and (ii) enforcing standardized labels on primary systems
    # note: equivalency is sensistive to lexicographic ordering of primary subsystem names,
    #       such that e.g. (A,B) - (A) is not equivalent to (A,B) - (B)
    def equivalent(self, other):
        return self.stripped() == other.stripped()

    # return human-readable text identifying this entropy vector
    def __repr__(self):
        signs = [ "+" if val > 0 else "-" for val in self.vals ]
        abs_vals = [ "" if val in [1,-1]
                     else ( " {}".format(val) if val > 0 else " {}".format(-val) )
                     for val in self.vals]
        text_subs = [ subsystem_text(sub) for sub in self.subs ]
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
        vec_L, vec_R = self.split(), other.split()
        systems = set.union(vec_L.systems, vec_R.systems)
        subs = vec_L.subs + vec_R.subs
        vals = vec_L.vals + vec_R.vals
        return e_vec(systems, subs, vals)
    def __sub__(self, other): return self + (-other) # subtraction of vectors: x - y

# given a set of entropy vectors on a system with a given partitioning,
# return the extreme rays of the minimal cone containing all vectors
def get_extreme_rays(rays, systems):
    cone = sage_cone([ ray.standard_form(systems) for ray in rays ])
    return set([ e_vec(ray, systems) for ray in cone.rays() ])

# get minimal entropy cone implied by positivity of a given entropy vector
# this entropy cone is determined by all positive quantities implied by
# permutation, purification duals, and reductions of the original positive vector
def get_minimal_cone(positive_vec):
    systems = positive_vec.systems

    # get all unique permutations of the seed vector
    permutation_vecs = set( positive_vec.relabeled(permutation)
                            for permutation in itertools.permutations(systems) )

    # get all purification duals of all permuted vectors
    purification_vecs = set( vec.purification_dual(sys)
                             for vec in permutation_vecs
                             for sys in vec.systems )

    # collect all vectors obtained by additionally making subsystems trivial
    positive_vecs = set( vec.reduced(subsystem)
                         for vec in set.union(permutation_vecs, purification_vecs)
                         for subsystem in power_set(vec.systems) )

    return get_extreme_rays(positive_vecs, systems)

# given a single positive entropy vector, generate all implied positive entropy vectors
# for a given n-partite system
def get_positive_vectors(systems, positive_vec):
    if type(systems) is str:
        systems = primary_subsystems(systems)

    # get minimal positive cone impliet by the given positive vector
    minimal_positive_cone = get_minimal_cone(positive_vec)

    # from the extreme rays of the positive cone, collect those that are "distinct"
    # i.e. equivalent up to (i) forgetting about subsystems which are not addressed,
    #                  and (ii) an order-preserving relabeling of primary subsystems
    distinct_rays = set( vec.stripped() for vec in minimal_positive_cone )

    # each ray above addressed k disjoint subsystems for some k; collect the set { k }
    subsystem_numbers = set([ len(ray.systems) for ray in distinct_rays ])

    # organize the distinct rays by the number of disjoint subsystems they address
    distinct_rays = { num : set([ ray for ray in distinct_rays
                                  if len(ray.systems) == num ])
                      for num in subsystem_numbers }

    system_vecs = set() # keep track of entropy vectors for this system
    positive_vecs = {} # keep track the positive entropy vectors for each p-partite system

    # any given extreme ray of the positive cone addresses some k-partite system,
    # but we generally may want to consider n-partite systems for n >= k,
    # so for each system size p we may need to consider
    for system_num in range(min(subsystem_numbers), len(systems)+1):
        positive_vecs[system_num] = { }
        # for every partition of p elements (i.e. every choice of disjoint subsystems)
        for subsystems in get_all_partitions(alphabet[:system_num]):
            subsystem_num = len(subsystems)
            # if we don't have any distinct rays addressing this subsystem number, skip it
            if subsystem_num not in subsystem_numbers: continue
            if positive_vecs[system_num].get(subsystem_num) is None:
                positive_vecs[system_num][subsystem_num] = set()
            new_systems = [ "".join(sorted(sub)) for sub in subsystems ]
            # construct all relabeled vectors and add them to the set of vectors we track
            relabeled_vecs = [ vec.relabeled(new_systems).split()
                               for vec in distinct_rays[subsystem_num] ]
            positive_vecs[system_num][subsystem_num].update(relabeled_vecs)

    # for each number of primary systems
    for system_num in positive_vecs.keys():
        # for each number of disjoint subsystems
        for subsystem_num in positive_vecs[system_num].keys():
            # find minimal cone containing all of these positive vectors
            cone = positive_vecs[system_num][subsystem_num] # THIS IS WRONG!!
            # cone = get_extreme_rays(positive_vecs[system_num][subsystem_num],
            #                         alphabet[:system_num])
            # for choice of system_num primary systems
            for chosen_systems in itertools.combinations(systems, system_num):
                # relabel the vectors we kept track of appropriately, and save them
                new_vecs = [ vec.relabeled(chosen_systems).split() for vec in cone ]
                system_vecs.update(new_vecs)

    return system_vecs
