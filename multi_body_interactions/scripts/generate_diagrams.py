#!/usr/bin/env python3

import sys
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import combinations, product

try:
    assert(len(sys.argv) == 3)
    particles = int(sys.argv[1])
    vertices = int(sys.argv[2])
except:
    print("usage: {} particles vertices [split]".format(sys.argv[0]))
    exit()

figsize = [ 4, 3 ]
params = { "text.usetex" : True,
           "text.latex.preamble" : [ r"\usepackage{dsfont}" ] }
plt.rcParams.update(params)
colors = [ c["color"] for c in mpl.rcParams['axes.prop_cycle'] ]

# generate all unique graphs for given number of particles and interaction vertices
def unique_graphs(particles, vertices):
    assert(particles > 1)
    assert(vertices > 0)

    start = r"$i_%i$" # label for initial particle vertices
    end = r"$f_%i$" # label for final particle vertices

    # make empty graph to use as a skeleton for other graphs
    skeleton = nx.DiGraph()
    for pp in range(particles): # add all exterior  vertices
        skeleton.add_node(start % pp, pos = (0,particles-pp), external = True)
        skeleton.add_node(end % pp, pos = (vertices+1,particles-pp), external = True)
    for vv in range(vertices): # add all internal vertices
        skeleton.add_node(vv, pos = (vertices/2,particles/2),
                          external = False, exchange = False)

    graphs = [] # list of all unique graphs
    pairs = combinations(range(particles),2) # all combinations of two particles

    # WLOG the first interaction is always between particles 0 and 1,
    #   so loop over all possible pair-wise interactions which follow
    for pair_list in product(pairs, repeat=vertices-1):

        # collapse pair_list into an ordered 1D list of particles which interact
        particle_list = [ pairs[ii] for pairs in pair_list for ii in range(2) ]

        # make sure all particles interact
        if not all([ pp in particle_list for pp in range(2,particles) ]): continue

        # WLOG particle n is the n-th particle to be involved
        first_position = [ particle_list.index(pp) for pp in range(2,particles) ]
        if not first_position == sorted(first_position): continue

        # full ordered list of pair-wise interactions
        interactions = [ (0,1) ] + list(pair_list)

        # keep track of particles' "current" location
        locations = [ start % pp for pp in range(particles) ]

        # keep track of whether we have made an arbitrary choice
        #   in selecting particles from a node for an interaction
        selected = [ False for pp in range(particles) ]

        graph = skeleton.copy()
        symmetry_factors = 0
        correct_selection_order = True # is convention for nuclear spin selection obeyed?
        for vv in range(vertices):
            p1, p2 = interactions[vv] # particles interacting at this vertex
            v1, v2 = locations[p1], locations[p2] # parent vertices for each particle
            if v1 == v2: # both particles are from the same vertex
                # add an appropriate edge with
                #   weight = (number of particles moving along the edge) = 2
                graph.add_edge(v1, vv, weight = 2)
            else:
                for pj, nj in ((p1,v1),(p2,v2)): # for particle and its parent vertex
                    # there are no selection concerns for external vertices
                    if type(nj) is not int: continue
                    # identify particles interacting at node nj
                    nj_p1, nj_p2 = interactions[nj]
                    # if only one particle is currently at the parent vertex,
                    #   we do not need to worry about selection order
                    if locations[nj_p1] != locations[nj_p2]: continue
                    # if two particles are currently at the parent vertex,
                    #   WLOG we take the higher-numbered particle
                    if pj != max(nj_p1,nj_p2):
                        correct_selection_order = False
                        break
                    # if either particle has previously been selected,
                    #   we must exclicitly flag this vertex,
                    #   as we can no longer simply WLOG take one particle;
                    #   we must now account for the possibility of a nuclear spin exchange
                    if selected[nj_p1] or selected[nj_p2]:
                        graph.node[nj]["exchange"] = True
                    else:
                        # if there was no nuclear spin exchange, pick up a symmetry factor
                        #   to account for the possibility of selecting either nucleus
                        symmetry_factors += 1
                    # we have made a selection of particles, so flag them
                    selected[nj_p1], selected[nj_p2] = True, True
                # if we did not select higher-numbered particles, skip this graph
                if not correct_selection_order: break
                graph.add_edge(v1, vv, weight = 1)
                graph.add_edge(v2, vv, weight = 1)
            # every time we address two particles not both coming from external vertices,
            #   there are two ways to do so, so we pick up a symmetry factor of 2
            if type(locations[p1]) is int and type(locations[p1]) is int:
                symmetry_factors += 1
            locations[p1] = vv
            locations[p2] = vv

        if not correct_selection_order: continue

        for pp in range(particles):
            graph.add_edge(locations[pp], end % pp, weight = 1)

        if not nx.is_weakly_connected(graph): continue

        # we get a factor of 1/2 for every node
        symmetry_factors -= (graph.number_of_nodes() - 2*particles)
        graph.graph["symmetries"] = symmetry_factors

        # add "empty" graph data which will be used when constructing diagrams
        for edge in graph.edges():
            graph.edges[edge]["ground"] = True
        graph.graph["copies"] = 1

        graphs.append(graph)

    return graphs

# generate all unique diagrams from a given graph
def unique_diagrams(graph):
    # number of internal vertices in graph
    vertices = len([ vv for vv in graph.nodes() if type(vv) is int ])

    # special case for one-vertex graphs
    if vertices == 1: return [ graph ]

    diagrams = [] # collection of all unique diagrams for this graph
    ground_edge_list = [] # list of boolean arrays identifying ground-state edges

    # loop over all combinations of intermediate ground states
    for ground_states in product((True,False), repeat=vertices-1):

        diagram = graph.copy()
        for edge in diagram.edges():
            # do not modify edges starting or terminating at external vertices
            if type(edge[0]) is not int or type(edge[1]) is not int: continue

            # if there are no ground states between the vertices of this edge,
            #   the entire edge corresponds to an excited state
            if not any(ground_states[edge[0]:edge[1]]):
                diagram.edges[edge]["ground"] = False

        # boolean array identifying whether internal edges are grounded
        ground_edges = [ d["ground"] for _,_,d in diagram.edges(data=True) ]

        # skip diagrams with no intermediate excited states
        if all(ground_edges): continue

        # if this is a repeat of a diagram we have already found,
        #   do not add it to this list of unique diagrams,
        #   but keep track of how many copies we have
        repeat = False
        for gg in range(len(ground_edge_list)):
            if ground_edges == ground_edge_list[gg]:
                repeat = True
                diagrams[gg].graph["copies"] += 1
                break
        if repeat: continue

        # todo: do not include diagrams with identical prefactors;
        #       keep track of copies instead

        ground_edge_list.append(ground_edges)
        diagrams.append(diagram)

    return diagrams

graphs = unique_graphs(particles,vertices)
print("unique graphs:",len(graphs))

diagrams = [ diagram for graph in graphs for diagram in unique_diagrams(graph) ]
print("unique diagrams:",len(diagrams))

display_graphs = diagrams if "split" in sys.argv else graphs

for diagram in display_graphs:
    # convert directed diagram to an undirected graph, otherwise figures look bad
    diagram = diagram.to_undirected()

    # make a figure to visualize this diagram;
    # identify the symmetry factor in the title, in addition to the copies of this diagram
    plt.figure(figsize = figsize)
    plt.title(r"$\log_2 S={}$, $C={}$".format(diagram.graph["symmetries"],
                                              diagram.graph["copies"]))

    # identify initial/final particle vertices,
    #   normal internal interaction vertices,
    #   and internal interaction vertices with nuclear spin exchange
    external_vertices = [ n for n,d in diagram.nodes(data=True)
                       if d["external"] == True ]
    internal_vertices = [ n for n,d in diagram.nodes(data=True)
                       if d["external"] == False and d["exchange"] == False ]
    exchange_vertices = [ n for n,d in diagram.nodes(data=True)
                       if d["external"] == False and d["exchange"] == True ]

    # identify and draw edges starting or terminating at an external vertex
    external_edges = [ (u,v) for u,v in diagram.edges() if u in external_vertices ]
    ground_edges = [ (u,v) for u,v,d in diagram.edges(data=True)
                     if u not in external_vertices
                     and d["ground"] == True and d["weight"] == 1 ]
    excited_edges = [ (u,v) for u,v,d in diagram.edges(data=True)
                      if u not in external_vertices
                      and d["ground"] == False and d["weight"] == 1 ]
    ground_double_edges = [ (u,v) for u,v,d in diagram.edges(data=True)
                            if u not in external_vertices
                            and d["ground"] == True and d["weight"] == 2 ]
    excited_double_edges = [ (u,v) for u,v,d in diagram.edges(data=True)
                             if u not in external_vertices
                             and d["ground"] == False and d["weight"] == 2 ]

    # draw all nodes, edges, and labels
    initial_positions = { v: d["pos"] for v,d in diagram.nodes(data=True) }
    pos = nx.spring_layout(diagram, pos = initial_positions)
    nx.draw_networkx_nodes(diagram, pos, nodelist = external_vertices,
                           node_color = colors[0])
    nx.draw_networkx_nodes(diagram, pos, nodelist = internal_vertices,
                           node_color = colors[1])
    nx.draw_networkx_nodes(diagram, pos, nodelist = exchange_vertices,
                           node_color = colors[2])
    nx.draw_networkx_edges(diagram, pos, edgelist = external_edges, width = 1)
    nx.draw_networkx_edges(diagram, pos, edgelist = ground_edges,
                           width = 1, style = "solid")
    nx.draw_networkx_edges(diagram, pos, edgelist = excited_edges,
                           width = 1, style = "dashed")
    nx.draw_networkx_edges(diagram, pos, edgelist = ground_double_edges,
                           width = 4, style = "solid")
    nx.draw_networkx_edges(diagram, pos, edgelist = excited_double_edges,
                           width = 4, style = "dashed")
    nx.draw_networkx_labels(diagram, pos)

    # clean up figure
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

plt.show()
