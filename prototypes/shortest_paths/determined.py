"""Driver for computing what fraction of the vertices are determined.

Solve the shortest-path problem on a graph using Dijkstra's algorithm.
At each iteration of the algoritm, calculate the fraction of vertices
in the labeled band that are determined from currently known values."""

#import sys
#import string
import random

import graph_generator
from Graph import *

print "Testing for ideal and actual determined fraction of vertices."
print "Using the simple correctness criterion.\n"

# The edge weight function.
edge_weight = lambda : random.random()
print "Ratio of largest to smallest edge weight = infinity.\n"

#
# Do the testing for grid graphs.
#

print "GRID GRAPHS:"
for size in range( 20, 101, 20 ):

    # Make the edges for a rectangular grid graph.
    num_vertices, edges =\
    graph_generator.rectangular_grid( size, size, edge_weight )

    # Make a graph from the edges.
    graph = Graph()
    graph.make( num_vertices, edges )

    # Compute the shortest path tree with vertex 0 as the source.
    print "\nFor the", size, "x", size, "rectangular grid graph:",\
          len( graph.vertices ), "vertices,",\
          len( graph.edges ), "edges."
    ideal, actual = graph.determined_vertices( 0 )
    print "ideal fraction = ", ideal, " actual fraction = ", actual

#
# Do the testing for sparse graphs.
#

print "\n\nSPARSE GRAPHS:"
for size in range( 200, 1001, 200 ):

    # Make the edges for a sparse graph.
    num_vertices, edges =\
    graph_generator.sparse( size, 4, edge_weight )

    # Make a graph from the edges.
    graph = Graph()
    graph.make( num_vertices, edges )

    # Compute the shortest path tree with vertex 0 as the source.
    print "\nFor the sparse graph:",\
          len( graph.vertices ), "vertices,",\
          len( graph.edges ), "edges."
    ideal, actual = graph.determined_vertices( 0 )
    print "ideal fraction = ", ideal, " actual fraction = ", actual

#
# Do the testing for complete graphs.
#

print "\n\nCOMPLETE GRAPHS:"
for size in range( 50, 251, 50 ):

    # Make the edges for a complete graph.
    num_vertices, edges =\
    graph_generator.complete( size, edge_weight )

    # Make a graph from the edges.
    graph = Graph()
    graph.make( num_vertices, edges )

    # Compute the shortest path tree with vertex 0 as the source.
    print "\nFor the complete graph:",\
          len( graph.vertices ), "vertices,",\
          len( graph.edges ), "edges."
    ideal, actual = graph.determined_vertices( 0 )
    print "ideal fraction = ", ideal, " actual fraction = ", actual

