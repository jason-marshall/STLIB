# Driver for testing the feasability of the sweep method of computing
# the shortest path on a graph.

import sys
import string
import random

import graph_generator
from Graph import *

#
# Make the graph.
#

# Make the edges for a 10x10 rectangular grid graph.
num_vertices, edges = graph_generator.rectangular_grid( 50, 50, lambda : 1 + 9 * random.random() ) 

# Make a graph for my algorithm.
graph = Graph()
graph.make( num_vertices, edges )

# Make a graph for Dijkstra's algorithm.
graph_dijkstra = Graph()
graph_dijkstra.make( num_vertices, edges )

#
# Compute the shortest path tree with vertex 0 as the source.
#

print "determined fraction = ", graph.marching_with_correctness_criterion(0,7)
print "number of iterations = ", graph_dijkstra.dijkstra( 0 )

if graph == graph_dijkstra:
    print "The two algorithms agree."
else:
    print "The two algorithms do not agree."

#
# Display the result.
#

#graph.display()
#graph_dijkstra.display()
