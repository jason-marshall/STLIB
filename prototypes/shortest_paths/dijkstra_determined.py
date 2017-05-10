"""Driver for computing what fraction of the vertices are determined.

Solve the shortest-path problem on a graph using Dijkstra's algorithm.
At each iteration of the algoritm, calculate the fraction of vertices
in the labeled band that are determined from currently known values."""

import sys
#import string
import random

import graph_generator
from Graph import *

print "Testing for determined fraction of vertices with Dijkstra's algorithm."

# The edge weight functions.
edge_weight_2 = lambda : 0.5 + 0.5 * random.random()
edge_weight_10 = lambda : 0.1 + 0.9 * random.random()
edge_weight_100 = lambda : 0.01 + 0.99 * random.random()
edge_weight_inf = lambda : random.random()
edge_weights = (edge_weight_2, edge_weight_10, edge_weight_100,\
                edge_weight_inf)
edge_weight_ratios = ("2", "10", "100", "infinity")

#
# Do the testing for grid graphs.
#

print "GRID GRAPHS:"

file = open('results/DijkstraDeterminedGrid.dat', 'w')
file.write('# DijkstraDeterminedGrid.dat\n')
file.write('# <# vertices> <ratio 2> <ratio 10> <ratio 100> <ratio infinity>\n')
x_sizes = [10, 20, 40, 80, 160]

for xs in x_sizes:
    number_of_times = min( 25600 / (xs*xs), 10 )
    print xs
    file.write( "%i" % (xs*xs) )
    for i in range(4):
        edge_weight = edge_weights[i]
        ratio = 0
        for n in range( number_of_times ):
            # Make the edges for a rectangular grid graph.
            num_vertices, edges =\
                          graph_generator.rectangular_grid( xs, xs,\
                                                            edge_weight )

            # Make a graph from the edges.
            graph = Graph()
            graph.make( num_vertices, edges )

            # Compute the shortest path tree with vertex 0 as the source.
            ratio += graph.dijkstra_determined_vertices( 0 )
        file.write( "\t%g" % (ratio / number_of_times) )
    file.write( '\n' )
file.close()

#
# Do the testing for sparse graphs.
#

print "\nRANDOM GRAPHS: 4 edges"

file = open('results/DijkstraDeterminedRandom4.dat', 'w')
file.write('# DijkstraDeterminedRandom4.dat\n')
file.write('# <# vertices> <ratio 2> <ratio 10> <ratio 100> <ratio infinity>\n')
sizes = [100, 200, 400, 800, 1600]

for size in sizes:
    number_of_times = min( 1600 / size, 10 )
    print size
    file.write( "%i" % size )
    for i in range(4):
        edge_weight = edge_weights[i]
        ratio = 0
        for n in range( number_of_times ):
            # Make the edges for a sparse graph.
            num_vertices, edges =\
                          graph_generator.sparse( size, 4, edge_weight )

            # Make a graph from the edges.
            graph = Graph()
            graph.make( num_vertices, edges )

            # Compute the shortest path tree with vertex 0 as the source.
            ratio += graph.dijkstra_determined_vertices( 0 )
        file.write( "\t%g" % (ratio / number_of_times) )
    file.write( '\n' )
file.close()


print "\nRANDOM GRAPHS: 32 edges"

file = open('results/DijkstraDeterminedRandom32.dat', 'w')
file.write('# DijkstraDeterminedRandom32.dat\n')
file.write('# <# vertices> <ratio 2> <ratio 10> <ratio 100> <ratio infinity>\n')
sizes = [100, 200, 400, 800, 1600]

for size in sizes:
    number_of_times = min( 1600 / size, 10 )
    print size
    file.write( "%i" % size )
    for i in range(4):
        edge_weight = edge_weights[i]
        ratio = 0
        for n in range( number_of_times ):
            # Make the edges for a sparse graph.
            num_vertices, edges =\
                          graph_generator.sparse( size, 32, edge_weight )

            # Make a graph from the edges.
            graph = Graph()
            graph.make( num_vertices, edges )

            # Compute the shortest path tree with vertex 0 as the source.
            ratio += graph.dijkstra_determined_vertices( 0 )
        file.write( "\t%g" % (ratio / number_of_times) )
    file.write( '\n' )
file.close()

#
# Do the testing for complete graphs.
#

print "\nCOMPLETE GRAPHS:"

file = open('results/DijkstraDeterminedComplete.dat', 'w')
file.write('# DijkstraDeterminedComplete.dat\n')
file.write('# <# vertices> <ratio 2> <ratio 10> <ratio 100> <ratio infinity>\n')
sizes = [10, 20, 40, 80, 160, 320, 640]

for size in sizes:
    number_of_times = min( 640 / size, 10 )
    print size
    file.write( "%i" % size )
    for i in range(4):
        edge_weight = edge_weights[i]
        ratio = 0
        for n in range( number_of_times ):
            # Make the edges for a complete graph.
            num_vertices, edges =\
                          graph_generator.complete( size, edge_weight )

            # Make a graph from the edges.
            graph = Graph()
            graph.make( num_vertices, edges )

            # Compute the shortest path tree with vertex 0 as the source.
            ratio += graph.dijkstra_determined_vertices( 0 )
        file.write( "\t%g" % (ratio / number_of_times) )
    file.write( '\n' )
file.close()
