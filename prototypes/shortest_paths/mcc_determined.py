"""Driver for computing what fraction of the vertices are determined.

Solve the shortest-path problem on a graph using the MCC algorithm.
At each iteration of the algoritm, calculate the fraction of vertices
in the labeled band that are determined from currently known values."""

import sys
#import string
import random

import graph_generator
from Graph import *

print "Testing for determined fraction of vertices with the MCC algorithm."

# The edge weight functions.
edge_weight_2 = lambda : 0.5 + 0.5 * random.random()
edge_weight_10 = lambda : 0.1 + 0.9 * random.random()
edge_weight_100 = lambda : 0.01 + 0.99 * random.random()
edge_weight_inf = lambda : random.random()
edge_weights = (edge_weight_2, edge_weight_10, edge_weight_100,\
                edge_weight_inf)
edge_weight_ratios = ('2', '10', '100', 'Infinity')

#
# Do the testing for grid graphs.
#

print "GRID GRAPHS:"

x_sizes = [10, 20, 40, 80, 160]

for i in range(4):
    edge_weight = edge_weights[i]
    print 'edge weight ratio = ', edge_weight_ratios[i]
    file = open('results/MCCDeterminedGrid' + edge_weight_ratios[i] + '.dat',
                'w')
    file.write('# MCCDeterminedGrid' + edge_weight_ratios[i] + '.dat\n')
    file.write('# <# vertices> <ratio 0> ... <ratio 5> <ideal ratio>\n')
    for xs in x_sizes:
        number_of_times = max( 1, min( 1600 / (xs*xs), 10 ) )
        print 'grid size = ', xs, 'x', xs
        file.write( "%i" % (xs*xs) )
        mcc_ratio = [0]*6
        ideal_ratio = 0
        for n in range( number_of_times ):
            # Make the edges for a rectangular grid graph.
            num_vertices, edges =\
                          graph_generator.rectangular_grid( xs, xs,\
                                                            edge_weight )

            # Make a graph from the edges.
            graph = Graph()
            graph.make( num_vertices, edges )
            
            for level in range(6):
                # Compute the shortest path tree with vertex 0 as the source.
                mcc_ratio[level] += graph.marching_with_correctness_criterion(
                    0, level)
            ideal_ratio += graph.determined_vertices( 0 )
        for level in range(6):
            file.write( "\t%f" % (mcc_ratio[level] / number_of_times) )
        file.write( "\t%f\n" % (ideal_ratio / number_of_times) )
    file.close()

#
# Do the testing for sparse graphs.
#

print "\nRANDOM GRAPHS: 4 edges"

sizes = [100, 200, 400, 800, 1600]

for i in range(4):
    edge_weight = edge_weights[i]
    print 'edge weight ratio = ', edge_weight_ratios[i]
    file = open('results/MCCDeterminedRandom4Edges' + edge_weight_ratios[i]
                + '.dat', 'w')
    file.write('# MCCDeterminedRandom4Edges' + edge_weight_ratios[i]
               + '.dat\n')
    file.write('# <# vertices> <ratio 0> ... <ratio 5> <ideal ratio>\n')
    for size in sizes:
        number_of_times = max( 1, min( 800 / size, 10 ) )
        print 'number of vertices = ', size
        file.write( "%i" % size )
        mcc_ratio = [0]*6
        ideal_ratio = 0
        for n in range( number_of_times ):
            # Make the edges for a sparse graph.
            num_vertices, edges =\
                          graph_generator.sparse( size, 4, edge_weight )

            # Make a graph from the edges.
            graph = Graph()
            graph.make( num_vertices, edges )
            
            for level in range(6):
                # Compute the shortest path tree with vertex 0 as the source.
                mcc_ratio[level] += graph.marching_with_correctness_criterion(
                    0, level)
            ideal_ratio += graph.determined_vertices( 0 )
        for level in range(6):
            file.write( "\t%f" % (mcc_ratio[level] / number_of_times) )
        file.write( "\t%f\n" % (ideal_ratio / number_of_times) )
    file.close()


print "\nRANDOM GRAPHS: 32 edges"

for i in range(4):
    edge_weight = edge_weights[i]
    print 'edge weight ratio = ', edge_weight_ratios[i]
    file = open('results/MCCDeterminedRandom32Edges' + edge_weight_ratios[i]
                + '.dat', 'w')
    file.write('# MCCDeterminedRandom32Edges' + edge_weight_ratios[i]
               + '.dat\n')
    file.write('# <# vertices> <ratio 0> ... <ratio 5> <ideal ratio>\n')
    for size in sizes:
        number_of_times = max( 1, min( 400 / size, 10 ) )
        print 'number of vertices = ', size
        file.write( "%i" % size )
        mcc_ratio = [0]*6
        ideal_ratio = 0
        for n in range( number_of_times ):
            # Make the edges for a sparse graph.
            num_vertices, edges =\
                          graph_generator.sparse( size, 32, edge_weight )

            # Make a graph from the edges.
            graph = Graph()
            graph.make( num_vertices, edges )
            
            for level in range(6):
                # Compute the shortest path tree with vertex 0 as the source.
                mcc_ratio[level] += graph.marching_with_correctness_criterion(
                    0, level)
            ideal_ratio += graph.determined_vertices( 0 )
        for level in range(6):
            file.write( "\t%f" % (mcc_ratio[level] / number_of_times) )
        file.write( "\t%f\n" % (ideal_ratio / number_of_times) )
    file.close()

#
# Do the testing for complete graphs.
#

print "\nCOMPLETE GRAPHS:"

sizes = [10, 20, 40, 80, 160, 320, 640]

for i in range(4):
    edge_weight = edge_weights[i]
    print 'edge weight ratio = ', edge_weight_ratios[i]
    file = open('results/MCCDeterminedComplete' + edge_weight_ratios[i]
                + '.dat', 'w')
    file.write('# MCCDeterminedComplete' + edge_weight_ratios[i]
               + '.dat\n')
    file.write('# <# vertices> <ratio 0> ... <ratio 5> <ideal ratio>\n')
    for size in sizes:
        number_of_times = max( 1, min( 320 / size, 10 ) )
        print 'number of vertices = ', size
        file.write( "%i" % size )
        mcc_ratio = [0]*6
        ideal_ratio = 0
        for n in range( number_of_times ):
            # Make the edges for a complete graph.
            num_vertices, edges =\
                          graph_generator.complete( size, edge_weight )

            # Make a graph from the edges.
            graph = Graph()
            graph.make( num_vertices, edges )
            
            for level in range(6):
                # Compute the shortest path tree with vertex 0 as the source.
                mcc_ratio[level] += graph.marching_with_correctness_criterion(
                    0, level)
            ideal_ratio += graph.determined_vertices( 0 )
        for level in range(6):
            file.write( "\t%f" % (mcc_ratio[level] / number_of_times) )
        file.write( "\t%f\n" % (ideal_ratio / number_of_times) )
    file.close()


