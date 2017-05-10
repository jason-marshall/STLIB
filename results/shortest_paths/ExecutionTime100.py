# ExecutionTime.py

import sys, os, math

edge_weight_ratios = [100]
edge_weight_ratio_strings = ['100']

#
# Do the testing for grid graphs.
#

print "GRID GRAPHS:"

x_sizes = [10, 20, 40, 80, 160, 320, 640]

for i in range(1):
    print 'edge weight ratio = ', edge_weight_ratio_strings[i]
    file = open('../results/ExecutionTimeGrid'
                + edge_weight_ratio_strings[i] + '.dat', 'w')
    file.write('# ExecutionTimeGrid' + edge_weight_ratio_strings[i] + '.dat\n')
    file.write('# <# vertices> <Dijkstra> <MCC 1> <MCC 3>\n')
    for xs in x_sizes:
        print 'grid size = ', xs, 'x', xs
        file.write( "%i " % (xs*xs) )

        # Dijkstra
        times = []
        for dummy in range(3):
            output = os.popen( '../time/GraphDijkstra.rectangular '
                               "%i %i %i" % (xs, xs, edge_weight_ratios[i])
                               ).readlines()
            t = eval( output[-1][:-1] )
            times.append( t )
            os.system( "sleep %i" % math.ceil( t / 2 ) )
        file.write( "\t%f " % min( times ) )
        
        # MCC level 1
        times = []
        for dummy in range(3):
            output = os.popen( '../time/GraphMCCSimple.rectangular '
                               "%i %i %i" % (xs, xs, edge_weight_ratios[i])
                               ).readlines()
            t = eval( output[-1][:-1] )
            times.append( t )
            os.system( "sleep %i" % math.ceil( t / 2 ) )
        file.write( "\t%f " % min( times ) )
        
        # MCC level 3
        times = []
        for dummy in range(3):
            output = os.popen( '../time/GraphMCC.rectangular '
                               "%i %i %i" % (xs, xs, edge_weight_ratios[i])
                               ).readlines()
            t = eval( output[-1][:-1] )
            times.append( t )
            os.system( "sleep %i" % math.ceil( t / 2 ) )
        file.write( "\t%f\n" % min( times ) )
        
    file.close()

#
# Do the testing for sparse graphs.
#

print "\nRANDOM GRAPHS: 4 edges"

sizes = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400,
         204800, 409600]

for i in range(1):
    print 'edge weight ratio = ', edge_weight_ratio_strings[i]
    file = open('../results/ExecutionTimeRandom4Edges'
                + edge_weight_ratio_strings[i] + '.dat', 'w')
    file.write('# ExecutionTimeRandom4Edges' + edge_weight_ratio_strings[i]
               + '.dat\n')
    file.write('# <# vertices> <Dijkstra> <MCC 1> <MCC 3>\n')
    for size in sizes:
        print 'grid size = ', size
        file.write( "%i " % size )

        # Dijkstra
        times = []
        for dummy in range(3):
            output = os.popen( '../time/GraphDijkstra.random '
                               "%i 4 %i" % (size, edge_weight_ratios[i])
                               ).readlines()
            t = eval( output[-1][:-1] )
            times.append( t )
            os.system( "sleep %i" % math.ceil( t / 2 ) )
        file.write( "\t%f " % min( times ) )
        
        # MCC level 1
        times = []
        for dummy in range(3):
            output = os.popen( '../time/GraphMCCSimple.random '
                               "%i 4 %i" % (size, edge_weight_ratios[i])
                               ).readlines()
            t = eval( output[-1][:-1] )
            times.append( t )
            os.system( "sleep %i" % math.ceil( t / 2 ) )
        file.write( "\t%f " % min( times ) )
        
        # MCC level 3
        times = []
        for dummy in range(3):
            output = os.popen( '../time/GraphMCC.random '
                               "%i 4 %i" % (size, edge_weight_ratios[i])
                               ).readlines()
            t = eval( output[-1][:-1] )
            times.append( t )
            os.system( "sleep %i" % math.ceil( t / 2 ) )
        file.write( "\t%f\n" % min( times ) )
        
    file.close()

print "\nRANDOM GRAPHS: 32 edges"

sizes = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400]

for i in range(1):
    print 'edge weight ratio = ', edge_weight_ratio_strings[i]
    file = open('../results/ExecutionTimeRandom32Edges'
                + edge_weight_ratio_strings[i] + '.dat', 'w')
    file.write('# ExecutionTimeRandom32Edges' + edge_weight_ratio_strings[i]
               + '.dat\n')
    file.write('# <# vertices> <Dijkstra> <MCC 1> <MCC 3>\n')
    for size in sizes:
        print 'grid size = ', size
        file.write( "%i " % size )

        # Dijkstra
        times = []
        for dummy in range(3):
            output = os.popen( '../time/GraphDijkstra.random '
                               "%i 32 %i" % (size, edge_weight_ratios[i])
                               ).readlines()
            t = eval( output[-1][:-1] )
            times.append( t )
            os.system( "sleep %i" % math.ceil( t / 2 ) )
        file.write( "\t%f " % min( times ) )
        
        # MCC level 1
        times = []
        for dummy in range(3):
            output = os.popen( '../time/GraphMCCSimple.random '
                               "%i 32 %i" % (size, edge_weight_ratios[i])
                               ).readlines()
            t = eval( output[-1][:-1] )
            times.append( t )
            os.system( "sleep %i" % math.ceil( t / 2 ) )
        file.write( "\t%f " % min( times ) )
        
        # MCC level 3
        times = []
        for dummy in range(3):
            output = os.popen( '../time/GraphMCC.random '
                               "%i 32 %i" % (size, edge_weight_ratios[i])
                               ).readlines()
            t = eval( output[-1][:-1] )
            times.append( t )
            os.system( "sleep %i" % math.ceil( t / 2 ) )
        file.write( "\t%f\n" % min( times ) )
        
    file.close()

#
# Do the testing for complete graphs.
#

print "\nCOMPLETE GRAPHS:"

sizes = [10, 20, 40, 80, 160, 320, 640, 1280]

for i in range(1):
    print 'edge weight ratio = ', edge_weight_ratio_strings[i]
    file = open('../results/ExecutionTimeComplete'
                + edge_weight_ratio_strings[i] + '.dat', 'w')
    file.write('# ExecutionTimeComplete' + edge_weight_ratio_strings[i]
               + '.dat\n')
    file.write('# <# vertices> <Dijkstra> <MCC 1> <MCC 3>\n')
    for size in sizes:
        print 'grid size = ', size
        file.write( "%i " % size )

        # Dijkstra
        times = []
        for dummy in range(3):
            output = os.popen( '../time/GraphDijkstra.dense '
                               "%i %i" % (size, edge_weight_ratios[i])
                               ).readlines()
            t = eval( output[-1][:-1] )
            times.append( t )
            os.system( "sleep %i" % math.ceil( t / 2 ) )
        file.write( "\t%f " % min( times ) )
        
        # MCC level 1
        times = []
        for dummy in range(3):
            output = os.popen( '../time/GraphMCCSimple.dense '
                               "%i %i" % (size, edge_weight_ratios[i])
                               ).readlines()
            t = eval( output[-1][:-1] )
            times.append( t )
            os.system( "sleep %i" % math.ceil( t / 2 ) )
        file.write( "\t%f " % min( times ) )
        
        # MCC level 3
        times = []
        for dummy in range(3):
            output = os.popen( '../time/GraphMCC.dense '
                               "%i %i" % (size, edge_weight_ratios[i])
                               ).readlines()
            t = eval( output[-1][:-1] )
            times.append( t )
            os.system( "sleep %i" % math.ceil( t / 2 ) )
        file.write( "\t%f\n" % min( times ) )
        
    file.close()
